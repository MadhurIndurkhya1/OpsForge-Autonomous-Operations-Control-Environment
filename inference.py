"""
OpsForge Inference Script
-------------------------
Runs a full episode using an LLM (via OpenAI-compatible API) as the agent.

Environment variables
---------------------
API_BASE_URL   -- e.g. https://api.openai.com/v1  or a local endpoint
MODEL_NAME     -- e.g. gpt-4o-mini
HF_TOKEN       -- your API key / HuggingFace token (used as Bearer token)

Usage
-----
    python inference.py                   # LLM agent, hard task
    python inference.py --task easy       # easy | medium | hard
    python inference.py --no-llm          # heuristic agent (no API key needed)
    python inference.py --task medium --quiet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

# Bring opsforge into path when running from project root
sys.path.insert(0, os.path.dirname(__file__))

from models import Action, Observation
from tasks import EasyTask, HardTask, MediumTask

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")

SYSTEM_PROMPT = """
You are an autonomous operations controller managing a ticket queue.
Each step you receive a JSON observation containing:
  - tickets: list of open tickets (id, type, severity 1-5, deadline, customer_value)
  - resources: { engineers, budget, time_remaining }
  - step: current step number

Your job is to pick ONE ticket to work on and decide how many resources to spend.

Rules:
- assign_engineers must be >= 1 and <= available engineers
- spend_budget must be >= 0 and <= available budget
- Minimum resources needed roughly scale with severity:
    severity 1 -> 1 engineer, $50 budget
    severity 5 -> 5 engineers, $1000 budget
- Do NOT over-spend; waste is penalised.
- Prefer high-severity, high-customer_value tickets with short deadlines.

You MUST respond with ONLY valid JSON in this exact shape:
{
  "ticket_id": "<id of chosen ticket>",
  "assign_engineers": <integer>,
  "spend_budget": <float>,
  "priority": <integer 1-5>
}
No explanation. No markdown. Only the JSON object.
""".strip()


# ---------------------------------------------------------------------------
# LLM agent
# ---------------------------------------------------------------------------

class LLMAgent:
    def __init__(self):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package not installed. "
                "Run: pip install openai  -- or use --no-llm flag."
            )
        self.client = OpenAI(
            api_key=HF_TOKEN or "no-key",
            base_url=API_BASE_URL,
        )

    def decide(self, obs: Observation) -> Action:
        """Ask the LLM to choose an action given the current observation."""
        obs_dict = obs.model_dump()
        user_message = f"Current observation:\n{json.dumps(obs_dict, indent=2)}"

        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.2,
            max_tokens=200,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if the model wraps the JSON
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        data = json.loads(raw)
        return Action(**data)


# ---------------------------------------------------------------------------
# Heuristic agent (no API key required)
# ---------------------------------------------------------------------------

class HeuristicAgent:
    """Improved heuristic with urgency awareness"""

    def decide(self, obs: Observation) -> Action:
        if not obs.tickets:
            raise ValueError("No tickets to act on")

        # Step 1: urgent tickets first
        urgent_tickets = [t for t in obs.tickets if t.deadline <= 2]

        if urgent_tickets:
            ticket = max(urgent_tickets, key=lambda t: t.severity)
        else:
            ticket = max(
                obs.tickets,
                key=lambda t: (
                    -t.deadline,
                    t.severity,
                    t.customer_value
                )
            )

        # Step 2: controlled resource usage
        engineers = max(1, min(2, obs.resources.engineers))
        budget = min(200.0, obs.resources.budget)

        return Action(
            ticket_id=ticket.id,
            assign_engineers=engineers,
            spend_budget=budget,
            priority=ticket.severity,
        )

# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(task_name: str, use_llm: bool = True, verbose: bool = True) -> Dict:
    task_map = {"easy": EasyTask, "medium": MediumTask, "hard": HardTask}
    if task_name not in task_map:
        raise ValueError(f"Unknown task '{task_name}'. Choose from: {list(task_map)}")

    task = task_map[task_name]()

    # Select agent
    agent = None
    if use_llm:
        try:
            agent = LLMAgent()
            print(f"Using LLM agent  -> {MODEL_NAME} @ {API_BASE_URL}")
        except Exception as exc:
            print(f"[warn] LLM agent unavailable ({exc}). Falling back to heuristic.")

    if agent is None:
        agent = HeuristicAgent()
        if not use_llm:
            print("Using heuristic agent (--no-llm mode).")

    obs = task.reset()

    print(f"\n{'='*55}")
    print(f"  OpsForge  |  Task: {task_name.upper()}")
    print(f"{'='*55}\n")

    step         = 0
    total_reward = 0.0

    while True:
        if not obs.tickets:
            print(f"[step {step}] No tickets in queue — episode ending.")
            break

        if verbose:
            print(
                f"[step {step:02d}] Tickets: {len(obs.tickets):2d} | "
                f"Engineers: {obs.resources.engineers:3d} | "
                f"Budget: ${obs.resources.budget:>8,.0f} | "
                f"Time left: {obs.resources.time_remaining}"
            )

        # Agent decides
        try:
            action = agent.decide(obs)
        except Exception as exc:
            print(f"  [error] Agent failed: {exc}. Using heuristic fallback.")
            action = HeuristicAgent().decide(obs)

        if verbose:
            print(
                f"  --> ticket={action.ticket_id}  "
                f"engineers={action.assign_engineers}  "
                f"budget=${action.spend_budget:,.0f}  "
                f"priority={action.priority}"
            )

        obs, reward, done, info = task.step(action)
        total_reward += reward.score
        step += 1

        if verbose:
            success_str = "OK " if info.get("resolution_success") else "FAIL"
            print(
                f"  <-- [{success_str}] reward={reward.score:+8.2f} | "
                f"cumulative={total_reward:+,.2f}"
            )
            if "accuracy" in info:
                print(f"       Priority accuracy: {info['accuracy']:.1%}")

        if done:
            break

    print(f"\n{'='*55}")
    print(f"  Episode finished after {step} steps.")
    print(f"  Total reward : {total_reward:+,.2f}")
    print(f"{'='*55}\n")

    result = task.evaluate()
    print("Evaluation:")
    for k, v in result.items():
        print(f"  {k:<22} {v}")
    print()

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OpsForge inference runner")
    parser.add_argument("--task",   default="hard", choices=["easy", "medium", "hard"])
    parser.add_argument("--no-llm", action="store_true", help="Use heuristic agent (no API key)")
    parser.add_argument("--quiet",  action="store_true", help="Suppress per-step output")
    args = parser.parse_args()

    run_episode(
        task_name=args.task,
        use_llm=not args.no_llm,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
