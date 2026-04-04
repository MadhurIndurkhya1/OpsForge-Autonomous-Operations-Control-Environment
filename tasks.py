"""
OpsForge Tasks
--------------
Three evaluation tasks of increasing difficulty.

Each task wraps OpsForgeEnv and optionally overrides config
or adds extra evaluation logic on top of the base environment.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from models import Action, Observation, Reward
from environment import OpsForgeEnv


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _most_urgent_ticket_id(obs: Observation) -> str:
    """Return the ticket ID with the highest urgency score."""
    if not obs.tickets:
        raise ValueError("No tickets available")
    return max(obs.tickets, key=lambda t: t.urgency_score).id


# ---------------------------------------------------------------------------
# Task 1 – Easy: Prioritize the correct ticket
# ---------------------------------------------------------------------------

class EasyTask:
    """
    The agent must always pick the MOST URGENT ticket.
    Resources are unlimited (clamped by env but replenished each step).

    Evaluation: fraction of steps where the agent picked the highest-urgency ticket.
    """

    name = "easy_prioritization"
    description = (
        "Pick the most urgent ticket each step. "
        "Resources are generous — focus purely on priority ordering."
    )

    def __init__(self):
        self.env = OpsForgeEnv(
            config={
                "max_steps": 10,
                "initial_engineers": 50,   # effectively unlimited
                "initial_budget": 50_000.0,
                "initial_tickets": 3,
                "new_tickets_per_step": 1,
            }
        )
        self._correct_picks = 0
        self._total_picks   = 0

    def reset(self) -> Observation:
        self._correct_picks = 0
        self._total_picks   = 0
        return self.env.reset()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        obs_before = self.env._build_obs()
        best_id    = _most_urgent_ticket_id(obs_before)

        obs, reward, done, info = self.env.step(action)

        self._total_picks += 1
        if action.ticket_id == best_id:
            self._correct_picks += 1

        info["correct_priority"] = action.ticket_id == best_id
        info["accuracy"]         = self._correct_picks / self._total_picks
        return obs, reward, done, info

    def evaluate(self) -> Dict:
        accuracy = self._correct_picks / max(1, self._total_picks)
        return {
            "task": self.name,
            "total_reward": self.env.state().total_reward,
            "priority_accuracy": round(accuracy, 3),
            "grade": "PASS" if accuracy >= 0.7 else "FAIL",
        }


# ---------------------------------------------------------------------------
# Task 2 – Medium: Efficient resource allocation
# ---------------------------------------------------------------------------

class MediumTask:
    """
    The agent must resolve tickets while minimising resource waste.
    Budget and engineers are constrained — over-spending is penalised.

    Evaluation: total reward (penalised for wasteful spending).
    """

    name = "medium_resource_efficiency"
    description = (
        "Resolve tickets efficiently. "
        "You have limited engineers and budget — avoid over-allocating."
    )

    def __init__(self):
        self.env = OpsForgeEnv(
            config={
                "max_steps": 15,
                "initial_engineers": 15,
                "initial_budget": 3_000.0,
                "initial_tickets": 4,
                "new_tickets_per_step": 1,
            }
        )

    def reset(self) -> Observation:
        return self.env.reset()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        return self.env.step(action)

    def evaluate(self) -> Dict:
        state = self.env.state()
        return {
            "task": self.name,
            "total_reward": round(state.total_reward, 2),
            "resolved":     len(state.resolved_tickets),
            "missed":       len(state.missed_deadlines),
            "grade": "PASS" if state.total_reward > 0 else "FAIL",
        }


# ---------------------------------------------------------------------------
# Task 3 – Hard: Full multi-step decision making
# ---------------------------------------------------------------------------

class HardTask:
    """
    Full episode with tight resources, many tickets, and strict deadlines.
    The agent must balance urgency, efficiency, and deadline management
    across 20 steps.

    Evaluation: composite score based on reward, resolved %, and missed deadlines.
    """

    name = "hard_full_episode"
    description = (
        "Full 20-step episode with tight resources and many tickets. "
        "Balance urgency, efficiency, and deadline management to maximise total reward."
    )

    def __init__(self):
        self.env = OpsForgeEnv(
            config={
                "max_steps": 20,
                "initial_engineers": 10,
                "initial_budget": 4_000.0,
                "initial_tickets": 6,
                "new_tickets_per_step": 2,
            }
        )

    def reset(self) -> Observation:
        return self.env.reset()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        return self.env.step(action)

    def evaluate(self) -> Dict:
        state   = self.env.state()
        resolved = len(state.resolved_tickets)
        missed   = len(state.missed_deadlines)
        total    = resolved + missed

        resolve_rate = resolved / max(1, total)
        composite    = state.total_reward - (missed * 50)

        return {
            "task":          self.name,
            "total_reward":  round(state.total_reward, 2),
            "composite":     round(composite, 2),
            "resolved":      resolved,
            "missed":        missed,
            "resolve_rate":  round(resolve_rate, 3),
            "grade":         "PASS" if composite > 0 and resolve_rate >= 0.5 else "FAIL",
        }
