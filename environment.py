"""
OpsForge Environment
--------------------
OpenEnv-compliant environment simulating an operations control centre.

API
---
    env = OpsForgeEnv()
    obs = env.reset()
    obs, reward, done, info = env.step(action)
    state = env.state()
"""

from __future__ import annotations

import random
import uuid
from typing import Dict, List, Optional, Tuple

from models import (
    Action,
    Observation,
    Resources,
    Reward,
    State,
    Ticket,
)
from reward import (
    attempt_resolution,
    compute_reward,
    missed_deadline_penalty,
)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "max_steps": 20,
    "initial_engineers": 10,
    "initial_budget": 5_000.0,
    "initial_tickets": 5,
    "new_tickets_per_step": 1,   # how many new tickets arrive each step
}


class OpsForgeEnv:
    """
    Autonomous Operations Control Environment.

    Tickets arrive every step. The agent picks ONE ticket per step,
    assigns resources, and tries to resolve it. The episode ends when
    time_remaining reaches 0 or all tickets are resolved early.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self._state: State = State()
        self._tickets: List[Ticket] = []
        self._resources: Resources = Resources(
            engineers=self.config["initial_engineers"],
            budget=self.config["initial_budget"],
            time_remaining=self.config["max_steps"],
        )

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Reset environment to initial state and return first observation."""
        self._state = State()
        self._resources = Resources(
            engineers=self.config["initial_engineers"],
            budget=self.config["initial_budget"],
            time_remaining=self.config["max_steps"],
        )
        self._tickets = [
            Ticket.generate(self._new_id(), step=0)
            for _ in range(self.config["initial_tickets"])
        ]
        return self._build_obs()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        """
        Execute one action.

        Parameters
        ----------
        action : Action  — which ticket to work on and how many resources to use.

        Returns
        -------
        observation : Observation
        reward      : Reward
        done        : bool
        info        : dict  — diagnostic info
        """
        if self._state.done:
            raise RuntimeError("Episode is done. Call reset() first.")

        info: Dict = {}

        # --- Find the target ticket ---
        ticket = self._find_ticket(action.ticket_id)
        if ticket is None:
            # Invalid action — small penalty, skip
            penalty = Reward(score=-10.0, breakdown={"error": "ticket_not_found"})
            self._state.total_reward += penalty.score
            self._advance_step()
            return self._build_obs(), penalty, self._state.done, {"error": "ticket_not_found"}

        # --- Clamp resource usage to what is available ---
        engineers_used = min(action.assign_engineers, self._resources.engineers)
        budget_used    = min(action.spend_budget,    self._resources.budget)

        # --- Attempt resolution ---
        success = attempt_resolution()
        info["resolution_success"] = success

        reward = compute_reward(
            ticket=ticket,
            action=Action(
                ticket_id=action.ticket_id,
                assign_engineers=engineers_used,
                spend_budget=budget_used,
                priority=action.priority,
            ),
            resources_before={
                "engineers": self._resources.engineers,
                "budget": self._resources.budget,
            },
            success=success,
        )

        # --- Update ticket and resources ---
        if success:
            ticket.resolved = True
            self._state.resolved_tickets.append(ticket.id)

        self._resources.engineers -= engineers_used
        self._resources.budget    -= budget_used

        # --- Advance time, decay deadlines, spawn new tickets ---
        expired = self._advance_step()
        for t_id in expired:
            penalty_val = self._penalty_for_id(t_id)
            reward = Reward(
                score=reward.score + penalty_val,
                breakdown={**reward.breakdown, f"deadline_miss_{t_id}": round(penalty_val, 2)},
            )
            self._state.missed_deadlines.append(t_id)

        self._state.total_reward += reward.score
        info["total_reward_so_far"] = self._state.total_reward
        info["tickets_remaining"]   = len([t for t in self._tickets if not t.resolved])

        return self._build_obs(), reward, self._state.done, info

    def state(self) -> State:
        """Return the current internal state (read-only snapshot)."""
        return self._state.model_copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_obs(self) -> Observation:
        active = [t for t in self._tickets if not t.resolved]
        return Observation(
            tickets=active,
            resources=Resources(
                engineers=self._resources.engineers,
                budget=self._resources.budget,
                time_remaining=self._resources.time_remaining,
            ),
            step=self._state.step_count,
        )

    def _find_ticket(self, ticket_id: str) -> Optional[Ticket]:
        for t in self._tickets:
            if t.id == ticket_id and not t.resolved:
                return t
        return None

    def _penalty_for_id(self, ticket_id: str) -> float:
        for t in self._tickets:
            if t.id == ticket_id:
                return missed_deadline_penalty(t)
        return 0.0

    def _advance_step(self) -> List[str]:
        """
        Tick time forward by 1.
        - Decrement deadlines on all unresolved tickets.
        - Remove tickets whose deadline has expired (collect penalties).
        - Spawn new tickets.
        - Update step counter and time_remaining.
        Returns list of expired ticket IDs.
        """
        self._state.step_count         += 1
        self._resources.time_remaining -= 1

        expired_ids: List[str] = []
        surviving: List[Ticket] = []

        for t in self._tickets:
            if t.resolved:
                surviving.append(t)
                continue
            t.deadline -= 1
            if t.deadline <= 0:
                expired_ids.append(t.id)
            else:
                surviving.append(t)

        self._tickets = surviving

        # Spawn new tickets
        for _ in range(self.config["new_tickets_per_step"]):
            self._tickets.append(
                Ticket.generate(self._new_id(), step=self._state.step_count)
            )

        # Check done condition
        if self._resources.time_remaining <= 0:
            self._state.done = True

        return expired_ids

    @staticmethod
    def _new_id() -> str:
        return f"TKT-{uuid.uuid4().hex[:6].upper()}"
