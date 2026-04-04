"""
OpsForge Reward Function
------------------------
Computes partial-credit rewards based on ticket resolution outcomes.
"""

from __future__ import annotations

import random

from models import Action, Reward, Ticket

# Resolution cost thresholds (per severity level)
ENGINEER_COST = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
BUDGET_COST   = {1: 50, 2: 150, 3: 300, 4: 600, 5: 1000}

# 80 % chance a ticket is actually resolved when worked on
RESOLUTION_PROBABILITY = 0.80


def compute_reward(
    ticket: Ticket,
    action: Action,
    resources_before: dict,
    success: bool,
) -> Reward:
    """
    Returns a Reward with full breakdown.

    Positive signals
    ----------------
    * Resolving a ticket           → severity × customer_value × 0.1
    * Acting before deadline       → bonus proportional to time left

    Negative signals
    ----------------
    * Wasted engineers / budget    → small penalty per unit over minimum needed
    * Failure to resolve           → −severity × 5
    * Missed deadline              → −customer_value × 0.05  (applied elsewhere)
    """

    breakdown: dict = {}

    # --- 1. Base resolution reward ---
    if success:
        base = ticket.severity * ticket.customer_value * 0.1
        deadline_bonus = ticket.deadline * 2.0          # reward acting early
        resolution_score = base + deadline_bonus
    else:
        resolution_score = -(ticket.severity * 5)       # failure penalty

    breakdown["resolution_score"] = round(resolution_score, 2)

    # --- 2. Resource efficiency penalty ---
    min_engineers = ENGINEER_COST[ticket.severity]
    min_budget    = BUDGET_COST[ticket.severity]

    over_eng = max(0, action.assign_engineers - min_engineers)
    over_bud = max(0, action.spend_budget - min_budget)

    waste_penalty = -(over_eng * 2.0 + over_bud * 0.01)
    breakdown["waste_penalty"] = round(waste_penalty, 2)

    # --- 3. Total ---
    total = resolution_score + waste_penalty
    breakdown["total"] = round(total, 2)

    return Reward(score=round(total, 2), breakdown=breakdown)


def attempt_resolution() -> bool:
    """Simulate 80 % success rate."""
    return random.random() < RESOLUTION_PROBABILITY


def missed_deadline_penalty(ticket: Ticket) -> float:
    """Penalty applied when a ticket's deadline expires."""
    return -(ticket.customer_value * 0.05 * ticket.severity)
