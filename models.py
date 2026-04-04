"""
OpsForge Models
---------------
All typed data structures used across the environment.

Uses Python dataclasses (stdlib) for zero external dependencies when
running locally. Pydantic is still listed in requirements.txt because
it is preferred in production / the inference script uses .model_dump().

Compatibility shim: .model_dump() is added to every dataclass so the
rest of the codebase can call it regardless of backend.
"""

from __future__ import annotations

import dataclasses
import random
from enum import Enum
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Mixin: makes dataclasses behave like Pydantic models (minimal surface)
# ---------------------------------------------------------------------------

class _ModelMixin:
    def model_dump(self) -> Dict[str, Any]:
        """Recursively convert to a JSON-serialisable dict."""
        return _to_dict(self)

    def model_copy(self):
        import copy
        return copy.deepcopy(self)


def _to_dict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, list):
        return [_to_dict(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, Enum):
        return obj.value
    return obj


# ---------------------------------------------------------------------------
# Ticket
# ---------------------------------------------------------------------------

class TicketType(str, Enum):
    BUG       = "bug"
    COMPLAINT = "complaint"
    FRAUD     = "fraud"
    REQUEST   = "request"


@dataclasses.dataclass
class Ticket(_ModelMixin):
    id: str
    type: TicketType
    severity: int           # 1 (low) - 5 (critical)
    deadline: int           # steps remaining
    customer_value: float   # revenue / importance score
    resolved: bool = False

    @property
    def urgency_score(self) -> float:
        """Higher severity + shorter deadline = more urgent."""
        return (self.severity * self.customer_value) / max(1, self.deadline)

    @classmethod
    def generate(cls, ticket_id: str, step: int = 0) -> "Ticket":
        t_type   = random.choice(list(TicketType))
        severity = random.randint(1, 5)
        if t_type in (TicketType.FRAUD, TicketType.BUG):
            severity = max(severity, random.randint(2, 5))
        return cls(
            id=ticket_id,
            type=t_type,
            severity=severity,
            deadline=random.randint(2, 8),
            customer_value=round(random.uniform(100, 10_000), 2),
        )


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Resources(_ModelMixin):
    engineers: int
    budget: float
    time_remaining: int


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Observation(_ModelMixin):
    tickets: List[Ticket]
    resources: Resources
    step: int = 0


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Action(_ModelMixin):
    ticket_id: str
    assign_engineers: int   # must be >= 1
    spend_budget: float     # must be >= 0
    priority: int           # 1-5 label chosen by the agent


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Reward(_ModelMixin):
    score: float
    breakdown: Dict[str, Any] = dataclasses.field(default_factory=dict)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class State(_ModelMixin):
    step_count: int = 0
    done: bool = False
    total_reward: float = 0.0
    resolved_tickets: List[str] = dataclasses.field(default_factory=list)
    missed_deadlines: List[str] = dataclasses.field(default_factory=list)
