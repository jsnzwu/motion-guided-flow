from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RuntimeState:
    epoch: int = 0
    step: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, object] = field(default_factory=dict)
    flags: dict[str, bool] = field(default_factory=dict)

