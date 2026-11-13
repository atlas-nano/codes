"""Small serializable records used by the standalone workflow.

These records deliberately avoid atomate2 classes, but they carry enough
metadata for a future adapter to create jobflow jobs without changing the
scientific workflow API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class WorkflowStep:
    name: str
    kind: str
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    command: list[str] | None = None
    depends_on: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WorkflowResult:
    workdir: str
    steps: list[WorkflowStep]
    artifacts: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "workdir": self.workdir,
            "steps": [step.to_dict() for step in self.steps],
            "artifacts": self.artifacts,
        }
