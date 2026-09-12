"""Durable, machine-readable progress for reconstruction workers."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline.cli_support import save_json


PHASE1_STAGES = (
    "Video analysis",
    "Keyframe selection",
    "Feature extraction",
    "Frame matching",
    "Sparse reconstruction",
    "Quality report",
)


@dataclass(frozen=True)
class Progress:
    stage: str
    stage_index: int
    stage_count: int
    percent: float
    message: str
    status: str = "running"
    current: int | None = None
    total: int | None = None
    updated_at: str = ""


class ProgressWriter:
    """Atomically publish progress that the server and UI can trust."""

    def __init__(self, output: Path, stages: tuple[str, ...] = PHASE1_STAGES):
        self.output = Path(output)
        self.stages = stages
        self.last_stage = stages[0]

    def update(
        self,
        stage: str,
        fraction: float,
        message: str,
        *,
        current: int | None = None,
        total: int | None = None,
        status: str = "running",
        details: dict[str, Any] | None = None,
    ) -> None:
        if stage not in self.stages:
            raise ValueError(f"Unknown reconstruction stage: {stage}")
        self.last_stage = stage
        index = self.stages.index(stage)
        fraction = min(1.0, max(0.0, float(fraction)))
        percent = 100.0 * (index + fraction) / len(self.stages)
        data = asdict(Progress(
            stage=stage,
            stage_index=index,
            stage_count=len(self.stages),
            percent=round(percent, 2),
            message=message,
            status=status,
            current=current,
            total=total,
            updated_at=datetime.now(timezone.utc).isoformat(),
        ))
        data["stages"] = list(self.stages)
        if details:
            data["details"] = details
        save_json(self.output / "progress.json", data)

    def complete(self, message: str = "Sparse reconstruction ready") -> None:
        self.update(self.stages[-1], 1.0, message, status="complete")

    def fail(self, stage: str, message: str) -> None:
        self.update(stage, 0.0, message, status="failed")
