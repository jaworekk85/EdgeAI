"""Capture reproducibility metadata without importing runtime packages."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from edge_ai.devices.profile import package_versions
from edge_ai.utils.serialization import nonempty, validate_types


@dataclass(frozen=True)
class Provenance:
    git_commit: str
    git_dirty: bool
    captured_at_utc: str
    runtime_versions: dict[str, str | None]

    def __post_init__(self) -> None:
        validate_types(self)
        if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", self.git_commit):
            raise ValueError("git_commit must be a full Git object id")
        timestamp = datetime.fromisoformat(self.captured_at_utc.replace("Z", "+00:00"))
        if timestamp.utcoffset() is None or timestamp.utcoffset().total_seconds() != 0:
            raise ValueError("provenance timestamp must be UTC")
        if not self.runtime_versions:
            raise ValueError("runtime_versions must include the selected runtime")
        for name in self.runtime_versions:
            nonempty(name, "runtime version name")


def capture_provenance(
    repository: Path, *, runtime_packages: tuple[str, ...] = (),
) -> Provenance:
    repository = repository.resolve(strict=True)

    def git(*arguments: str) -> str:
        # Per-command trust for the explicitly selected workspace, never global config.
        try:
            completed = subprocess.run(
                ["git", "-c", f"safe.directory={repository.as_posix()}", "-C", str(repository),
                 *arguments], check=True, capture_output=True, text=True, timeout=15,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError("Cannot capture Git provenance from this repository") from exc
        return completed.stdout.strip()

    versions = package_versions(("edge-ai-research", "PyYAML", *runtime_packages))
    versions["mock"] = "1"
    return Provenance(
        git_commit=git("rev-parse", "HEAD"),
        git_dirty=bool(git("status", "--porcelain", "--untracked-files=all")),
        captured_at_utc=datetime.now(timezone.utc).isoformat(), runtime_versions=versions,
    )
