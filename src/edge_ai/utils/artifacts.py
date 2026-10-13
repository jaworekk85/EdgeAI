"""Content identities and confined workspace files."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from edge_ai.utils.serialization import validate_types


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_digest(value: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("SHA-256 must contain 64 lowercase hexadecimal characters")


def workspace_file(workspace: Path, relative: str, *, private: bool = False) -> Path:
    if Path(relative).is_absolute():
        raise ValueError("asset paths must be workspace-relative")
    root = workspace.resolve(strict=True)
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("asset path escapes workspace or is not a file")
    if path.is_relative_to((root / "private").resolve()) and not private:
        raise ValueError("public asset path refers to private/")
    return path


@dataclass(frozen=True)
class ArtifactIdentity:
    path: str
    sha256: str
    role: Literal["checkpoint", "compressed_checkpoint", "onnx"]
    checkpoint_sha256: str
    export_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_types(self)
        if not self.path or Path(self.path).is_absolute() or ".." in Path(self.path).parts:
            raise ValueError("artifact must have a confined relative path")
        validate_digest(self.sha256)
        validate_digest(self.checkpoint_sha256)
        if self.role == "checkpoint" and self.sha256 != self.checkpoint_sha256:
            raise ValueError("checkpoint identity hashes disagree")
        if self.role == "onnx" and not self.export_metadata:
            raise ValueError("ONNX artifact requires export provenance")
        if self.role == "compressed_checkpoint" and not self.export_metadata.get(
            "compression_details"
        ):
            raise ValueError("derived checkpoint requires compression lineage")

    def verify(self, workspace: Path, *, private: bool = False) -> Path:
        path = workspace_file(workspace, self.path, private=private)
        if sha256_file(path) != self.sha256:
            raise ValueError("artifact SHA-256 mismatch")
        return path
