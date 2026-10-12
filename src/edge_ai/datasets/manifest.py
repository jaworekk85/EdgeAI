"""Deterministic Imagenette selection with full 1000-class labels."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from random import Random

from edge_ai.datasets.preprocessing import PreprocessingSpec, preprocess_image
from edge_ai.utils.artifacts import sha256_file, validate_digest, workspace_file
from edge_ai.utils.serialization import positive, to_dict, validate_types

# Imagenette WordNet IDs mapped to the canonical ImageNet-1K output order.
IMAGENETTE_LABELS = {
    "n01440764": 0,
    "n02102040": 217,
    "n02979186": 482,
    "n03000684": 491,
    "n03028079": 497,
    "n03394916": 566,
    "n03417042": 569,
    "n03425413": 571,
    "n03445777": 574,
    "n03888257": 701,
}


@dataclass(frozen=True)
class ImageSample:
    path: str
    sha256: str
    label: int
    wnid: str

    def __post_init__(self) -> None:
        validate_types(self)
        validate_digest(self.sha256)
        if Path(self.path).is_absolute() or ".." in Path(self.path).parts:
            raise ValueError("image path must be workspace-relative")
        if IMAGENETTE_LABELS.get(self.wnid) != self.label:
            raise ValueError("Imagenette label is not its ImageNet-1K output index")


@dataclass(frozen=True)
class DatasetManifest:
    dataset: str
    split: str
    seed: int
    samples: tuple[ImageSample, ...]
    preprocessing: PreprocessingSpec
    archive_url: str
    archive_sha256: str
    archive_bytes: int

    def __post_init__(self) -> None:
        validate_types(self)
        positive(self.seed, "dataset seed", zero=True)
        validate_digest(self.archive_sha256)
        positive(self.archive_bytes, "archive bytes")
        if self.dataset != "imagenette2-160" or self.split != "val" or not self.samples:
            raise ValueError("reference requires Imagenette v2 160px validation samples")
        if len({sample.path for sample in self.samples}) != len(self.samples):
            raise ValueError("manifest contains duplicate images")

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(
                to_dict(self),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()

    def verify(self, workspace: Path) -> None:
        for sample in self.samples:
            if sha256_file(workspace_file(workspace, sample.path)) != sample.sha256:
                raise ValueError("image SHA-256 mismatch")

    def batch(self, workspace: Path, index: int):
        sample = self.samples[index % len(self.samples)]
        return preprocess_image(
            workspace_file(workspace, sample.path), self.preprocessing
        ), sample.label


def select_manifest(
    workspace: Path,
    image_root: Path,
    archive: Path,
    *,
    archive_url: str,
    seed: int,
    per_class: int,
    preprocessing: PreprocessingSpec,
) -> DatasetManifest:
    positive(per_class, "per_class")
    random = Random(seed)
    selected = []
    for wnid, label in sorted(IMAGENETTE_LABELS.items()):
        candidates = sorted((image_root / wnid).glob("*.JPEG"))
        if len(candidates) < per_class:
            raise ValueError(f"insufficient validation images in {wnid}")
        for path in sorted(random.sample(candidates, per_class)):
            selected.append(
                ImageSample(
                    path.resolve().relative_to(workspace.resolve()).as_posix(),
                    sha256_file(path),
                    label,
                    wnid,
                )
            )
    return DatasetManifest(
        "imagenette2-160",
        "val",
        seed,
        tuple(selected),
        preprocessing,
        archive_url,
        sha256_file(archive),
        archive.stat().st_size,
    )
