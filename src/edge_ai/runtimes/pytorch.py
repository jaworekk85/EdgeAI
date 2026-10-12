"""Synchronous CPU PyTorch inference with no imports at module load time."""

from pathlib import Path
from typing import Callable

from edge_ai.config import ExecutionSettings
from edge_ai.models.loading import load_pytorch_model
from edge_ai.models.variant import ModelVariant
from edge_ai.runtimes.base import MemoryReport
from edge_ai.utils.artifacts import ArtifactIdentity
from edge_ai.utils.serialization import positive


class PyTorchRuntime:
    def __init__(
        self,
        workspace: Path,
        artifact: ArtifactIdentity,
        settings: ExecutionSettings,
        *,
        loader: Callable = load_pytorch_model,
    ) -> None:
        self.workspace, self.artifact, self.settings = workspace, artifact, settings
        self.loader = loader
        self.model = self.variant = self._batch = None

    def load(self, variant: ModelVariant) -> None:
        import torch

        if self.model is not None:
            raise RuntimeError("release the loaded model first")
        if variant.runtime != "pytorch" or self.artifact.role != "checkpoint":
            raise ValueError("PyTorch requires a checkpoint artifact")
        if variant.weights_digest != self.artifact.checkpoint_sha256:
            raise ValueError("checkpoint identity differs from variant")
        path = self.artifact.verify(self.workspace)
        torch.set_num_threads(self.settings.intra_op_threads)
        if torch.get_num_interop_threads() != self.settings.inter_op_threads:
            torch.set_num_interop_threads(self.settings.inter_op_threads)
        self.model = self.loader(variant, path).cpu().eval()
        self.variant = variant

    def _require_loaded(self) -> None:
        if self.model is None:
            raise RuntimeError("no model loaded")

    def prepare_batch(self, batch):
        import numpy as np
        import torch

        self._require_loaded()
        if not isinstance(batch, np.ndarray) or batch.dtype != np.float32:
            raise ValueError("input must be a float32 NumPy array")
        if batch.ndim != 4 or batch.shape[1:] != (3, *self.variant.resolution):
            raise ValueError("input must have matching NCHW shape")
        if batch.shape[0] <= 0 or not np.isfinite(batch).all():
            raise ValueError("input must be nonempty and finite")
        return torch.from_numpy(np.ascontiguousarray(batch))

    def bind_batch(self, batch) -> None:
        self._require_loaded()
        self._batch = batch

    def warmup(self, n: int) -> None:
        positive(n, "warmup", zero=True)
        if type(n) is not int or self._batch is None:
            raise ValueError("warmup requires an integer count and bound native batch")
        for _ in range(n):
            self.infer(self._batch)

    def infer(self, batch):
        import torch

        self._require_loaded()
        if not isinstance(batch, torch.Tensor) or batch.device.type != "cpu":
            raise ValueError("PyTorch inference requires prepared CPU tensor")
        if (
            batch.dtype != torch.float32
            or batch.ndim != 4
            or batch.shape[1:]
            != (
                3,
                *self.variant.resolution,
            )
            or batch.shape[0] <= 0
        ):
            raise ValueError("invalid native tensor shape or dtype")
        with torch.inference_mode():
            return self.model(batch)

    def synchronize(self) -> None:
        self._require_loaded()

    def memory_report(self) -> MemoryReport:
        self._require_loaded()
        return MemoryReport(
            model_file_bytes=self.artifact.verify(self.workspace).stat().st_size,
            measured=True,
            method="artifact stat; RSS measured by profiler",
        )

    def release(self) -> None:
        self.model = self.variant = self._batch = None
