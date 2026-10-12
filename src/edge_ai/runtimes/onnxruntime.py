"""ONNX Runtime CPU backend; never imports PyTorch."""

from pathlib import Path

from edge_ai.config import ExecutionSettings
from edge_ai.models.variant import ModelVariant
from edge_ai.runtimes.base import MemoryReport
from edge_ai.utils.artifacts import ArtifactIdentity
from edge_ai.utils.serialization import positive


class ONNXRuntime:
    def __init__(self, workspace: Path, artifact: ArtifactIdentity, settings: ExecutionSettings):
        self.workspace, self.artifact, self.settings = workspace, artifact, settings
        self.session = self.variant = self._batch = None

    def load(self, variant: ModelVariant) -> None:
        import onnxruntime as ort

        if self.session is not None:
            raise RuntimeError("release the loaded model first")
        if variant.runtime != "onnxruntime" or self.artifact.role != "onnx":
            raise ValueError("ONNX Runtime requires ONNX artifact")
        if variant.weights_digest != self.artifact.checkpoint_sha256:
            raise ValueError("canonical checkpoint identity differs from variant")
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.settings.intra_op_threads
        options.inter_op_num_threads = self.settings.inter_op_threads
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        path = self.artifact.verify(self.workspace)
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=[self.settings.provider]
        )
        self.variant = variant
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or len(self.session.get_outputs()) != 1:
            self.release()
            raise ValueError("reference ONNX requires one input and output")
        self.input_name = inputs[0].name
        self.input_shape = inputs[0].shape

    def _require_loaded(self):
        if self.session is None:
            raise RuntimeError("no model loaded")

    def prepare_batch(self, batch):
        import numpy as np

        self._require_loaded()
        if not isinstance(batch, np.ndarray) or batch.dtype != np.float32:
            raise ValueError("input must be a float32 NumPy array")
        if batch.ndim != 4 or batch.shape[1:] != (3, *self.variant.resolution):
            raise ValueError("input must have matching NCHW shape")
        if batch.shape[0] <= 0 or not np.isfinite(batch).all():
            raise ValueError("input must be nonempty and finite")
        if any(
            isinstance(expected, int) and actual != expected
            for actual, expected in zip(batch.shape, self.input_shape)
        ):
            raise ValueError("input shape differs from exported static shape")
        return np.ascontiguousarray(batch)

    def bind_batch(self, batch) -> None:
        self._batch = self.prepare_batch(batch)

    def warmup(self, n: int) -> None:
        positive(n, "warmup", zero=True)
        if type(n) is not int or self._batch is None:
            raise ValueError("warmup requires an integer count and bound native batch")
        for _ in range(n):
            self.infer(self._batch)

    def infer(self, batch):
        self._require_loaded()
        return self.session.run(None, {self.input_name: batch})[0]

    def synchronize(self) -> None:
        self._require_loaded()

    def memory_report(self):
        self._require_loaded()
        return MemoryReport(
            model_file_bytes=self.artifact.verify(self.workspace).stat().st_size,
            measured=True,
            method="artifact stat; RSS measured by profiler",
        )

    def release(self) -> None:
        self.session = self.variant = self._batch = None
