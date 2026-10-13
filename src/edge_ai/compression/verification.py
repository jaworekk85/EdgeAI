"""Exclusive preparation, finite-output checks and observed CPU kernel evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from edge_ai.compression.recipe import KernelEvidence
from edge_ai.devices.profile import capture_device_profile
from edge_ai.utils.artifacts import ArtifactIdentity, sha256_file, workspace_file
from edge_ai.utils.serialization import to_dict


def reserve_directory(destination: Path, workspace: Path) -> Path:
    root = destination.resolve()
    workspace = workspace.resolve(strict=True)
    if not root.is_relative_to(workspace) or root.is_relative_to((workspace / "private").resolve()):
        raise ValueError("public compression destination escapes workspace or enters private/")
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "incomplete.json", {"status": "incomplete"})
    return root


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(to_dict(value), stream, indent=2, allow_nan=False)
        stream.write("\n")


def finish(identity: ArtifactIdentity, root: Path) -> ArtifactIdentity:
    write_json(root / "identity.json", identity)
    write_json(root / "complete.json", {"status": "complete"})
    return identity


def load_inline_onnx(source: ArtifactIdentity, workspace: Path):
    """Read a public inline graph without following external tensor file paths."""
    import onnx

    model = onnx.load(str(source.verify(workspace)), load_external_data=False)

    def check_nodes(nodes):
        for node in nodes:
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.TENSOR:
                    check_tensor(attribute.t)
                elif attribute.type == onnx.AttributeProto.TENSORS:
                    for tensor in attribute.tensors:
                        check_tensor(tensor)
                elif attribute.type == onnx.AttributeProto.GRAPH:
                    check_graph(attribute.g)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for graph in attribute.graphs:
                        check_graph(graph)

    def check_tensor(tensor):
        if tensor.data_location == onnx.TensorProto.EXTERNAL or tensor.external_data:
            raise ValueError(
                "CPU compression requires inline tensors; external data is unsupported"
            )

    def check_graph(graph):
        for tensor in graph.initializer:
            check_tensor(tensor)
        for tensor in graph.sparse_initializer:
            check_tensor(tensor.values)
            check_tensor(tensor.indices)
        check_nodes(graph.node)

    check_graph(model.graph)
    for function in model.functions:
        check_nodes(function.node)
    return model


def probe_onnx(
    path: Path, batches: tuple, workspace: Path, *, quantized: bool
) -> KernelEvidence | None:
    import numpy as np
    import onnx
    import onnxruntime as ort

    if not batches:
        raise ValueError("compression verification requires input batches")
    onnx.checker.check_model(str(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_profiling = quantized
    options.profile_file_prefix = str(path.parent / "kernel_probe")
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    try:
        input_name = session.get_inputs()[0].name
        for batch in batches:
            output = session.run(None, {input_name: batch})[0]
            if (
                output.ndim != 2
                or output.shape[0] != batch.shape[0]
                or not np.isfinite(output).all()
            ):
                raise ValueError("compressed ONNX output is not finite classifier logits")
    finally:
        profile = session.end_profiling() if quantized else None
    if not quantized:
        return None
    counts = observed_quantized_operators(Path(profile).read_bytes())
    if not counts:
        raise ValueError("no quantized CPU kernels observed; refusing a QDQ-only support claim")
    return KernelEvidence(
        sha256_file(path),
        Path(profile).relative_to(workspace).as_posix(),
        sha256_file(Path(profile)),
        capture_device_profile().name,
        ort.__version__,
        counts,
    )


def observed_quantized_operators(profile: bytes) -> dict[str, int]:
    events = json.loads(profile)
    if not isinstance(events, list):
        raise ValueError("kernel profile must contain an event list")
    names = {"QLinearConv", "QGemm", "QLinearMatMul", "MatMulInteger", "ConvInteger", "MatMulNBits"}
    counts = {}
    for event in events:
        arguments = event.get("args", {})
        name = arguments.get("op_name")
        if arguments.get("provider") == "CPUExecutionProvider" and name in names:
            counts[name] = counts.get(name, 0) + 1
    return counts


def verify_float_graphs(source: Path, derived: Path, batches: tuple) -> None:
    """Check lossless preprocessing separately from the subsequent lossy transform."""
    import numpy as np
    import onnxruntime as ort

    if not batches:
        raise ValueError("FP32 graph parity requires verification inputs")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    sessions = [
        ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        for path in (source, derived)
    ]
    for batch in batches:
        outputs = [
            session.run(None, {session.get_inputs()[0].name: batch})[0] for session in sessions
        ]
        np.testing.assert_allclose(outputs[1], outputs[0], rtol=1e-4, atol=5e-5)
        if not np.array_equal(outputs[0].argmax(axis=1), outputs[1].argmax(axis=1)):
            raise ValueError("FP32 preprocessing changed top-1 predictions")


def verify_prepared_identity(identity: ArtifactIdentity, workspace: Path) -> None:
    """Verify derived identity, completion and raw evidence before execution."""
    from edge_ai.compression.recipe import CompressionDetails
    from edge_ai.utils.serialization import from_dict

    path = identity.verify(workspace)
    metadata = identity.export_metadata.get("compression_details")
    if metadata is None:
        raise ValueError("compressed artifact lacks transformation lineage")
    root = path.parent
    if json.loads((root / "complete.json").read_text(encoding="utf-8")) != {"status": "complete"}:
        raise ValueError("compression artifact is not complete")
    if json.loads((root / "identity.json").read_text(encoding="utf-8")) != to_dict(identity):
        raise ValueError("saved artifact identity differs from prepared description")
    details = from_dict(CompressionDetails, metadata)
    details.source_artifact.verify(workspace)
    if details.calibration_manifest is not None:
        details.calibration_manifest.verify(workspace)
    evidence = details.kernel_evidence
    if evidence is not None:
        profile = workspace_file(workspace, evidence.profile_path)
        raw_profile = profile.read_bytes()
        if (
            hashlib.sha256(raw_profile).hexdigest() != evidence.profile_sha256
            or identity.sha256 != evidence.artifact_sha256
        ):
            raise ValueError("kernel profile identity differs from derived artifact evidence")
        if observed_quantized_operators(raw_profile) != evidence.operator_counts:
            raise ValueError("declared kernel operators differ from the raw profile")
