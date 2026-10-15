"""Exclusive ONNX export with numerical verification and recorded settings."""

import json
from pathlib import Path

from edge_ai.devices.profile import package_versions
from edge_ai.utils.artifacts import ArtifactIdentity, sha256_file
from edge_ai.utils.serialization import nonempty, positive, to_dict


def export_onnx(
    model,
    example,
    destination: Path,
    *,
    workspace: Path,
    checkpoint_sha256: str,
    verification_batches: tuple = (),
    atol: float = 5e-5,
    tolerance_reason: str | None = None,
) -> ArtifactIdentity:
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    positive(atol, "export absolute tolerance")
    if atol != 5e-5 and tolerance_reason is None:
        raise ValueError("changed export tolerance requires an explicit numerical reason")
    if tolerance_reason is not None:
        nonempty(tolerance_reason, "export tolerance reason")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "incomplete.json").write_text('{"status":"incomplete"}', encoding="utf-8")
    model = model.cpu().eval()
    torch.set_num_threads(1)
    program = torch.onnx.export(
        model,
        (example,),
        dynamo=True,
        verbose=False,
        opset_version=18,
        external_data=False,
        input_names=["images"],
        output_names=["logits"],
    )
    path = destination / "model.onnx"
    program.save(str(path), external_data=False)
    onnx.checker.check_model(str(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    batches = verification_batches or (example.numpy(),)
    maximum_error = 0.0
    for batch in batches:
        with torch.inference_mode():
            expected = model(torch.from_numpy(batch)).numpy()
        actual = session.run(None, {"images": batch})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=atol)
        if not np.array_equal(actual.argmax(axis=1), expected.argmax(axis=1)):
            raise ValueError("ONNX/PyTorch top-1 predictions differ")
        maximum_error = max(maximum_error, float(np.max(np.abs(actual - expected))))
    metadata = {
        "exporter": "torch.onnx.export",
        "dynamo": True,
        "requested_opset": 18,
        "actual_opsets": {item.domain: item.version for item in onnx.load(str(path)).opset_import},
        "external_data": False,
        "input_name": "images",
        "output_name": "logits",
        "input_shape": list(example.shape),
        "torch": str(torch.__version__),
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "verification_samples": len(batches),
        "rtol": 1e-4,
        "atol": atol,
        "tolerance_reason": tolerance_reason or (
            "CPU FP32 backend rounding; observed near-zero difference 1.78e-5"
        ),
        "export_versions": package_versions(
            ("torch", "torchvision", "onnx", "onnxscript", "onnx_ir")
        ),
        "max_abs_error": maximum_error,
        "top1_equal": True,
    }
    identity = ArtifactIdentity(
        path.relative_to(workspace).as_posix(),
        sha256_file(path),
        "onnx",
        checkpoint_sha256,
        metadata,
    )
    with (destination / "identity.json").open("x", encoding="utf-8") as stream:
        json.dump(to_dict(identity), stream, indent=2)
    with (destination / "complete.json").open("x", encoding="utf-8") as stream:
        json.dump({"status": "complete"}, stream)
    return identity
