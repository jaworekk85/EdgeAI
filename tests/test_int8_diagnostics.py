"""Independent numerical checks for the diagnostic controls, not accuracy targets."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def diagnostic():
    pytest.importorskip("numpy")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    path = Path(__file__).resolve().parents[1] / "scripts/diagnose_int8.py"
    spec = importlib.util.spec_from_file_location("diagnose_int8", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_component_controls_isolate_rounding_and_promote_signed_subtraction(diagnostic, tmp_path):
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    # Nonzero INT8 zero point makes subtracting in int8 overflow. The diagnostic
    # must match the runtime DQ result and independent float64 arithmetic.
    weights = np.array([[-25.45, -25.4], [0.07, -0.11]], dtype=np.float32)
    quantized = np.array([[-128, -128], [127, 126]], dtype=np.int8)
    scale = np.array(0.1, dtype=np.float32)
    zero = np.array(127, dtype=np.int8)
    tensors = [
        helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2]),
        helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2]),
    ]
    float_graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"])],
        "float",
        tensors[:1],
        tensors[1:],
        [numpy_helper.from_array(weights, "w")],
    )
    qdq_graph = helper.make_graph(
        [
            helper.make_node("QuantizeLinear", ["x", "xs", "xz"], ["xq"]),
            helper.make_node("DequantizeLinear", ["xq", "xs", "xz"], ["xdq"]),
            helper.make_node("DequantizeLinear", ["w_quantized", "ws", "wz"], ["wdq"]),
            helper.make_node("MatMul", ["xdq", "wdq"], ["y"]),
        ],
        "qdq",
        tensors[:1],
        tensors[1:],
        [
            numpy_helper.from_array(quantized, "w_quantized"),
            numpy_helper.from_array(scale, "ws"),
            numpy_helper.from_array(zero, "wz"),
            numpy_helper.from_array(np.array(0.5, dtype=np.float32), "xs"),
            numpy_helper.from_array(np.array(0, dtype=np.int8), "xz"),
        ],
    )
    paths = [tmp_path / "float.onnx", tmp_path / "qdq.onnx"]
    for graph, path in zip((float_graph, qdq_graph), paths):
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 10
        onnx.save(model, path)
    before = [path.read_bytes() for path in paths]
    controls = diagnostic.component_controls(*paths, tmp_path)
    x = np.array([[0.22, -0.77]], dtype=np.float32)
    restored = (quantized.astype(np.float64) - 127) * float(scale)
    rounding_error = controls["constant_errors"][0]["relative_l2"]
    assert rounding_error == pytest.approx(
        np.linalg.norm(weights - restored) / np.linalg.norm(weights)
    )
    for mode, expected in (
        ("weights_only", x @ restored),
        ("activations_only", (np.rint(x / 0.5) * 0.5) @ weights),
    ):
        runtime = diagnostic.session(controls["controls"][mode]["path"])
        actual = runtime.run(None, {"x": x})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert [path.read_bytes() for path in paths] == before


def test_classifier_rounding_control_preserves_zero_columns_and_sources(diagnostic, tmp_path):
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    weights = np.array([[0, -0.2], [0, 0.33], [0, 1.7]], dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"])],
        "classifier",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
        [numpy_helper.from_array(weights, "w")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    source, destination = tmp_path / "normalized.onnx", tmp_path / "control.onnx"
    onnx.save(model, source)
    before = source.read_bytes()
    settings = diagnostic.weight_only_control(source, destination)
    restored = numpy_helper.to_array(onnx.load(destination).graph.initializer[0])
    assert settings["tensors"][0]["elements"] == weights.size
    assert settings["integer_kernel_claim"] is False
    assert np.isfinite(restored).all() and np.count_nonzero(restored[:, 0]) == 0
    assert np.max(np.abs(restored - weights)) <= 1.7 / 254 + 1e-7
    assert source.read_bytes() == before


def test_diagnostic_refuses_incompatible_or_nonfinite_tensors(diagnostic):
    import numpy as np

    for left, right in (([1, 2], [1]), ([1], [np.nan]), ([np.inf], [0])):
        with pytest.raises(ValueError, match="incompatible.*nonfinite"):
            diagnostic.error(left, right)
