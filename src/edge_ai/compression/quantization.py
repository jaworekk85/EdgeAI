"""CPU FP16 checkpoint conversion and calibrated/weight-only ONNX quantization."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from edge_ai.compression.pruning import weight_targets
from edge_ai.compression.recipe import CompressionDetails, CompressionRecipe
from edge_ai.compression.spec import CompressionSpec
from edge_ai.compression.verification import (
    finish,
    load_inline_onnx,
    probe_onnx,
    reserve_directory,
    verify_float_graphs,
)
from edge_ai.datasets.manifest import DatasetManifest
from edge_ai.devices.profile import package_versions
from edge_ai.utils.artifacts import ArtifactIdentity, sha256_file
from edge_ai.utils.serialization import to_dict


def checkpoint_fp16(
    source: ArtifactIdentity,
    nominal: CompressionSpec,
    recipe: CompressionRecipe,
    destination: Path,
    *,
    workspace: Path,
) -> tuple[ArtifactIdentity, CompressionSpec]:
    import torch

    if recipe.method != "fp16" or source.role != "checkpoint":
        raise ValueError("FP16 conversion requires an original checkpoint")
    state = torch.load(source.verify(workspace), map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError("FP16 conversion requires a nonempty tensor state dict")
    names, total = [], 0
    converted = {}
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError("checkpoint contains a non-tensor entry")
        if value.is_floating_point():
            if value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError("FP16 source must contain finite FP32 tensors")
            result = value.half()
            if not torch.isfinite(result).all():
                raise ValueError("FP16 conversion overflowed")
            converted[name] = result
            names.append(name)
            total += value.numel()
        else:
            converted[name] = value.clone()
    details = CompressionDetails(
        recipe,
        source,
        tuple(names),
        total,
        total,
        "FP16 floating state tensors and inputs; CPU kernel internal arithmetic unverified",
        package_versions(("torch",)),
    )
    root = reserve_directory(destination, workspace)
    path = root / "weights.pt"
    with path.open("xb") as stream:
        torch.save(converted, stream)
    identity = ArtifactIdentity(
        path.relative_to(workspace).as_posix(),
        sha256_file(path),
        "compressed_checkpoint",
        source.checkpoint_sha256,
        {"compression_details": to_dict(details)},
    )
    spec = replace(
        nominal,
        method="quantization",
        precision="fp16",
        kernel_support=None,
        kernel_support_method="FP16 storage/tensor dtype verified; CPU arithmetic unknown",
        details=details,
    )
    return finish(identity, root), spec


def normalize_gemm(model) -> None:
    """Lossless supported Gemm -> MatMul/Add, to expose classifier weight-only ops."""
    import numpy as np
    from onnx import helper, numpy_helper

    initializers = {item.name: item for item in model.graph.initializer}
    existing = set(initializers) | {value for node in model.graph.node for value in node.output}
    nodes = []
    for index, node in enumerate(model.graph.node):
        if node.op_type != "Gemm":
            nodes.append(node)
            continue
        attributes = {
            attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute
        }
        if (
            attributes.get("alpha", 1) != 1
            or attributes.get("beta", 1) != 1
            or attributes.get("transA", 0) != 0
            or node.input[1] not in initializers
        ):
            raise ValueError(
                "INT4 normalization supports constant Gemm with alpha=beta=1, transA=0"
            )
        original = numpy_helper.to_array(initializers[node.input[1]])
        if original.ndim != 2:
            raise ValueError("Gemm constant weights must be a matrix")
        values = original.T.copy() if attributes.get("transB", 0) else original.copy()
        name = f"compression_gemm_{index}_weight"
        intermediate = f"compression_gemm_{index}_output"
        if name in existing or intermediate in existing:
            raise ValueError("INT4 normalization tensor name collision")
        existing.update((name, intermediate))
        model.graph.initializer.append(
            numpy_helper.from_array(np.ascontiguousarray(values), name=name)
        )
        has_bias = len(node.input) == 3 and bool(node.input[2])
        output = intermediate if has_bias else node.output[0]
        nodes.append(
            helper.make_node(
                "MatMul",
                [node.input[0], name],
                [output],
                name=f"compression_gemm_{index}_matmul",
            )
        )
        if has_bias:
            nodes.append(
                helper.make_node(
                    "Add",
                    [output, node.input[2]],
                    list(node.output),
                    name=f"compression_gemm_{index}_bias",
                )
            )
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    used = {value for node in model.graph.node for value in node.input}
    # No subgraphs in this initial supported path; don't delete nested references.
    if any(
        attribute.type == 5 or attribute.type == 10
        for node in model.graph.node
        for attribute in node.attribute
    ):
        raise ValueError("INT4 normalization does not support control-flow subgraphs")
    tensors = [tensor for tensor in model.graph.initializer if tensor.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(tensors)


def quantize_onnx(
    source: ArtifactIdentity,
    nominal: CompressionSpec,
    recipe: CompressionRecipe,
    destination: Path,
    *,
    workspace: Path,
    verification_batches: tuple,
    calibration: DatasetManifest | None = None,
    evaluation: DatasetManifest | None = None,
) -> tuple[ArtifactIdentity, CompressionSpec]:
    import numpy as np
    import onnx
    from onnx import numpy_helper
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quant_pre_process,
        quantize_static,
    )

    if source.role != "onnx" or recipe.method not in ("int8", "int4"):
        raise ValueError("ONNX quantization requires original ONNX and INT8/INT4 recipe")
    model = load_inline_onnx(source, workspace)
    targets = weight_targets(model)
    initializers = {item.name: item for item in model.graph.initializer}
    total = sum(int(np.prod(initializers[name].dims)) for name, *_ in targets)
    if total == 0:
        raise ValueError("no eligible quantization weights")
    if recipe.method == "int8":
        if calibration is None or evaluation is None or calibration.split != "train":
            raise ValueError("INT8 requires separate train calibration and evaluation manifests")
        # Check overlap before reserving a directory or fitting scales.
        if (
            calibration.preprocessing != evaluation.preprocessing
            or (
                {sample.path for sample in calibration.samples}
                & {sample.path for sample in evaluation.samples}
            )
            or (
                {sample.sha256 for sample in calibration.samples}
                & {sample.sha256 for sample in evaluation.samples}
            )
        ):
            raise ValueError("calibration overlaps evaluation or has different preprocessing")
        calibration.verify(workspace)
        evaluation.verify(workspace)
    elif calibration is not None:
        raise ValueError("INT4 weight-only quantization does not use calibration")
    root = reserve_directory(destination, workspace)
    path = root / "model.onnx"
    if recipe.method == "int8":
        preprocessed = root / "preprocessed.onnx"
        quant_pre_process(
            source.verify(workspace),
            preprocessed,
            skip_optimization=False,
            skip_symbolic_shape=True,
            skip_onnx_shape=False,
            save_as_external_data=False,
        )
        verify_float_graphs(source.verify(workspace), preprocessed, verification_batches)
        input_name = model.graph.input[0].name

        class Reader(CalibrationDataReader):
            def __init__(self):
                self.index = 0

            def get_next(self):
                if self.index >= len(calibration.samples):
                    return None
                batch, _ = calibration.batch(workspace, self.index)
                self.index += 1
                return {input_name: batch}

        quantize_static(
            preprocessed,
            path,
            Reader(),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            op_types_to_quantize=["Conv", "Gemm", "MatMul"],
            calibrate_method=CalibrationMethod.MinMax,
            calibration_providers=["CPUExecutionProvider"],
            use_external_data_format=False,
        )
        derived = onnx.load(str(path))
        quantized_weights = [
            tensor
            for tensor in derived.graph.initializer
            if tensor.data_type == 3 and len(tensor.dims) >= 2
        ]
        changed = tuple(tensor.name for tensor in quantized_weights)
        target = sum(int(np.prod(tensor.dims)) for tensor in quantized_weights)
        scope = (
            "partial Conv/Gemm/MatMul QDQ INT8; signed per-channel weights; MinMax calibration"
        )
        settings = {"format": "QDQ", "per_channel": True, "calibration": "MinMax", "signed": True}
    else:
        from onnxruntime.quantization.matmul_nbits_quantizer import (
            DefaultWeightOnlyQuantConfig,
            MatMulNBitsQuantizer,
        )

        normalize_gemm(model)
        normalized = root / "normalized.onnx"
        onnx.save_model(model, str(normalized))
        # Independently check normalization matches the source, before lossy quantization.
        verify_float_graphs(source.verify(workspace), normalized, verification_batches)
        source_weights = {tensor.name: tensor for tensor in model.graph.initializer}
        names = {
            node.input[1]
            for node in model.graph.node
            if node.op_type == "MatMul" and node.input[1] in source_weights
        }
        target = sum(numpy_helper.to_array(source_weights[name]).size for name in names)
        if not target:
            raise ValueError("INT4 has no eligible constant MatMul weights")
        configuration = DefaultWeightOnlyQuantConfig(
            block_size=recipe.block_size,
            is_symmetric=True,
            accuracy_level=4,
            quant_format=QuantFormat.QOperator,
            op_types_to_quantize=("MatMul",),
            bits=4,
        )
        quantizer = MatMulNBitsQuantizer(model, algo_config=configuration)
        quantizer.process()
        quantizer.model.save_model_to_file(str(path), use_external_data_format=False)
        changed = tuple(sorted(names))
        scope = (
            "INT4 blockwise MatMul weights only; FP32 activations/Conv; compatible Gemm normalized"
        )
        settings = {
            "format": "MatMulNBits",
            "block_size": recipe.block_size,
            "symmetric": True,
            "accuracy_level": 4,
            "algorithm": "RTN",
        }
    if not changed:
        raise ValueError("quantization changed no eligible tensors")
    evidence = probe_onnx(path, verification_batches, workspace, quantized=True)
    if recipe.method == "int4" and "MatMulNBits" not in evidence.operator_counts:
        raise ValueError("INT4 weight-only kernels were not observed")
    if recipe.method == "int8" and not set(evidence.operator_counts) - {"MatMulNBits"}:
        raise ValueError("INT8 integer kernels were not observed")
    details = CompressionDetails(
        recipe,
        source,
        changed,
        target,
        total,
        scope,
        package_versions(("onnx", "onnxruntime", "numpy")),
        calibration_manifest=calibration,
        kernel_evidence=evidence,
    )
    identity = ArtifactIdentity(
        path.relative_to(workspace).as_posix(),
        sha256_file(path),
        "onnx",
        source.checkpoint_sha256,
        {
            "compression_details": to_dict(details),
            "settings": settings,
            "fp32_preprocessing_parity": {
                "samples": len(verification_batches), "rtol": 1e-4, "atol": 5e-5,
                "top1_equal": True,
            },
            "actual_opsets": {
                item.domain: item.version for item in onnx.load(str(path)).opset_import
            },
        },
    )
    spec = replace(
        nominal,
        method="quantization",
        precision=recipe.precision,
        calibration=calibration.digest() if calibration else None,
        kernel_support=True,
        kernel_support_method="observed quantized CPU operators; hashed preparation profile",
        details=details,
    )
    return finish(identity, root), spec
