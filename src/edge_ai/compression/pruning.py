"""Magnitude masks for dense ONNX weights, with explicit eligibility and axes."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from edge_ai.compression.recipe import CompressionDetails, CompressionRecipe
from edge_ai.compression.spec import CompressionSpec
from edge_ai.compression.verification import finish, load_inline_onnx, probe_onnx, reserve_directory
from edge_ai.devices.profile import package_versions
from edge_ai.utils.artifacts import ArtifactIdentity, sha256_file
from edge_ai.utils.serialization import to_dict


def magnitude_mask(weights, recipe: CompressionRecipe, *, output_axis: int, reduction_axis: int):
    """Return a new bool mask; never mutate weights. Ties use stable index order."""
    import numpy as np

    weights = np.asarray(weights)
    if not np.issubdtype(weights.dtype, np.floating) or not np.isfinite(weights).all():
        raise ValueError("pruning requires finite floating-point weights")
    if (
        not weights.size
        or output_axis == reduction_axis
        or (not 0 <= output_axis < weights.ndim or not 0 <= reduction_axis < weights.ndim)
    ):
        raise ValueError("invalid pruning tensor/axes")
    if recipe.method == "unstructured":
        count = int(weights.size * recipe.amount)
        if count == 0:
            raise ValueError("pruning amount masks no weight entries")
        mask = np.ones(weights.size, dtype=bool)
        mask[np.argsort(np.abs(weights).reshape(-1), kind="stable")[:count]] = False
        return mask.reshape(weights.shape)
    if recipe.method == "channel":
        values = np.moveaxis(weights, output_axis, 0)
        count = int(values.shape[0] * recipe.amount)
        if count == 0:
            raise ValueError("pruning amount masks no channels")
        norms = np.sum(values.astype(np.float64) ** 2, axis=tuple(range(1, values.ndim)))
        mask = np.ones(values.shape, dtype=bool)
        mask[np.argsort(norms, kind="stable")[:count]] = False
        return np.moveaxis(mask, 0, output_axis)
    if recipe.method != "2:4":
        raise ValueError("recipe is not a pruning method")
    if weights.shape[reduction_axis] % 4:
        raise ValueError("2:4 requires a reduction axis divisible by four")
    values = np.moveaxis(weights, reduction_axis, -1)
    groups = values.reshape(*values.shape[:-1], -1, 4)
    indices = np.argsort(np.abs(groups), axis=-1, kind="stable")[..., :2]
    mask = np.ones(groups.shape, dtype=bool)
    np.put_along_axis(mask, indices, False, axis=-1)
    return np.moveaxis(mask.reshape(values.shape), -1, reduction_axis)


def weight_targets(model) -> list[tuple[str, int, int]]:
    """Constant Conv/Gemm/MatMul weights, with output/reduction axes."""
    initializers = {item.name: item for item in model.graph.initializer}
    targets = {}
    for node in model.graph.node:
        if node.op_type not in ("Conv", "Gemm", "MatMul") or len(node.input) < 2:
            continue
        name = node.input[1]
        if name not in initializers:
            continue
        tensor = initializers[name]
        if tensor.data_type != 1:
            raise ValueError("compression expects FP32 source weights")
        if node.op_type == "Conv":
            if len(tensor.dims) != 4:
                raise ValueError("pruning supports Conv2d weights only")
            axes = (0, 1)
        else:
            if len(tensor.dims) != 2:
                raise ValueError("Gemm weights must be matrices")
            attributes = {item.name: item.i for item in node.attribute}
            axes = (0, 1) if node.op_type == "Gemm" and attributes.get("transB", 0) else (1, 0)
        if name in targets and targets[name] != axes:
            raise ValueError("shared weights have inconsistent reduction axes")
        targets[name] = axes
    return [(name, *axes) for name, axes in targets.items()]


def prune_onnx(
    source: ArtifactIdentity,
    nominal: CompressionSpec,
    recipe: CompressionRecipe,
    destination: Path,
    *,
    workspace: Path,
    verification_batches: tuple,
) -> tuple[ArtifactIdentity, CompressionSpec]:
    import numpy as np
    import onnx
    from onnx import numpy_helper

    if source.role != "onnx" or recipe.method not in ("unstructured", "channel", "2:4"):
        raise ValueError("ONNX pruning requires a floating-point ONNX source and pruning recipe")
    model = load_inline_onnx(source, workspace)
    targets = weight_targets(model)
    if not targets:
        raise ValueError("no eligible Conv/Gemm/MatMul weight tensors")
    initializers = {item.name: item for item in model.graph.initializer}
    changed, skipped = [], []
    target_elements = total_elements = zeros_before = zeros_after = masked = 0
    for name, output_axis, reduction_axis in targets:
        tensor = initializers[name]
        values = numpy_helper.to_array(tensor).copy()
        total_elements += values.size
        if recipe.method == "2:4" and values.shape[reduction_axis] % 4:
            skipped.append(name)
            continue
        if recipe.method == "channel" and int(values.shape[output_axis] * recipe.amount) == 0:
            skipped.append(name)
            continue
        if recipe.method == "unstructured" and int(values.size * recipe.amount) == 0:
            skipped.append(name)
            continue
        mask = magnitude_mask(
            values, recipe, output_axis=output_axis, reduction_axis=reduction_axis
        )
        result = values * mask
        target_elements += values.size
        zeros_before += int(np.count_nonzero(values == 0))
        zeros_after += int(np.count_nonzero(result == 0))
        masked += int(np.count_nonzero(~mask))
        changed.append(name)
        tensor.CopyFrom(numpy_helper.from_array(result, name=name))
    details = CompressionDetails(
        recipe,
        source,
        tuple(changed),
        target_elements,
        total_elements,
        "FP32 dense masked Conv/Gemm/MatMul weights; bias retained; no graph compaction",
        package_versions(("onnx", "numpy")),
        zeros_before,
        zeros_after,
        masked,
        tuple(skipped),
    )
    root = reserve_directory(destination, workspace)
    path = root / "model.onnx"
    onnx.save_model(model, str(path))
    probe_onnx(path, verification_batches, workspace, quantized=False)
    identity = ArtifactIdentity(
        path.relative_to(workspace).as_posix(),
        sha256_file(path),
        "onnx",
        source.checkpoint_sha256,
        {"compression_details": to_dict(details), "parent_export": source.export_metadata},
    )
    spec = replace(
        nominal,
        method=recipe.compression_method,
        sparsity_level=zeros_after / target_elements,
        sparsity_pattern={"unstructured": "unstructured", "channel": "channel", "2:4": "2:4"}[
            recipe.method
        ],
        kernel_support=False,
        kernel_support_method="dense CPU graph; no sparse kernels enabled",
        details=details,
    )
    return finish(identity, root), spec
