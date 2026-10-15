"""Bounded, immutable CPU quantization diagnosis; no deployment/profile claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from edge_ai.compression.reference import load_compression_prepared
from edge_ai.compression.verification import (
    load_inline_onnx,
    reserve_directory,
    verify_float_graphs,
    write_json,
)
from edge_ai.datasets.manifest import DatasetManifest
from edge_ai.devices.profile import capture_device_profile
from edge_ai.records.store import new_run_id
from edge_ai.utils.artifacts import ArtifactIdentity, sha256_file, workspace_file
from edge_ai.utils.provenance import capture_provenance
from edge_ai.utils.serialization import from_dict


def session(path: Path, *, optimize: bool = False):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if optimize
        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def error(left, right) -> dict:
    import numpy as np

    left = np.asarray(left, dtype=np.float64).ravel()
    right = np.asarray(right, dtype=np.float64).ravel()
    if left.shape != right.shape or not (np.isfinite(left).all() and np.isfinite(right).all()):
        raise ValueError("diagnostic tensors have incompatible shapes or nonfinite values")
    difference = left - right
    return {
        "relative_l2": float(np.linalg.norm(difference) / max(np.linalg.norm(left), 1e-12)),
        "max_absolute": float(np.max(np.abs(difference))),
    }


def evaluate(path: Path, manifest: DatasetManifest, workspace: Path) -> dict:
    import numpy as np

    runtime = session(path, optimize=True)
    predictions, labels = [], []
    for index in range(len(manifest.samples)):
        batch, label = manifest.batch(workspace, index)
        logits = runtime.run(None, {runtime.get_inputs()[0].name: batch})[0]
        if not np.isfinite(logits).all():
            raise ValueError("nonfinite diagnostic logits")
        predictions.append(int(np.argmax(logits)))
        labels.append(label)
    correct = sum(left == right for left, right in zip(predictions, labels))
    return {
        "correct": correct,
        "count": len(labels),
        "top1": correct / len(labels),
        "predictions": predictions,
        "labels": labels,
        "artifact_sha256": sha256_file(path),
    }


def weight_only_control(normalized: Path, destination: Path, block_size: int = 32) -> dict:
    """Numerical INT8 RTN control on exactly the INT4 classifier weights.

    Save dequantized FP32 constants, not an INT8 deployment artifact. No integer
    kernel claim: this isolates weight rounding without activation quantization.
    """
    import numpy as np
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(normalized), load_external_data=False)
    names = {node.input[1] for node in model.graph.node if node.op_type == "MatMul"}
    changed = []
    for index, tensor in enumerate(model.graph.initializer):
        if tensor.name not in names:
            continue
        values = numpy_helper.to_array(tensor).copy()
        if values.ndim != 2 or values.dtype != np.float32:
            raise ValueError("control requires constant FP32 MatMul matrices")
        restored = values.copy()
        for start in range(0, values.shape[0], block_size):
            block = values[start : start + block_size]
            scale = np.maximum(np.max(np.abs(block), axis=0) / 127, np.finfo(np.float32).tiny)
            quantized = np.clip(np.rint(block / scale), -127, 127).astype(np.int8)
            restored[start : start + block_size] = quantized.astype(np.float32) * scale
        changed.append(
            {"name": tensor.name, "elements": int(values.size), **error(values, restored)}
        )
        model.graph.initializer[index].CopyFrom(numpy_helper.from_array(restored, tensor.name))
    if not changed or len(changed) != len(names):
        raise ValueError("control did not cover all classifier MatMul weights")
    onnx.save(model, str(destination))
    return {
        "scope": "classifier weight rounding only; FP32 execution and activations",
        "bits": 8,
        "block_size": block_size,
        "axis": 0,
        "rounding": "nearest even; symmetric [-127,127]; absmax/127 per block/column",
        "integer_kernel_claim": False,
        "tensors": changed,
    }


def component_controls(float_path: Path, qdq_path: Path, root: Path) -> dict:
    """Isolate stored weight/bias rounding from activation QDQ, without refitting."""
    import copy

    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    original = onnx.load(str(float_path))
    quantized = onnx.load(str(qdq_path))
    floats = {tensor.name: tensor for tensor in original.graph.initializer}
    constants = {
        tensor.name: numpy_helper.to_array(tensor) for tensor in quantized.graph.initializer
    }
    errors = []
    for node in quantized.graph.node:
        if node.op_type != "DequantizeLinear" or node.input[0] not in constants:
            continue
        source_name = node.input[0].removesuffix("_quantized")
        if source_name not in floats:
            raise ValueError("quantized constant has no original FP32 counterpart")
        values = constants[node.input[0]].astype(np.float64)
        scale, zero = constants[node.input[1]], constants[node.input[2]]
        axis = next(
            (helper.get_attribute_value(attr) for attr in node.attribute if attr.name == "axis"), 1
        )
        if scale.size > 1:
            shape = [1] * values.ndim
            shape[axis] = scale.size
            restored = (values - zero.astype(np.float64).reshape(shape)) * scale.reshape(shape)
        else:
            restored = (values - zero.astype(np.float64)) * scale
        errors.append(
            {
                "tensor": source_name,
                "quantized_dtype": str(constants[node.input[0]].dtype),
                **error(numpy_helper.to_array(floats[source_name]), restored),
            }
        )
    paths = {}
    for mode in ("weights_only", "activations_only"):
        model = copy.deepcopy(quantized)
        replacements, nodes, removed = {}, [], []
        for node in model.graph.node:
            node.input[:] = [replacements.get(name, name) for name in node.input]
            if mode == "weights_only" and node.op_type in ("QuantizeLinear", "DequantizeLinear"):
                if node.input[0] not in constants:
                    replacements[node.output[0]] = node.input[0]
                    removed.append(node.name)
                    continue
            if mode == "activations_only" and node.op_type == "DequantizeLinear":
                if node.input[0] in constants:
                    source_name = node.input[0].removesuffix("_quantized")
                    model.graph.initializer.append(
                        numpy_helper.from_array(
                            numpy_helper.to_array(floats[source_name]).copy(), node.output[0]
                        )
                    )
                    removed.append(node.name)
                    continue
            nodes.append(node)
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        for output in model.graph.output:
            output.name = replacements.get(output.name, output.name)
        used = {name for node in nodes for name in node.input}
        retained = [tensor for tensor in model.graph.initializer if tensor.name in used]
        del model.graph.initializer[:]
        model.graph.initializer.extend(retained)
        path = root / (mode + ".onnx")
        onnx.checker.check_model(model)
        onnx.save(model, str(path))
        paths[mode] = {"path": path, "removed_nodes": removed, "integer_kernel_claim": False}
    return {"controls": paths, "constant_errors": errors}


def depthwise_input_control(
    float_path: Path,
    qdq_path: Path,
    root: Path,
    manifests: dict[str, DatasetManifest],
    workspace: Path,
) -> dict:
    """Isolate rounding of the first depthwise input using original FP32 values."""
    import copy

    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(str(float_path))
    qdq = onnx.load(str(qdq_path))
    target = next(
        node
        for node in model.graph.node
        if node.op_type == "Conv"
        and any(attr.name == "group" and attr.i > 1 for attr in node.attribute)
    )
    activation, output = target.input[0], target.output[0]
    quantizer = next(
        node
        for node in qdq.graph.node
        if node.op_type == "QuantizeLinear" and node.input[0] == activation
    )
    constants = {tensor.name: tensor for tensor in qdq.graph.initializer}
    scale = numpy_helper.to_array(constants[quantizer.input[1]])
    zero = numpy_helper.to_array(constants[quantizer.input[2]])
    if scale.size != 1 or zero.size != 1:
        raise ValueError("depthwise diagnostic expects per-tensor activation quantization")
    shape_info = {item.name: item for item in (*model.graph.input, *model.graph.value_info)}
    shape = tuple(dim.dim_value for dim in shape_info[activation].type.tensor_type.shape.dim)
    if not shape or not all(shape):
        raise ValueError("depthwise activation shape is not concrete")
    isolated_path = root / "depthwise_float.onnx"
    onnx.utils.extract_model(str(float_path), str(isolated_path), [activation], [output])
    isolated = onnx.load(str(isolated_path))
    # Preserve original weights/bias; only QDQ the previously FP32 input.
    edited = copy.deepcopy(isolated)
    edited.graph.initializer.extend([constants[name] for name in quantizer.input[1:]])
    dq_name = "diagnostic_depthwise_input_dequantized"
    nodes = [
        copy.deepcopy(quantizer),
        helper.make_node(
            "DequantizeLinear", [quantizer.output[0], *quantizer.input[1:]], [dq_name]
        ),
    ]
    for node in edited.graph.node:
        node.input[:] = [dq_name if name == activation else name for name in node.input]
        nodes.append(node)
    del edited.graph.node[:]
    edited.graph.node.extend(nodes)
    rounded_path = root / "depthwise_input_qdq.onnx"
    onnx.checker.check_model(edited)
    onnx.save(edited, str(rounded_path))
    runtimes = [session(isolated_path), session(rounded_path)]
    float_augmented = session(root / "float_aug.onnx")
    channels = shape[1]
    result = {}
    for split, manifest in manifests.items():
        observations = [[] for _ in range(channels)]
        global_error = 0.0
        for index in range(len(manifest.samples)):
            batch, _ = manifest.batch(workspace, index)
            values = float_augmented.run(
                [activation + "_ReshapedSavedOutput"], {float_augmented.get_inputs()[0].name: batch}
            )[0].reshape(shape)
            limits = np.iinfo(zero.dtype)
            quantized = np.clip(np.rint(values / scale) + zero, limits.min, limits.max)
            restored = (quantized - zero) * scale
            outputs = [runtime.run(None, {activation: values})[0] for runtime in runtimes]
            global_error += error(*outputs)["relative_l2"]
            for channel, rows in enumerate(observations):
                left, right = values[:, channel], restored[:, channel]
                rows.append({
                    "input_rms_mean": float(np.sqrt(np.mean(left.astype(np.float64) ** 2))),
                    "input_relative_l2_mean": error(left, right)["relative_l2"],
                    "output_relative_l2_mean": error(
                        outputs[0][:, channel], outputs[1][:, channel])["relative_l2"],
                    "rounded_to_zero_fraction_mean": float(np.mean(right == 0)),
                    "original_zero_fraction_mean": float(np.mean(left == 0)),
                    "newly_zero_fraction_mean": float(np.mean((right == 0) & (left != 0))),
                })
        count = len(manifest.samples)
        result[split] = {
            "samples": count,
            "isolated_output_relative_l2_mean": global_error / count,
            "channels": [
                {"channel": channel, **{key: mean(row[key] for row in rows) for key in rows[0]}}
                for channel, rows in enumerate(observations)
            ],
        }
    return {
        "node": target.name,
        "input": activation,
        "output": output,
        "activation_scale": float(scale),
        "activation_zero_point": int(zero),
        "weights_bias": "original FP32; identical in both isolated graphs",
        "splits": result,
    }


def activation_errors(
    float_path: Path,
    qdq_path: Path,
    destination: Path,
    manifests: dict[str, DatasetManifest],
    workspace: Path,
) -> dict:
    import numpy as np
    import onnx
    from onnx import numpy_helper
    from onnxruntime.quantization.qdq_loss_debug import (
        create_activation_matching,
        modify_model_output_intermediate_tensors,
    )

    # The debug API appends reshaped outputs. Execute one image at a time and
    # immediately reduce to statistics instead of collecting every activation.
    suffix = "_ReshapedSavedOutput"
    float_aug, qdq_aug = destination / "float_aug.onnx", destination / "qdq_aug.onnx"
    modify_model_output_intermediate_tensors(float_path, float_aug)
    modify_model_output_intermediate_tensors(qdq_path, qdq_aug)
    sessions = [session(float_aug), session(qdq_aug)]
    models = [onnx.load(str(float_path)), onnx.load(str(qdq_path))]
    order = {name: index for index, node in enumerate(models[0].graph.node) for name in node.output}
    producers = {name: node.op_type for node in models[0].graph.node for name in node.output}
    initializers = {item.name: numpy_helper.to_array(item) for item in models[1].graph.initializer}
    ranges = {}
    for node in models[1].graph.node:
        if node.op_type != "QuantizeLinear":
            continue
        scale = initializers[node.input[1]]
        zero = initializers[node.input[2]]
        limits = np.iinfo(zero.dtype)
        ranges[node.input[0]] = (
            float((limits.min - int(zero)) * scale),
            float((limits.max - int(zero)) * scale),
            float(scale),
        )

    result = {}
    for split, manifest in manifests.items():
        aggregates = {}
        for index in range(len(manifest.samples)):
            batch, _ = manifest.batch(workspace, index)
            activations = []
            for runtime in sessions:
                outputs = runtime.run(None, {runtime.get_inputs()[0].name: batch})
                activations.append(
                    {
                        item.name[: -len(suffix)]: [value]
                        for item, value in zip(runtime.get_outputs(), outputs)
                        if item.name.endswith(suffix)
                    }
                )
            matches = create_activation_matching(activations[1], activations[0])
            for name, match in matches.items():
                if "float" not in match:
                    continue
                before, after, original = (
                    match["pre_qdq"][0],
                    match["post_qdq"][0],
                    match["float"][0],
                )
                local, accumulated = error(before, after), error(original, after)
                low, high, _ = ranges.get(name, (-float("inf"), float("inf"), None))
                aggregates.setdefault(name, []).append({
                    "local_relative_l2": local["relative_l2"],
                    "accumulated_relative_l2": accumulated["relative_l2"],
                    "max_absolute": accumulated["max_absolute"],
                    "clipped_elements": int(np.count_nonzero((before < low) | (before > high))),
                    "elements": int(before.size) if name in ranges else 0,
                })
            del outputs, activations, matches
        summaries = []
        for name, rows in aggregates.items():
            stats = {"tensor": name, "order": order.get(name, -1),
                     "producer": producers.get(name, "input"), "samples": len(rows)}
            for prefix in ("local", "accumulated"):
                key = prefix + "_relative_l2"
                stats[key + "_mean"] = mean(row[key] for row in rows)
                stats[key + "_max"] = max(row[key] for row in rows)
            stats["max_absolute"] = max(row["max_absolute"] for row in rows)
            for key in ("clipped_elements", "elements"):
                stats[key] = sum(row[key] for row in rows)
            stats["clipped_fraction"] = stats["clipped_elements"] / max(stats["elements"], 1)
            if name in ranges:
                low, high, scale = ranges[name]
                stats.update({"range_min": low, "range_max": high, "scale": scale})
            summaries.append(stats)
        result[split] = sorted(summaries, key=lambda item: (item["order"], item["tensor"]))
        print(f"Matched {len(aggregates)} activation tensors on {split}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    args = parser.parse_args()
    workspace = args.workspace.resolve(strict=True)
    prepared = workspace_file(workspace, args.prepared.as_posix())
    loaded = load_compression_prepared(prepared, workspace)
    assets, matrix, evaluation = loaded.assets, loaded.matrix, loaded.manifest
    variants = {
        item["model"]["compression"]["precision"]: item
        for item in assets["variants"]
        if item["model"]["runtime"] == "onnxruntime"
        and item["model"]["compression"]["method"] in ("none", "quantization")
    }
    identities = {
        key: from_dict(ArtifactIdentity, item["artifact"]) for key, item in variants.items()
    }
    int8, int4 = identities["int8"], identities["int4"]
    calibration = from_dict(
        DatasetManifest, int8.export_metadata["compression_details"]["calibration_manifest"]
    )
    if len(evaluation.samples) > 64 or len(calibration.samples) > 64:
        raise ValueError("bounded diagnosis supports at most 64 images per split")
    root = reserve_directory(
        workspace / "artifacts/compression_diagnostics" / new_run_id(matrix), workspace
    )
    write_json(
        root / "inputs.json",
        {
            "prepared": prepared.relative_to(workspace).as_posix(),
            "prepared_sha256": sha256_file(prepared),
            "identities": identities,
            "evaluation": evaluation,
            "calibration": calibration,
            "provenance": capture_provenance(
                workspace, runtime_packages=("onnx", "onnxruntime", "numpy")
            ),
            "device": capture_device_profile(),
            "purpose": "bounded quality diagnosis; no profiling/energy claims",
            "diagnostic_script_sha256": sha256_file(Path(__file__)),
        },
    )
    float_path = int8.verify(workspace).parent / "preprocessed.onnx"
    normalized = int4.verify(workspace).parent / "normalized.onnx"
    # Validate auxiliary graphs as public inline files before runtime loading.
    for path in (float_path, normalized):
        identity = ArtifactIdentity(
            path.relative_to(workspace).as_posix(),
            sha256_file(path),
            "onnx",
            int8.checkpoint_sha256,
            {"diagnostic_auxiliary": True},
        )
        load_inline_onnx(identity, workspace)
    batches = tuple(evaluation.batch(workspace, i)[0] for i in range(len(evaluation.samples)))
    verify_float_graphs(identities["fp32"].verify(workspace), float_path, batches)
    verify_float_graphs(identities["fp32"].verify(workspace), normalized, batches)
    source_graphs = {"float": float_path, "normalized": normalized, "qdq": int8.verify(workspace)}
    write_json(
        root / "graph_hashes.json",
        {
            path.relative_to(workspace).as_posix(): sha256_file(path)
            for path in source_graphs.values()
        },
    )
    copies = {}
    for name, path in source_graphs.items():
        target = root / (name + "_source.onnx")
        # ORT debug shape inference writes temporary siblings. Run it exclusively
        # on these new copies, so it cannot touch old preparation directories.
        with target.open("xb") as stream:
            stream.write(path.read_bytes())
        if sha256_file(target) != sha256_file(path):
            raise ValueError("diagnostic source copy does not match original")
        copies[name] = target
    float_path, normalized, qdq_path = copies["float"], copies["normalized"], copies["qdq"]
    control_path = root / "classifier_int8_rounding_fp32.onnx"
    control = weight_only_control(normalized, control_path)
    summaries = {
        key: evaluate(identity.verify(workspace), evaluation, workspace)
        for key, identity in identities.items()
    }
    summaries["int8_classifier_weight_rounding_control"] = {
        **evaluate(control_path, evaluation, workspace),
        "settings": control,
    }
    components = component_controls(float_path, qdq_path, root)
    for key, settings in components["controls"].items():
        path = settings.pop("path")
        summaries[key] = {**evaluate(path, evaluation, workspace), "settings": settings}
    write_json(root / "constant_errors.json", components["constant_errors"])
    write_json(root / "quality.json", summaries)
    print(json.dumps({key: item["top1"] for key, item in summaries.items()}), flush=True)
    errors = activation_errors(
        float_path,
        qdq_path,
        root,
        {"calibration": calibration, "evaluation": evaluation},
        workspace,
    )
    write_json(root / "activations.json", errors)
    write_json(
        root / "depthwise_input_control.json",
        depthwise_input_control(
            float_path,
            qdq_path,
            root,
            {"calibration": calibration, "evaluation": evaluation},
            workspace,
        ),
    )
    write_json(
        root / "output_hashes.json",
        {path.name: sha256_file(path) for path in sorted(root.iterdir()) if path.is_file()},
    )
    write_json(root / "complete.json", {"status": "complete"})
    print(root.relative_to(workspace).as_posix(), flush=True)


if __name__ == "__main__":
    main()
