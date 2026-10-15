from dataclasses import replace

import pytest

from edge_ai.compression.recipe import CompressionRecipe
from edge_ai.utils.serialization import from_dict, to_dict


@pytest.mark.parametrize(
    "arguments",
    [
        {"id": "bad/id", "method": "int8"},
        {"id": "bad", "method": "unstructured"},
        {"id": "bad", "method": "channel", "amount": 1},
        {"id": "bad", "method": "channel", "amount": 0},
        {"id": "bad", "method": "int8", "amount": 0.5},
        {"id": "bad", "method": "2:4", "amount": 0.5},
        {"id": "bad", "method": "int4", "block_size": 15},
        {"id": "bad", "method": "int4", "block_size": 24},
        {"id": "bad", "method": "fp16", "block_size": 64},
        {"id": "bad", "method": "fp16", "int8_scope": "linear"},
        {"id": "bad", "method": "int8", "int8_scope": "unknown"},
    ],
)
def test_invalid_compression_recipes(arguments):
    with pytest.raises(ValueError):
        CompressionRecipe(**arguments)


def test_array_masks_exact_axes_ties_and_no_input_mutation():
    np = pytest.importorskip("numpy")
    from edge_ai.compression.pruning import magnitude_mask

    values = np.array([[1.0, -1.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
    original = values.copy()
    mask = magnitude_mask(
        values, CompressionRecipe("u", "unstructured", 0.25), output_axis=0, reduction_axis=1
    )
    np.testing.assert_array_equal(mask, [[False, False, True, True], [True] * 4])
    channel = magnitude_mask(
        values, CompressionRecipe("c", "channel", 0.5), output_axis=0, reduction_axis=1
    )
    np.testing.assert_array_equal(channel, [[False] * 4, [True] * 4])
    pattern = magnitude_mask(values, CompressionRecipe("s", "2:4"), output_axis=0, reduction_axis=1)
    np.testing.assert_array_equal(pattern, [[False, False, True, True]] * 2)
    np.testing.assert_array_equal(values, original)
    transposed = magnitude_mask(
        values.T, CompressionRecipe("t", "2:4"), output_axis=1, reduction_axis=0
    )
    np.testing.assert_array_equal(transposed, pattern.T)


def test_conv_2of4_groups_input_channels_separately_at_each_spatial_position():
    np = pytest.importorskip("numpy")
    from edge_ai.compression.pruning import magnitude_mask

    values = np.arange(1, 33, dtype=np.float32).reshape(2, 4, 2, 2)
    mask = magnitude_mask(values, CompressionRecipe("s", "2:4"), output_axis=0, reduction_axis=1)
    assert (mask.sum(axis=1) == 2).all()
    assert not mask[:, :2].any() and mask[:, 2:].all()
    with pytest.raises(ValueError, match="divisible"):
        magnitude_mask(
            values[:, :3], CompressionRecipe("s", "2:4"), output_axis=0, reduction_axis=1
        )
    with pytest.raises(ValueError, match="finite"):
        magnitude_mask(
            values * float("nan"), CompressionRecipe("s", "2:4"), output_axis=0, reduction_axis=1
        )


@pytest.fixture(scope="module")
def compression_assets(tmp_path_factory):
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    Image = pytest.importorskip("PIL.Image")
    from onnx import TensorProto, helper, numpy_helper

    from edge_ai.compression.accounting import count_nominal
    from edge_ai.datasets.manifest import DatasetManifest, ImageSample
    from edge_ai.datasets.preprocessing import PreprocessingSpec
    from edge_ai.utils.artifacts import ArtifactIdentity, sha256_file

    root = tmp_path_factory.mktemp("compression")
    torch.manual_seed(12)
    torch.set_num_threads(1)

    def model():
        return torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 1), torch.nn.ReLU(), torch.nn.Flatten(), torch.nn.Linear(16, 1000)
        ).eval()

    baseline = model()
    checkpoint = root / "source.pt"
    torch.save(baseline.state_dict(), checkpoint)
    digest = sha256_file(checkpoint)
    pt = ArtifactIdentity("source.pt", digest, "checkpoint", digest)
    nodes = [
        helper.make_node("Conv", ["images", "cw", "cb"], ["conv"]),
        helper.make_node("Relu", ["conv"], ["relu"]),
        helper.make_node("Flatten", ["relu"], ["flat"], axis=1),
        helper.make_node("Gemm", ["flat", "lw", "lb"], ["logits"], transB=1),
    ]
    arrays = [
        numpy_helper.from_array(value.detach().numpy(), name=name)
        for name, value in (
            ("cw", baseline[0].weight),
            ("cb", baseline[0].bias),
            ("lw", baseline[3].weight),
            ("lb", baseline[3].bias),
        )
    ]
    graph = helper.make_graph(
        nodes,
        "fixture",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 2, 2])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 1000])],
        arrays,
    )
    proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    path = root / "source.onnx"
    onnx.save(proto, path)
    ort = ArtifactIdentity(
        "source.onnx", sha256_file(path), "onnx", digest, {"fixture": "generated"}
    )
    spec = PreprocessingSpec(resize_short_edge=2, crop=(2, 2))
    manifests = {}
    for split, color in (("val", (90, 100, 120)), ("train", (20, 80, 60))):
        image = root / f"{split}.JPEG"
        Image.new("RGB", (4, 3), color).save(image)
        manifests[split] = DatasetManifest(
            "imagenette2-160",
            split,
            7,
            (ImageSample(image.name, sha256_file(image), 0, "n01440764"),),
            spec,
            "fixture:generated",
            digest,
            checkpoint.stat().st_size,
        )
    nominal = count_nominal(baseline, (1, 3, 2, 2))
    batches = (manifests["val"].batch(root, 0)[0],)
    return root, pt, ort, nominal, manifests, batches, model, np


@pytest.fixture(scope="module")
def compressed_variants(compression_assets):
    from edge_ai.compression.pruning import prune_onnx
    from edge_ai.compression.quantization import checkpoint_fp16, quantize_onnx

    root, pt, ort, nominal, manifests, batches, _, _ = compression_assets
    results = {}
    for method in ("fp16", "int8", "int4", "unstructured", "channel", "2:4"):
        recipe = CompressionRecipe(
            method.replace(":", "of"),
            method,
            amount=0.25 if method in ("channel", "unstructured") else None,
        )
        destination = root / recipe.id
        if method == "fp16":
            result = checkpoint_fp16(pt, nominal, recipe, destination, workspace=root)
        elif method in ("int8", "int4"):
            result = quantize_onnx(
                ort,
                nominal,
                recipe,
                destination,
                workspace=root,
                verification_batches=batches,
                calibration=manifests["train"] if method == "int8" else None,
                evaluation=manifests["val"],
            )
        else:
            result = prune_onnx(
                ort, nominal, recipe, destination, workspace=root, verification_batches=batches
            )
        results[method] = result
    return results


@pytest.mark.parametrize("method", ["fp16", "int8", "int4", "unstructured", "channel", "2:4"])
def test_real_transforms_identity_coverage_and_kernels(
    compression_assets, compressed_variants, method
):
    from edge_ai.compression.verification import verify_prepared_identity

    root, pt, ort, nominal, _, _, _, _ = compression_assets
    identity, spec = compressed_variants[method]
    identity.verify(root)
    verify_prepared_identity(identity, root)
    assert identity.sha256 != (pt.sha256 if method == "fp16" else ort.sha256)
    assert identity.checkpoint_sha256 == pt.sha256
    assert (spec.parameters, spec.flops) == (nominal.parameters, nominal.flops)
    assert from_dict(type(spec), to_dict(spec)) == spec
    details = spec.details
    if method in ("int8", "int4"):
        assert spec.kernel_support is True and details.kernel_evidence.operator_counts
    elif method == "fp16":
        assert spec.kernel_support is None and identity.role == "compressed_checkpoint"
        import torch

        values = torch.load(identity.verify(root), weights_only=True)
        assert all(value.dtype == torch.float16 for value in values.values())
    else:
        assert spec.kernel_support is False and details.zeros_after > details.zeros_before
        assert "without kernel support" in spec.sparsity_label()
    if method == "int4":
        assert set(details.kernel_evidence.operator_counts) == {"MatMulNBits"}
        assert details.target_elements < details.total_weight_elements
    if method == "2:4":
        assert details.skipped_tensors == ("cw",)
        assert details.masked_elements == 8000 and details.zeros_after == 8000


def test_compression_refuses_overwrite_overlap_and_missing_evidence(
    compression_assets, compressed_variants
):
    from edge_ai.compression.quantization import checkpoint_fp16, quantize_onnx
    from edge_ai.compression.verification import verify_prepared_identity

    root, pt, ort, nominal, manifests, batches, _, _ = compression_assets
    with pytest.raises(FileExistsError):
        checkpoint_fp16(
            pt, nominal, CompressionRecipe("fp16", "fp16"), root / "fp16", workspace=root
        )
    overlapping = replace(manifests["val"], split="train")
    with pytest.raises(ValueError, match="overlaps"):
        quantize_onnx(
            ort,
            nominal,
            CompressionRecipe("bad", "int8"),
            root / "overlap",
            workspace=root,
            verification_batches=batches,
            calibration=overlapping,
            evaluation=manifests["val"],
        )
    assert not (root / "overlap").exists()
    identity, spec = compressed_variants["int8"]
    with pytest.raises(ValueError, match="operator evidence"):
        replace(spec, kernel_support=False)
    with pytest.raises(ValueError, match="kernel evidence"):
        replace(spec.details, kernel_evidence=None)
    profile = root / spec.details.kernel_evidence.profile_path
    original = profile.read_bytes()
    profile.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="profile identity"):
        verify_prepared_identity(identity, root)
    profile.write_bytes(original)


@pytest.mark.parametrize("method", ["fp16", "int8", "int4", "unstructured", "channel", "2:4"])
def test_transformed_cpu_record_reports_and_baseline_attribution(
    compression_assets, compressed_variants, method
):
    pytest.importorskip("matplotlib")
    from edge_ai.config import ExecutionSettings, ExperimentConfig, PowerSettings
    from edge_ai.datasets.workload import Workload
    from edge_ai.devices.profile import capture_device_profile
    from edge_ai.models.variant import ModelVariant
    from edge_ai.profiling.protocol import ProfilingProtocol
    from edge_ai.profiling.runner import profile_cpu
    from edge_ai.records.schema import RunRecord
    from edge_ai.records.store import ResultStore, new_run_id
    from edge_ai.reporting.comparison import baseline_changes
    from edge_ai.reporting.render import render_reports
    from edge_ai.utils.provenance import Provenance

    root, pt, _, nominal, manifests, batches, make_model, np = compression_assets
    identity, spec = compressed_variants[method]
    backend = "pytorch" if method == "fp16" else "onnxruntime"
    model = ModelVariant(
        "fixture-" + spec.details.recipe.id,
        "fixture",
        "classification",
        "generated",
        pt.sha256,
        spec.precision,
        spec,
        (2, 2),
        backend,
        "fixture",
    )
    config = ExperimentConfig(
        "compression_fixture",
        7,
        model,
        Workload(
            "classification",
            "dataset",
            "val.JPEG",
            (2, 2),
            1,
            split="val",
            preprocessing=manifests["val"].preprocessing,
            manifest_sha256=manifests["val"].digest(),
        ),
        protocol=ProfilingProtocol(1, 2, 1),
        execution=ExecutionSettings(),
        power=PowerSettings(),
        artifact=identity,
    )
    if backend == "pytorch":
        import torch

        from edge_ai.runtimes.pytorch import PyTorchRuntime

        def loader(variant, path):
            instance = make_model().half()
            instance.load_state_dict(torch.load(path, weights_only=True))
            return instance

        runtime = PyTorchRuntime(root, identity, config.execution, loader=loader)
        runtime.load(model)
        with pytest.raises(ValueError, match="overflowed"):
            runtime.prepare_batch(np.full((1, 3, 2, 2), 1e10, dtype=np.float32))
        runtime.release()
        version = str(torch.__version__)
    else:
        import onnxruntime

        from edge_ai.runtimes.onnxruntime import ONNXRuntime

        runtime = ONNXRuntime(root, identity, config.execution)
        version = onnxruntime.__version__
    result = profile_cpu(runtime, config, lambda index: (batches[0], 0))
    record = RunRecord(
        new_run_id(config),
        config,
        Provenance("b" * 40, False, "2026-10-13T10:00:00Z", {backend: version}),
        capture_device_profile(),
        result.timings,
        result.memory,
        result.run_duration_s,
        power_trace=result.power_trace,
        energy=result.energy,
        end_to_end_timings=result.end_to_end_timings,
        quality_metric="top1_accuracy",
        quality=result.quality,
        schema_version=4,
        idle_power_trace=result.idle_power_trace,
        dataset_manifest=to_dict(manifests["val"]),
    )
    assert record.energy.energy_j is None and np.isfinite(record.quality)
    reports = render_reports([record])
    store = ResultStore(root)
    path = store.write(record, reports=reports)
    assert store.read(path) == record and (path / "compression.json").exists()
    if spec.details.kernel_evidence is not None:
        assert (path / "traces/compression_kernel_profile.json").read_bytes() == (
            root / spec.details.kernel_evidence.profile_path
        ).read_bytes()
    assert b"Compression coverage" in reports["report.md"]
    with pytest.raises(ValueError, match="schema version 4"):
        replace(record, schema_version=2)
    baseline_model = replace(model, id="baseline", precision="fp32", compression=nominal)
    baseline_config = replace(config, model=baseline_model, artifact=spec.details.source_artifact)
    # Descriptor baseline suffices here; real FP32 baselines are exercised by the public matrix.
    baseline = replace(record, config=baseline_config, schema_version=2, run_id="baseline")
    assert baseline_changes(baseline, record) == {"compression"}
    private_details = replace(
        spec.details,
        source_artifact=replace(spec.details.source_artifact, path="private/source"),
    )
    private_spec = replace(spec, details=private_details)
    private_artifact = replace(
        identity,
        export_metadata={
            **identity.export_metadata,
            "compression_details": to_dict(private_details),
        },
    )
    private_config = replace(
        config, model=replace(model, compression=private_spec), artifact=private_artifact
    )
    private_record = replace(record, config=private_config, run_id="private-leak-test")
    with pytest.raises(ValueError, match="private source"):
        store.write(private_record, reports=reports)
    if spec.details.kernel_evidence is not None:
        for field, value in (
            ("artifact_sha256", "0" * 64),
            ("device_name", "other-host"),
            ("runtime_version", "other-version"),
        ):
            wrong_details = replace(
                spec.details,
                kernel_evidence=replace(
                    spec.details.kernel_evidence,
                    **{field: value},
                ),
            )
            wrong_artifact = replace(
                identity,
                export_metadata={
                    **identity.export_metadata,
                    "compression_details": to_dict(wrong_details),
                },
            )
            wrong_config = replace(
                config,
                model=replace(
                    model,
                    compression=replace(
                        spec,
                        details=wrong_details,
                    ),
                ),
                artifact=wrong_artifact,
            )
            with pytest.raises(ValueError, match="artifact/device/runtime"):
                replace(record, config=wrong_config)


def test_existing_matmul_weights_supported_and_gemm_normalization_lossless(compression_assets):
    import onnx

    from edge_ai.compression.pruning import weight_targets
    from edge_ai.compression.quantization import normalize_gemm, quantize_onnx
    from edge_ai.compression.verification import verify_float_graphs
    from edge_ai.utils.artifacts import ArtifactIdentity, sha256_file

    root, pt, source, nominal, _, batches, _, _ = compression_assets
    model = onnx.load(str(source.verify(root)))
    normalize_gemm(model)
    path = root / "matmul_source.onnx"
    onnx.save_model(model, str(path))
    verify_float_graphs(source.verify(root), path, batches)
    assert ("compression_gemm_3_weight", 1, 0) in weight_targets(model)
    identity = ArtifactIdentity(
        path.name, sha256_file(path), "onnx", pt.sha256, {"fixture": "normalized"}
    )
    derived, spec = quantize_onnx(
        identity,
        nominal,
        CompressionRecipe("matmul", "int4"),
        root / "matmul_quantized",
        workspace=root,
        verification_batches=batches,
    )
    assert spec.details.kernel_evidence.operator_counts["MatMulNBits"] > 0
    assert derived.export_metadata["fp32_preprocessing_parity"]["top1_equal"]


def test_schema_calibration_and_kernel_provenance_fail_closed(
    compression_assets, compressed_variants
):
    from edge_ai.compression.verification import observed_quantized_operators

    _, _, _, _, manifests, _, _, _ = compression_assets
    _, spec = compressed_variants["int8"]
    with pytest.raises(ValueError, match="linear-only INT8"):
        replace(spec.details, recipe=replace(spec.details.recipe, int8_scope="linear"))
    with pytest.raises(ValueError, match="calibration identity"):
        replace(spec, calibration="not-the-manifest")
    with pytest.raises(ValueError, match="integer CPU kernels"):
        replace(
            spec.details,
            kernel_evidence=replace(
                spec.details.kernel_evidence, operator_counts={"MatMulNBits": 1}
            ),
        )
    with pytest.raises(ValueError, match="overlap"):
        spec.details.validate_evaluation(replace(manifests["train"], split="val"))
    profile = b'[{"args":{"provider":"CUDAExecutionProvider","op_name":"QGemm"}}]'
    assert observed_quantized_operators(profile) == {}


def test_pruning_preserves_bias_and_dense_shapes(compression_assets, compressed_variants):
    import onnx
    from onnx import numpy_helper

    root, _, source, _, _, _, _, np = compression_assets
    identity, _ = compressed_variants["channel"]
    before = {
        tensor.name: numpy_helper.to_array(tensor)
        for tensor in onnx.load(str(source.verify(root))).graph.initializer
    }
    after = {
        tensor.name: numpy_helper.to_array(tensor)
        for tensor in onnx.load(str(identity.verify(root))).graph.initializer
    }
    assert before.keys() == after.keys()
    assert all(before[name].shape == after[name].shape for name in before)
    np.testing.assert_array_equal(before["cb"], after["cb"])
    np.testing.assert_array_equal(before["lb"], after["lb"])
    assert (after["lw"] == 0).all(axis=1).sum() == 250


def test_external_onnx_tensors_rejected_before_following_paths(compression_assets):
    import onnx

    from edge_ai.compression.pruning import prune_onnx
    from edge_ai.utils.artifacts import ArtifactIdentity, sha256_file

    root, _, source, nominal, _, batches, _, _ = compression_assets
    model = onnx.load(str(source.verify(root)))
    tensor = model.graph.initializer[0]
    tensor.ClearField("raw_data")
    tensor.data_location = onnx.TensorProto.EXTERNAL
    tensor.external_data.add(key="location", value="private/unavailable.bin")
    path = root / "external.onnx"
    path.write_bytes(model.SerializeToString())
    identity = ArtifactIdentity(
        path.name,
        sha256_file(path),
        "onnx",
        source.checkpoint_sha256,
        {"fixture": "external-refused"},
    )
    with pytest.raises(ValueError, match="inline tensors"):
        prune_onnx(
            identity,
            nominal,
            CompressionRecipe("external", "unstructured", 0.25),
            root / "external_result",
            workspace=root,
            verification_batches=batches,
        )
    assert not (root / "external_result").exists()


def test_int8_linear_retains_conv_fp32_and_records_partial_integer_coverage(compression_assets):
    import onnx
    from onnx import numpy_helper

    from edge_ai.compression.quantization import quantize_onnx
    from edge_ai.utils.artifacts import sha256_file

    root, _, source, nominal, manifests, batches, _, np = compression_assets
    identity, spec = quantize_onnx(
        source, nominal, CompressionRecipe("linear", "int8", int8_scope="linear"),
        root / "linear", workspace=root, verification_batches=batches,
        calibration=manifests["train"], evaluation=manifests["val"],
    )
    before = {tensor.name: numpy_helper.to_array(tensor)
              for tensor in onnx.load(str(source.verify(root))).graph.initializer}
    after = {tensor.name: numpy_helper.to_array(tensor)
             for tensor in onnx.load(str(identity.verify(root))).graph.initializer}
    np.testing.assert_array_equal(before["cw"], after["cw"])
    np.testing.assert_array_equal(before["cb"], after["cb"])
    assert after["cw"].dtype == np.float32
    assert spec.details.changed_tensors == ("lw_quantized",)
    assert spec.details.target_elements < spec.details.total_weight_elements
    assert "QGemm" in spec.details.kernel_evidence.operator_counts
    assert "QLinearConv" not in spec.details.kernel_evidence.operator_counts
    assert identity.export_metadata["settings"]["op_types"] == ["Gemm", "MatMul"]
    assert sha256_file(root / source.path) == source.sha256
