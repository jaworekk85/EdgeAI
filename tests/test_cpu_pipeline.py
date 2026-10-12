import json
import subprocess
import sys
from dataclasses import replace

import pytest

from edge_ai.config import ExecutionSettings, ExperimentConfig, PowerSettings
from edge_ai.datasets.manifest import IMAGENETTE_LABELS, DatasetManifest, ImageSample
from edge_ai.datasets.preprocessing import PreprocessingSpec, preprocess_image
from edge_ai.datasets.workload import Workload
from edge_ai.export.onnx import export_onnx
from edge_ai.models.variant import ModelVariant
from edge_ai.profiling.protocol import ProfilingProtocol
from edge_ai.profiling.runner import profile_cpu
from edge_ai.records.schema import RunRecord
from edge_ai.records.store import ResultStore, new_run_id
from edge_ai.reporting.comparison import comparison_rows
from edge_ai.reporting.render import pareto_indices, render_reports
from edge_ai.utils.artifacts import ArtifactIdentity, sha256_file
from edge_ai.utils.provenance import Provenance
from edge_ai.utils.serialization import from_dict, to_dict


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    pytest.importorskip("onnxruntime")
    Image = pytest.importorskip("PIL.Image")
    root = tmp_path_factory.mktemp("cpu-pipeline")
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 2, 1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(2, 1000),
    ).eval()
    checkpoint = root / "weights.pt"
    torch.save(model.state_dict(), checkpoint)
    digest = sha256_file(checkpoint)
    artifact = ArtifactIdentity("weights.pt", digest, "checkpoint", digest)
    spec = PreprocessingSpec(resize_short_edge=8, crop=(8, 8))
    image = root / "image.JPEG"
    Image.new("RGB", (12, 9), (120, 90, 60)).save(image)
    sample = ImageSample("image.JPEG", sha256_file(image), 0, "n01440764")
    manifest = DatasetManifest(
        "imagenette2-160",
        "val",
        7,
        (sample,),
        spec,
        "fixture:generated-image",
        digest,
        checkpoint.stat().st_size,
    )
    (root / "manifest.json").write_text(json.dumps(to_dict(manifest)), encoding="utf-8")
    from edge_ai.compression.accounting import count_nominal

    nominal = count_nominal(model, (1, 3, 8, 8))
    variant = ModelVariant(
        "tiny",
        "fixture",
        "classification",
        "generated-fixture",
        digest,
        "fp32",
        nominal,
        (8, 8),
        "pytorch",
        "tiny",
    )
    example = preprocess_image(image, spec)
    onnx = export_onnx(
        model,
        torch.from_numpy(example),
        root / "export",
        workspace=root,
        checkpoint_sha256=digest,
        verification_batches=(example,),
    )
    config = ExperimentConfig(
        "offline",
        7,
        variant,
        Workload(
            "classification",
            "dataset",
            "manifest.json",
            (8, 8),
            1,
            preprocessing=spec,
            manifest_sha256=manifest.digest(),
        ),
        protocol=ProfilingProtocol(2, 3, 1),
        execution=ExecutionSettings(),
        power=PowerSettings(),
        artifact=artifact,
    )
    return root, model, config, manifest, onnx, np


def make_record(assets, backend, *, sustained_duration_s=None, sustained_max_iterations=100000):
    root, model, config, manifest, onnx, _ = assets
    if sustained_duration_s is not None:
        config = replace(config, protocol=replace(
            config.protocol, sustained_duration_s=sustained_duration_s,
            sustained_max_iterations=sustained_max_iterations,
        ))
    if backend == "pytorch":
        from edge_ai.runtimes.pytorch import PyTorchRuntime

        runtime = PyTorchRuntime(root, config.artifact, config.execution, loader=lambda *_: model)
    else:
        from edge_ai.runtimes.onnxruntime import ONNXRuntime

        config = replace(config, model=replace(config.model, runtime="onnxruntime"), artifact=onnx)
        runtime = ONNXRuntime(root, onnx, config.execution)
    from edge_ai.devices.profile import capture_device_profile

    result = profile_cpu(runtime, config, lambda index: manifest.batch(root, index))
    record = RunRecord(
        new_run_id(config),
        config,
        Provenance("b" * 40, False, "2026-10-11T10:00:00Z", {backend: "fixture"}),
        capture_device_profile(),
        result.timings,
        result.memory,
        result.run_duration_s,
        power_trace=result.power_trace,
        energy=result.energy,
        end_to_end_timings=result.end_to_end_timings,
        quality_metric="top1_accuracy",
        quality=result.quality,
        schema_version=3 if result.sustained is not None else 2,
        idle_power_trace=result.idle_power_trace,
        dataset_manifest=to_dict(manifest),
        sustained=result.sustained,
    )
    return record


@pytest.mark.parametrize("backend", ["pytorch", "onnxruntime"])
def test_real_cpu_pipeline_records_reports_and_immutability(assets, backend):
    pytest.importorskip("matplotlib")
    record = make_record(assets, backend)
    assert len(record.timings.durations_s) == 3
    assert record.timings.measured and record.energy.energy_j is None
    assert record.memory.peak_ram_bytes > 0 and record.memory.rss_bytes
    reports = render_reports([record])
    root = assets[0]
    store = ResultStore(root)
    run = store.write(record, reports=reports)
    assert store.read(run) == record
    assert (run / "plots" / "quality_vs_energy.png").read_bytes().startswith(b"\x89PNG")
    assert b"synthetic" in (run / "report.md").read_bytes()
    assert (run / "traces" / "idle_power.json").exists()
    # Actual Phase 2 records omit fields introduced in 3A; no stored file is migrated.
    legacy = to_dict(record)
    legacy.pop("sustained")
    legacy["config"]["protocol"].pop("sustained_max_iterations")
    assert from_dict(RunRecord, legacy) == record
    with pytest.raises(FileExistsError):
        store.write(record, reports=reports)
    with pytest.raises(ValueError, match="complete reports"):
        store.write(replace(record, run_id="other"))


def test_partial_nominal_counter_uses_declared_convention(assets):
    compression = assets[2].model.compression
    # Conv: 1*2*8*8*(2*3 + bias 1); Linear: 1000*(2*2 + bias 1).
    assert compression.flops == 128 * 7 + 1000 * 5
    assert compression.parameters == 8 + 3000
    assert "pooling" in compression.counting_exclusions


@pytest.mark.parametrize("backend", ["pytorch", "onnxruntime"])
def test_sustained_record_store_reports_and_validation(assets, backend):
    pytest.importorskip("matplotlib")
    record = make_record(assets, backend, sustained_duration_s=10, sustained_max_iterations=6)
    assert record.schema_version == 3 and record.sustained.stop_reason == "inference_cap"
    assert len(record.timings.durations_s) == 6 and record.energy.inference_count == 6
    assert from_dict(RunRecord, to_dict(record)) == record
    store = ResultStore(assets[0])
    reports = render_reports([record])
    run = store.write(record, reports=reports)
    assert store.read(run) == record
    assert (run / "traces" / "sustained.json").is_file()
    assert b"inference_cap" in reports["report.md"]
    assert b"thermal cause is unknown" in reports["report.md"]
    with pytest.raises(FileExistsError):
        store.write(record, reports=reports)
    with pytest.raises(ValueError, match="summary differs"):
        replace(record, sustained=replace(record.sustained, stop_reason="duration_reached"))
    with pytest.raises(ValueError, match="schema version 3"):
        replace(record, schema_version=2)
    with pytest.raises(ValueError, match="timing count"):
        replace(record, timings=replace(
            record.timings, starts_s=record.timings.starts_s[:3],
            durations_s=record.timings.durations_s[:3],
        ))
    with pytest.raises(ValueError, match="inference count"):
        replace(record, energy=replace(
            record.energy, inference_count=3,
            per_inference_estimate_j=record.energy.energy_estimate_j / 3,
        ))
    short = make_record(assets, backend)
    with pytest.raises(ValueError, match="profiling_protocol"):
        comparison_rows([short, record])
    rows = comparison_rows([short, record], allow_mismatch=True)
    assert "profiling_protocol" in rows[0]["settings_mismatch"]


def test_onnx_backend_import_and_inference_without_torch(assets):
    root, _, config, manifest, onnx, _ = assets
    code = """
import importlib.abc, json, sys
from pathlib import Path
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'torchvision'):
            raise RuntimeError('Unexpected Torch import')
sys.meta_path.insert(0, NoTorch())
from edge_ai.config import ExecutionSettings
from edge_ai.models.variant import ModelVariant
from edge_ai.runtimes.onnxruntime import ONNXRuntime
from edge_ai.utils.artifacts import ArtifactIdentity
from edge_ai.utils.serialization import from_dict
import numpy as np
root = Path(sys.argv[1])
data = json.loads(sys.argv[2])
runtime = ONNXRuntime(root, from_dict(ArtifactIdentity, data['artifact']), ExecutionSettings())
runtime.load(from_dict(ModelVariant, data['model']))
batch = runtime.prepare_batch(np.zeros((1, 3, 8, 8), dtype=np.float32))
assert runtime.infer(batch).shape == (1, 1000)
runtime.release()
"""
    changed = replace(config.model, runtime="onnxruntime")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(root),
            json.dumps({"model": to_dict(changed), "artifact": to_dict(onnx)}),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_preprocessing_matches_torchvision_v1(assets):
    torch = pytest.importorskip("torch")
    from PIL import Image
    from torchvision.models import MobileNet_V3_Small_Weights

    image = assets[0] / "image.JPEG"
    actual = preprocess_image(image, PreprocessingSpec())
    with Image.open(image) as source:
        expected = MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()(source.convert("RGB"))
    torch.testing.assert_close(torch.from_numpy(actual[0]), expected, rtol=0, atol=0)


def test_full_class_mapping_is_independent_of_ten_class_positions():
    assert list(IMAGENETTE_LABELS.values()) == [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]
    with pytest.raises(ValueError, match="ImageNet-1K"):
        ImageSample("fixture.JPEG", "a" * 64, 1, "n02102040")


def test_class_indices_match_official_pretrained_output_categories():
    pytest.importorskip("torchvision")
    from torchvision.models import MobileNet_V3_Small_Weights

    expected = [
        "tench",
        "English springer",
        "cassette player",
        "chain saw",
        "church",
        "French horn",
        "garbage truck",
        "gas pump",
        "golf ball",
        "parachute",
    ]
    categories = MobileNet_V3_Small_Weights.IMAGENET1K_V1.meta["categories"]
    assert [categories[index] for index in IMAGENETTE_LABELS.values()] == expected


def test_version_two_new_provenance_cannot_be_omitted(assets):
    record = make_record(assets, "onnxruntime")
    with pytest.raises(ValueError, match="version 2"):
        replace(record, schema_version=1)
    with pytest.raises(ValueError, match="complete execution/data"):
        replace(record, dataset_manifest=None)
    with pytest.raises(ValueError, match="idle power"):
        replace(record, energy=replace(record.energy, idle_power_w=2))


def test_runtime_changes_are_not_claimed_as_algorithm_changes(assets):
    old = make_record(assets, "pytorch")
    new = make_record(assets, "onnxruntime")
    new = replace(
        new, config=replace(new.config, baseline_id=old.run_id, changed_components=("runtime",))
    )
    assert comparison_rows([old, new], allow_mismatch=True)[1]["changed_components"] == ("runtime",)


def test_reports_annotate_overrides_in_table_and_png(assets):
    pytest.importorskip("matplotlib")
    reports = render_reports(
        [make_record(assets, "pytorch"), make_record(assets, "onnxruntime")], allow_mismatch=True
    )
    assert b"runtime" in reports["report.csv"]
    assert b"Settings mismatch" in reports["plots/quality_vs_latency.png"]
    assert b"onnxruntime" in reports["plots/quality_vs_energy.png"]


def test_pareto_ties_and_dominated_points():
    assert pareto_indices([1, 2, 3, 1], [0.5, 0.4, 0.9, 0.5]) == [0, 2, 3]
    with pytest.raises(ValueError):
        pareto_indices([float("nan")], [0.5])


def test_checkpoint_digest_and_invalid_batch_fail(assets):
    root, model, config, _, onnx, np = assets
    from edge_ai.runtimes.onnxruntime import ONNXRuntime

    runtime = ONNXRuntime(root, onnx, config.execution)
    runtime.load(replace(config.model, runtime="onnxruntime"))
    with pytest.raises(ValueError):
        runtime.prepare_batch(np.zeros((1, 3, 4, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        runtime.prepare_batch(np.zeros((1, 3, 8, 8), dtype=np.float64))
    runtime.release()
    with pytest.raises(RuntimeError):
        runtime.infer(np.zeros((1, 3, 8, 8), dtype=np.float32))
    bad = replace(config.artifact, sha256="a" * 64, checkpoint_sha256="a" * 64)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        bad.verify(root)


def test_legacy_record_without_new_fields_still_reads(record):
    data = to_dict(record)
    for field in ("idle_power_trace", "dataset_manifest"):
        data.pop(field)
    for field in ("artifact", "execution", "power"):
        data["config"].pop(field)
    data["config"]["model"].pop("architecture")
    data["config"]["model"]["compression"].pop("counting_exclusions")
    data["config"]["workload"].pop("preprocessing")
    data["config"]["workload"].pop("manifest_sha256")
    for field in ("sampling_interval_s", "sample_times_s", "rss_bytes", "unavailable"):
        data["memory"].pop(field)
    assert from_dict(RunRecord, data) == record


def test_group_reader_refuses_missing_peers(assets):
    pytest.importorskip("matplotlib")
    from edge_ai.reporting.comparison import load_stored_comparison

    first = make_record(assets, "onnxruntime")
    second = replace(first, run_id=new_run_id(first.config))
    reports = render_reports([first, second])
    store = ResultStore(assets[0])
    first_path = store.write(first, reports=reports)
    with pytest.raises(ValueError, match="not completely stored"):
        load_stored_comparison(store, first_path)
    store.write(second, reports=reports)
    assert len(load_stored_comparison(store, first_path)) == 2


def test_public_record_cannot_embed_private_image_paths(assets):
    record = make_record(assets, "onnxruntime")
    manifest = from_dict(DatasetManifest, record.dataset_manifest)
    manifest = replace(manifest, samples=(replace(manifest.samples[0], path="private/image.JPEG"),))
    config = replace(
        record.config,
        workload=replace(
            record.config.workload,
            manifest_sha256=manifest.digest(),
        ),
    )
    record = replace(record, config=config, dataset_manifest=to_dict(manifest))
    with pytest.raises(ValueError, match="private source"):
        ResultStore(assets[0]).write(record, reports=render_reports([record]))
