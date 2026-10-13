# Running the Phase 2 CPU reference

The approved pipeline runs MobileNetV3 Small/Large with explicit `IMAGENET1K_V1`
FP32 weights on 20 seeded validation images from Imagenette v2, two per class.
Both CPU backends use RGB/NCHW inputs, batch 1, 224 x 224, the same resize/crop/
normalization, five warm-up calls and 20 measured inferences per run.
The 160px source images are upsampled as prescribed by the selected weights.

## Tested environment

Windows, Python 3.10.11:

| Dependency | Version |
| --- | --- |
| PyTorch / Torchvision | 2.9.1+cpu / 0.24.1+cpu |
| NumPy / Pillow | 2.2.6 / 11.3.0 |
| ONNX / ONNXScript / ONNX IR | 1.19.1 / 0.5.7 / 0.1.12 |
| ONNX Runtime (CPU) | 1.23.2 |
| psutil / matplotlib | 7.2.2 / 3.10.9 |

Create/activate the environment described in the README, then install CPU wheels
first so they satisfy the declared extras:

```sh
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev,cpu,onnx,reporting]"
python -m pip check
```

The CPU route is documented by [PyTorch](https://pytorch.org/get-started/previous-versions/#v291).
Core-only users can install `.[dev]`; heavy imports remain lazy. Prepared ONNX
inference imports neither PyTorch nor Torchvision. ONNX IR 1.0 was incompatible
with this exporter stack during verification, so 0.1.12 is pinned.

## Preparation and execution

From the repository root:

```sh
python -m edge_ai.reference prepare --config configs/experiments/reference_cpu.yaml
```

Preparation downloads the full archive (99,003,388 bytes) and two checkpoints
(10,306,551 and 22,139,423 bytes), verifies upstream checksums, extracts data,
selects the manifest and exports ONNX. Reserve about 1 GB for data/artifacts/
partial preparations/results in addition to the environment. Full SHA-256
identities are saved for the archive, selected images, checkpoints and exports.

The command prints `artifacts/reference/<config-and-version-hash>/prepared.json`.
Matching prepared assets are verified/reused. Failed preparations remain in place;
retries use fresh directories. Existing files are never silently overwritten.
Extraction rejects traversal/links and refuses conflicting existing contents.

Run using your printed path. In the verified environment:

```sh
python -m edge_ai.reference run --prepared artifacts/reference/9029ef52961e20ad/prepared.json
```

Running never downloads or exports implicitly. Prepared assets must match the
config, stratified subset, image hashes and model hashes. Each configuration runs
in a fresh subprocess with explicitly recorded thread/session settings.

The first complete four-run CLI took **16.795 seconds** here, including worker
startup, local loading, preprocessing, warm-up, inference, quality, reporting and
storage. Downloads/export are separate setup costs. This verifies the approximately
two-minute target on this laptop, not performance on other machines.

Four new directories under `results/reference/<run_id>/` contain resolved config,
device profile, schema-v2 record, raw model/end-to-end/RSS/active/idle traces,
Markdown/CSV table, comparison peer IDs and two Pareto PNGs. Reports compare
Small/Large within each runtime. Cross-runtime reports require an explicit
`allow_mismatch=True` and show the differing settings in tables and plots.

## Numerical and measurement limits

Exports use `dynamo=True`, static shape/batch, opset 18 and no external weights.
Both models were checked on all 20 selected real inputs: identical top-1 outputs
and logits within `rtol=1e-4, atol=5e-5`. Initial `atol=1e-5` rejected a near-zero
logit difference of `1.78e-5`; the final observed maximum absolute differences
were `4.63e-5` for Small and `1.83e-5` for Large. Export metadata records versions,
tolerances, observed error and artifact identity; tolerances never change silently.

Quality uses the full 1000-class output with checked Imagenette WordNet-ID mapping.
The 20-image subset validates plumbing and is too small for research conclusions.
Nominal FLOPs are partial Conv2d/Linear counts (MAC = two FLOPs, bias adds included),
with exclusions explicitly listed. Memory is sampled total process RSS, including
libraries/input/runtime, and may miss short peaks; allocator/VRAM values remain null.

Power is explicitly assumed constant watts over the observed measurement window.
A monotonic timestamp does not make a synthetic value measured. Total, per-inference
and above-idle estimates preserve that provenance. **No measured energy or energy
savings are reported.** See [profiling boundaries](profiling_protocol.md).

## Verification and contracts

```sh
python -m pytest -q
python -m ruff check src tests
python -m pytest tests -q --reference-prepared artifacts/reference/9029ef52961e20ad/prepared.json
```

Default tests use offline generated models/images and never download. Real CPU
tests skip if extras are absent. The opt-in public smoke must pass for phase
acceptance; specify `tests` to discover its extra option. Symlink tests may skip
when Windows denies directory symlink creation.

The suite has a typed `ReferenceConfig` loader. Each resolved `ExperimentConfig`
contains model/workload/protocol, execution/power settings and deployment artifact.
`RunRecord` version 2 adds raw idle/RSS provenance and the embedded dataset manifest.
Version 1 remains readable without rewriting old files. Canonical learned weights
are separate from ONNX deployment identity so a backend change is labeled runtime,
not model compression. Nominal count changes from architecture size are model
metadata, not an applied compression transformation.

`ResultStore.write(record, reports=...)` writes bounded report bytes before the
completion marker. `load_stored_comparison(store, run)` refuses missing/incomplete
peers. A completed individual run does not establish completion of its group.
Immutability is enforced by the API, not filesystem permissions against manual
edits. Known private paths are checked; remote source classification remains the
caller's responsibility.

All downloaded assets, prepared artifacts and runs stay local and are ignored by
Git. Upstream attribution is preserved; weights/images are not redistributed.
Phase 3A adds [sustained CPU profiling and offline telemetry](phase_3a_usage.md).
Jetson execution remains Phase 3B; [CPU portions of phases 4–6](cpu_hardware_roadmap.md)
may proceed through their own reviews while device validation is pending.
See [Phase 4 compression usage](phase_4_usage.md) for transforming and comparing
derived variants from the same verified assets, including method/quality limitations.
