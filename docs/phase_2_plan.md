# Phase 2 proposal: CPU runtimes and reference pipeline

Status: approved by the user and implemented across 2026-10-11/12. This document
preserves the reviewed design; see [CPU usage](cpu_reference.md) for implemented
commands, tested versions and numerical tolerances. Phase 1 is on GitHub at `a68189e`.

## Outcome and scope

One reference configuration runs two pretrained model sizes through PyTorch and
ONNX Runtime on a CPU, captures quality, timing, memory and synthetic energy,
then writes immutable records, a comparison table and Pareto plots. This is
pipeline validation; its small-sample values are not research findings.

Keep the standard-library/PyYAML core and existing mocks runnable without any
inference libraries. No training, quantization/pruning, adaptive control, condition
transforms, TensorRT, thermal experiments or sustained hardware runs in Phase 2.

## Proposed reference choices

| Setting | Proposal |
| --- | --- |
| Family / architectures | MobileNetV3 Small and Large |
| Weights | Explicit Torchvision `IMAGENET1K_V1` for both; never moving `DEFAULT` |
| Precision | FP32 in both backends |
| Data | Imagenette v2, 160px source archive, validation split |
| Smoke subset | Two images per class, 20 total; seeded selection from sorted paths |
| Model input | RGB, NCHW, 224 x 224, batch 1 |
| Preprocessing | Bilinear resize of shorter edge to 256, center crop 224, ImageNet normalization |
| Quality | Top-1 accuracy among all 1000 ImageNet classes |
| Protocol | Five warm-up calls, 20 measured batches, one repetition per run |
| CPU execution | One intra-op thread; sequential execution; settings saved explicitly |
| Matrix | Small/Large x PyTorch/ONNX Runtime = four separate records |
| Power | Explicit constant synthetic power, with synthetic idle baseline |

Small and Large have distinct published pretrained variants. Both V1 weight
sets use the same resize/crop/normalization, whereas Large's current default
selects V2. Choosing V1 explicitly avoids an accidental preprocessing change.
See the official [Small](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html)
and [Large](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_large.html)
weight documentation.

Imagenette comprises ten ImageNet classes, as documented by its
[maintainers](https://github.com/fastai/imagenette). Its local ten-class labels
must be mapped to the original 1000-class output indices; logits must not be
restricted to ten classes. Use a documented, tested WordNet-ID mapping and save
the mapping, selected relative paths and SHA-256 hashes in a dataset manifest.
The archive URL/checksum is available in the official
[Torchvision loader](https://docs.pytorch.org/vision/stable/_modules/torchvision/datasets/imagenette.html).
The 160px source is intentionally upsampled by the prescribed preprocessing;
the archive size and transformation are recorded, not described as native 224px data.

## Preparation and repeatable execution

Separate asset preparation from benchmark execution. A preparation command will
download the public archive and two checkpoints to ignored `data/` and `artifacts/`,
validate downloads, select the manifest, compute checkpoint/image SHA-256 values,
and export ONNX artifacts. Only prepared assets are used by the benchmark; it
fails with instructions if assets are missing instead of downloading implicitly.

The archive download contains more images than the 20-image smoke subset. Check
its advertised size before downloading, record its actual size afterward, and
state the storage requirement in setup instructions. Do not commit images, weights
or generated results. Cache names include artifact/config identities; matching
artifacts are verified and reused, different artifacts get new paths. No cached
file is silently overwritten. Extract archives only within the intended data root.

Use the supported `torch.onnx.export(..., dynamo=True)` path and explicitly record
exporter settings, opset, input/output names, static batch/input shape, versions and
artifact SHA-256. Verify ONNX structure and PyTorch/ORT logits on real prepared
inputs before benchmarking. FP32 parity starts at `rtol=1e-4`, `atol=1e-5` and
top-1 equality; any adjustment requires a documented numerical reason.
The exporter approach follows the official
[PyTorch tutorial](https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html).

The smoke target of approximately two minutes applies to a full four-run command
with prepared assets: worker startup, loading, preprocessing, warm-up, inference,
quality evaluation, report generation and immutable storage. Downloads and first
export are separate setup costs and are timed/documented separately. Achieving
that target was verified on this laptop (first full CLI: 16.795 s).

## Modules and contracts

| Module | Proposed responsibility |
| --- | --- |
| `models` | Explicit MobileNetV3 constructors, verified checkpoint loading and identity |
| `export` | Exclusive ONNX artifact creation, export metadata and parity verification |
| `runtimes/pytorch.py` | Lazy PyTorch import, CPU eval/inference mode, native batch preparation, lifecycle |
| `runtimes/onnxruntime.py` | Lazy ORT import, explicit CPU provider, native arrays, lifecycle |
| `profiling` | Serial warm-up/measurement runner, separate timing scopes, RSS sampling |
| `datasets` | Manifest selection/loading and shared deterministic preprocessing |
| `evaluation` | Top-1 correctness/count aggregation, outside model-only timing |
| `compression` | One shared nominal parameter/FLOP counter; no model transformations |
| `power` | A synthetic source covering the actual measurement window, distinct from the Phase 1 fixture |
| `reporting` | Validated Markdown/CSV tables and headless Pareto plots |
| `records` | Versioned record additions and writing all artifacts before completion |
| `experiments/reference` | Small orchestration entry point with explicit backend choices |

Extend the runtime protocol with `prepare_batch(batch)` and `synchronize()`.
Warm-up still uses an explicitly bound representative native batch. CPU inference
is synchronous; synchronization is a no-op for these adapters and provides a
later hardware timing boundary. `infer` returns native logits; evaluation/output
conversion happens outside the model timing window. Every failure path releases
the runtime and stops sampling. Mock implementations cover these contracts too.

ORT uses `CPUExecutionProvider` explicitly and recorded session/thread options,
following its [Python API](https://onnxruntime.ai/docs/api/python/api_summary.html)
and [thread settings](https://onnxruntime.ai/docs/performance/tune-performance/threading.html).
The ORT backend must load prepared ONNX without importing PyTorch. Shared image
preprocessing will use Pillow/NumPy and be checked against the selected Torchvision
weight transforms. Preparation may use Torchvision; runtime dispatch stays a
small explicit conditional, with no plugin registry.

Run each configuration in a fresh subprocess to avoid carrying allocator state
from one model/backend into another. Capture the device profile and provenance
inside the worker. Parent orchestration selects assets and records worker failures;
it must not relabel failed jobs as completed runs.

## Timing, quality and memory boundaries

Capture `perf_counter` timestamps on one monotonic time basis. Perform warm-up
before opening the power/timed window. For each measured item, record model-only
time around the prepared native input and completed inference. Independently record
end-to-end time from image loading through preprocessing, inference and output
materialization. Task metric aggregation and report generation remain outside
both scopes. Save one raw sample per configured inference, with excluded warm-up
count, scope and units. Quality is accumulated from those measured predictions;
no extra uncounted inference contributes to the power window.

Measure artifact file size and sampled process RSS using optional `psutil` at a
recorded interval, initially 10 ms. Start RSS sampling before model load, stop
after the run, and label the peak as a sampled process peak including libraries,
inputs and runtime state. A sample interval can miss short-lived peaks; neither
Python `tracemalloc` nor file size substitutes for native allocation measurements.
CPU VRAM and unavailable runtime allocator totals remain null with reasons.

Use a single shared nominal counter on the loaded PyTorch architecture before
timing. Count parameters and convolution/linear FLOPs under an explicit convention
(one multiply-add = two FLOPs; bias adds included). Label this as partial operator
coverage, list exclusions such as activations/normalization/pooling, and retain
the input shape/method. Reuse that architecture accounting for the matching ONNX
artifact; do not copy a published FLOP number with an unknown counting convention.

## Energy provenance and record evolution

The Phase 1 `MockPower` creates a fixed-duration trace independent of execution.
Keep that deterministic test fixture intact. Add a separate constant-power mock
whose timestamped synthetic values span the observed measurement window. It uses
the same monotonic clock as profiling but retains `measured=False`; clock basis
does not turn a modeled power value into a physical observation.

Save its assumed watts, seed/config, sampling policy and window. Energy comes
only from integrating this explicitly synthetic trace and remains in
`energy_estimate_j`, with `estimate_method`. Window-average per-inference estimates
divide by the actual counted inferences. Idle power has a separate raw trace and
synthetic provenance. Above-idle reports preserve estimate labels. These plots
test report plumbing and cannot establish device efficiency or energy savings.

Add small typed fields for execution settings, preprocessing, dataset manifest
identity, deployment artifact identity/export metadata, raw idle trace, memory
sampling provenance and nominal-count coverage. Keep canonical checkpoint identity
separate from ONNX artifact identity so changing deployment runtime is not
misclassified as changing the learned model.

Introduce record schema version 2 for real Phase 2 runs. The reader continues to
validate/read version 1 without editing stored files; old optional observations
remain unavailable. Version 2 validation requires the new provenance for real
backends and retains every section 9 invariant. Completion markers carry and
match the record version. Migration/round-trip tests include saved version 1 fixtures.

## Immutable reports and comparisons

`ResultStore.write` currently completes a run before any plots exist. Extend it
to accept a bounded set of prepared report artifacts, with safe relative names
and exclusive writes. Stage table/plot bytes before storage and write them with
the record before `complete.json`. Never append reports to a completed run.

Default comparisons are grouped by runtime: Small versus Large within PyTorch,
and Small versus Large within ORT. Each run stores its table and the corresponding
group's quality-versus-latency and quality-versus-energy-estimate plots, with peer
run IDs. Prepare the group's records and reports before finalizing its storage.
Partial storage failure remains visible; referencing an unavailable peer is not
treated as a fully stored group.

Cross-runtime comparisons require explicit `allow_mismatch=True`. Differing
settings appear in the table and on the plots, not just in console output. Also
check workload manifest, preprocessing, CPU thread settings and quality metric;
matching resolution/runtime alone is insufficient for quality comparisons.

Tables retain nominal parameters/FLOPs and sparsity support beside measured latency,
memory, modeled power and estimated energy. Pareto front calculation is tested for
ties/dominated points and uses only compatible, available quantities. Show a
prominent synthetic-energy label and no measured-energy point for CPU mock power.
Use per-inference window-average energy on the energy axis, not unequal run totals.

## Dependencies and validation

Keep core dependencies unchanged. Add optional CPU inference/export/reporting
extras: PyTorch/Torchvision, NumPy, Pillow, ONNX/ONNXScript, CPU ONNX Runtime,
psutil and matplotlib. Pin a compatible tested stack after checking Python 3.10
Windows CPU wheels; use the official CPU installation route rather than
implicitly downloading a CUDA stack. Document the exact tested versions and
capture versions in each run. The verified installed stack is in `cpu_reference.md`.

Verification gates before stopping Phase 2:

1. Existing CPU/mock invariant tests still pass with inference dependencies absent.
2. Fake-clock tests prove warm-up exclusion, correct timing scopes/counts, power
   window coverage, and cleanup after load/inference/sampler/report failures.
3. Offline tiny generated model fixtures exercise real PyTorch and ORT adapters,
   ONNX export/parity, input validation and record version compatibility.
4. Independent class-mapping tests catch ten-class/1000-class label confusion;
   selected manifests and preprocessing are deterministic and checksummed.
5. Report tests cover mismatch annotations, sparsity/nominal fields, unavailable
   metrics, measured/estimated separation and plot generation before completion.
6. An opt-in prepared-asset reference smoke executes all four public pretrained
   runs, records total wall time, produces standard outputs and stays under the
   approximately two-minute target on this laptop. It must run for phase acceptance;
   a skipped optional test alone does not establish success.
7. Run Ruff, dependency checks and repository whitespace checks, update `STATUS.md`,
   and stop for review before any Phase 3 work.

The repository distribution license remains an open question from Phase 0.
Preserve upstream asset attribution and do not infer image redistribution terms
from the dataset repository's software license. No assets are redistributed by
this proposal.
