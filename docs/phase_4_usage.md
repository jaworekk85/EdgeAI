# Phase 4 CPU compression usage

Phase 4 adds CPU compression tools and a small reference validation. Use the
[existing CPU dependency setup](cpu_reference.md); no new dependencies, Jetson,
CUDA, TensorRT, asset download or training is required. The [design](phase_4_plan.md)
was recorded before implementation. Hardware validation remains in 3B.

## Prepare and run

The default [matrix](../configs/compression/reference_cpu.yaml) uses MobileNetV3
Small with FP16, INT8, weight-only INT4 and three pruning recipes. Preparation
creates a new directory, validates the original prepared assets, creates derived
artifacts, and checks inference. It never overwrites an existing preparation.

```sh
python -m edge_ai.compression.reference prepare --reference-prepared artifacts/reference/9029ef52961e20ad/prepared.json
```

Use the printed `artifacts/compression/<id>/prepared.json` path:

```sh
python -m edge_ai.compression.reference run --prepared artifacts/compression/<id>/prepared.json
```

The measured matrix contains eight new runs: PyTorch FP32/FP16, and ORT FP32,
INT8, INT4, unstructured, channel and 2:4 pruning. Workers have separate processes;
thread/session settings, warm-up, inputs and measurement protocol match the CPU
reference. Preparation/calibration are outside benchmark timing and energy.
One full validation matrix completed in 23.046 seconds on this laptop.

Reports compare within each runtime, use a fresh FP32 baseline for every group,
and identify derived variants as **compression changes**. No cross-runtime speed
comparison is implied. The original twenty validation images are reused, with
quality, real latency, artifact bytes, sampled process RSS, nominal dense
parameters/FLOPs and synthetic energy estimates saved together. Runs are pipeline
validation, not research findings.

Customize recipes, amounts, INT4 block size or calibration images per class in
a separate YAML under `configs/` and pass `--config`. A supported architecture
must already exist in the original prepared reference. Changed matrices require
new preparation. There is no automatic grid search, quality tuning or retraining.
An invalid, incomplete, missing-variant or tampered matrix fails before workers
start. Failed preparations are retained; rerun creates a different directory.
Public ONNX inputs must store tensors inline. External tensor file references,
including nested constants, are rejected before any referenced files are loaded.

## Method boundaries

| Method | Implemented behavior | Interpretation |
| --- | --- | --- |
| FP16 | Convert floating state tensors, including floating buffers, to half; load a half PyTorch model and half inputs | File size/tensor dtype are verified. CPU kernel internal arithmetic/acceleration is unknown; FP16 may be slower |
| INT8 | Static signed QDQ with per-channel weights and MinMax calibration; eligible constant Conv/Gemm/MatMul weights | Actual quantized CPU operator executions are observed; other operations can remain FP32. This is a small calibration, not accuracy optimization |
| INT4 | Symmetric RTN blockwise MatMulNBits, accuracy level 4, configurable power-of-two block size >=16 | Weight-only, partial coverage; activations and Conv remain FP32. Compatible constant Gemm is first normalized to MatMul/Add |
| Unstructured | Mask the smallest magnitudes per eligible tensor, with stable index-order ties | Dense tensors/graph are retained; real sparse kernels are not enabled |
| Channel | Mask low-norm output-channel weight rows/columns, with bias retained | Channels are not physically removed; dense parameter/FLOP counts and tensor shapes remain unchanged |
| 2:4 | Mask two entries per group of four along Conv input channels or Gemm/MatMul reduction dimension | Ineligible layers are skipped. At most two nonzero entries remain per group; dense CPU execution has no sparse kernel support |

Mask amounts are rounded down per tensor/channel count; tensors too small for a
nonzero mask are explicitly skipped. A transformation that adds no zeros fails.
Before/after zeros, masked counts, eligible/candidate elements, changed names,
skipped names and axes implied by the operator are recorded. Sparsity level refers
to **target weight entries**, not all model parameters. FP16's conversion scope
includes floating state buffers; its element count is not a parameter count.

INT8 calibration selects one training image per Imagenette class by default,
from the already extracted archive. Its split, seed, preprocessing, labels and
image hashes are embedded; paths and content digests must be disjoint from
evaluation. FP32 preprocessing and INT4 Gemm normalization are separately checked
against the original ONNX outputs before lossy conversion (`rtol=1e-4`, `atol=5e-5`,
identical top-1). The tested ORT preprocessor takes a file path: passing its
`ModelProto` overload exposed an external-initializer error in this installed stack.

## Kernel and result provenance

ONNX preparation probes the exact derived file with CPU execution and preserves
the raw profile. Observed integer/packed-weight operator families, provider,
artifact digest, profile digest, host identity and ORT version constitute the
evidence. Counts are **execution events across verification inputs**, not unique
layers. Merely loading QDQ or counting graph nodes does not establish execution.
Raw-profile counts are verified against metadata before benchmarking/storage.

Schema-4 records require typed compression details and matching artifact lineage,
calibration identity, source checkpoint, device/runtime and kernel evidence.
Ordinary/sustained schemas 1–3 remain readable without rewriting. Root pretrained
weights retain their identity; a derived checkpoint has a distinct
`compressed_checkpoint` role and hash. Original `checkpoint` hashes still must
match the root checkpoint identity. Derived artifacts use exclusive directories.

Each compressed run includes `compression.json`. Quantized runs additionally
copy the original hashed preparation profile to
`traces/compression_kernel_profile.json` before completion. This profile is
preparation evidence, separate from the benchmark's model/end-to-end/power traces.
Known private lineage, calibration and profile paths cannot enter public results.
Result storage remains immutable through the API, not protection from manual
filesystem edits. Prepared asset identities are verified before measurement.

## Observed validation limitations

The initial twenty-image check gave 70% top-1 for the FP32 baselines, FP16 and
weight-only INT4; INT8 gave 5%, unstructured pruning 40%, and channel/2:4 pruning
0%. These are observations for this fixed small fixture and recipe, not general
accuracy estimates or recommended compression settings. FP16 was substantially
slower on this CPU despite a smaller artifact.

The FP32 preprocessed graph reproduced all original outputs exactly in the
diagnostic check. INT8's poor fixture quality also remained with graph optimization
disabled, so it was not explained solely by optimized integer execution. The
authorized [INT8 follow-up diagnosis](int8_diagnosis.md) subsequently isolated
activation QDQ as sufficient to reproduce the collapse and demonstrated loss of
small channel signals at the first depthwise input. Classifier INT8 weight rounding
and full-network weight-only controls retained quality on this fixture. No
calibration search, layer exclusion sweep or fine-tuning was performed. A separate
recovery task is needed to select an accuracy/performance target on a representative
subset and validate a revised deployment recipe.

TensorRT compatibility, native FP16/INT8/INT4 execution, 2:4 eligibility and sparse
kernel selection, shared-memory behavior and measured device energy still need
the actual Jetson. This tooling implements mask-based channel pruning, not general
network graph surgery. Core imports and CPU mocks require no hardware libraries.

## Verify

```sh
python -m pytest tests -q
python -m ruff check src tests
python -m pytest tests -q --reference-prepared artifacts/reference/9029ef52961e20ad/prepared.json --compression-prepared artifacts/compression/<id>/prepared.json
```

Default tests use generated offline graphs/checkpoints/images. Optional fixture
tests skip without CPU extras. The opt-in command checks the original four-run
reference, its sustained version, and the eight-run compression matrix; it does
not prepare or download implicitly.
