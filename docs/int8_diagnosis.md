# Phase 4 INT8 diagnosis (2026-10-14)

**Finding: activation quantization is sufficient to cause the collapse in this
prepared MobileNetV3 Small fixture. Weight rounding alone does not reproduce it.**
The first depthwise input exposes a concrete loss of small channel signals under
the shared per-tensor activation scale. This is a numerical sensitivity of the
current recipe; no graph-export or preprocessing defect was found.

## Fixed controls

All controls use the same original weights and twenty verified Imagenette validation
images, with ImageNet-1K output labels. Calibration remains the original disjoint
ten training images. No data were downloaded, no scales were refitted and no
training, calibration search or layer exclusion sweep was performed.

| Graph/control | Correct / 20 | Top-1 |
| --- | ---: | ---: |
| Original FP32 ONNX | 14 | 70% |
| Existing classifier-only INT4 | 14 | 70% |
| INT8 rounding of the same classifier weights; FP32 execution | 14 | 70% |
| Existing full Conv/Gemm INT8 QDQ recipe | 1 | 5% |
| Keep its quantized weight/bias constants; remove activation QDQ | 15 | 75% |
| Restore original FP32 weights/bias; keep activation QDQ | 0 | 0% |

The classifier control rounds exactly the two normalized classifier MatMul weight
matrices (1,613,824 entries) in blocks of 32 along K, per column, with symmetric
nearest-even INT8 rounding and absmax/127 scales. The restored constants execute
in FP32: this isolates numerical error, **not INT8 kernel speed/support**. INT4
and this control have equal tensor scope, not identical quantization algorithms.

The last two controls edit the existing QDQ graph without fitting anything. They
retain respectively its exact weight/bias quantization or its exact activation
quantization. All are diagnostic graphs, not deployment recipes. The single extra
correct answer in the weights-only control is not evidence of improved accuracy.
These twenty-image observations are pipeline checks, not population estimates.

## Where the error starts

Streaming [ORT QDQ activation matching](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#quantization-debugging)
matched 103 activation tensors on each split. Mean relative L2 error is calculated
per image, then averaged; local QDQ error compares before/after rounding in the
same graph, while accumulated error compares the QDQ result against FP32.

On validation images, accumulated relative L2 error is about 0.039 after the stem
Conv and 0.040 after its HardSwish, then reaches **0.449** at the first depthwise
Conv (`node_Conv_439`, output `getitem_3`). Depthwise layers process channels
separately, so small channel errors can be hidden by a good global input error.

An isolated control feeds **original FP32 HardSwish activations** to this Conv,
with **identical original FP32 weights/biases** in both graphs. Only one graph
rounds its input using the existing activation QDQ parameters. This alone produces
mean relative L2 output error **0.290** on validation and **0.290** on calibration.
It demonstrates a substantial early error independently of accumulated upstream
errors, weight rounding, and integer Conv kernels.

The input uses one shared activation scale **0.1204849482**, zero point -125.
Channels have very different signal magnitudes. For channel 7 on validation:

- Mean input RMS: **0.03150**, much smaller than the quantization step.
- Mean fraction newly rounded to zero: **93.68%** (original zero fraction 0%).
- Mean relative L2 input error: **0.930**.
- Mean relative L2 depthwise output error in that channel: **0.421**.

Channel 6 has isolated output error about 0.834 and channel 12 about 0.715.
Similar behavior on calibration images shows that validation-only clipping cannot
alone explain the problem. More calibration images are not an established fix:
MinMax can still assign a coarse shared step when channel ranges differ widely.

All 108 matched weight/bias constants were inspected with subtraction promoted to
float64 to prevent integer overflow in diagnostics. The largest relative L2 weight
error was about 0.0145; the first depthwise weights' error was about 0.00409.
The full graph controls, rather than weight error magnitude alone, establish the
activation contribution. Previously verified FP32 preprocessing parity and the
poor INT8 result without graph optimizations remain consistent with this finding.

## Reproduction and retained evidence

From the repository root, with the existing CPU/ONNX extras and prepared assets:

```powershell
.venv\Scripts\python.exe scripts/diagnose_int8.py --prepared artifacts/compression/20261013T132850452730Z_1e0116cc38/prepared.json
```

Each invocation reserves a fresh ignored directory. The final verified evidence
for this session is under:

```text
artifacts/compression_diagnostics/20261013T221657564220Z_33004012bd/
```

It contains input artifact identities, manifests, host/Git/runtime provenance and
script SHA-256, source graph hashes/copies, graph controls, per-image predictions,
weight/bias errors, split-specific activation statistics, isolated channel
statistics and output hashes. `complete.json` is written last. ORT's augmentation
and temporary shape-inference files operate only on fresh source copies; previous
source files and result records remain intact. Runs from intermediate diagnostic
versions remain separate. The bounded CLI refuses more than 64 images per split.

Augmented graphs use CPU sessions with graph optimization disabled and one thread
for matching; quality controls use ordinary optimized CPU sessions with one thread.
No latency, measured energy, sparse support or Jetson performance is inferred.
Signed-subtraction, graph-isolation and classifier-rounding checks use independent
generated fixtures; the actual public diagnostic provides the integration check.

## Consequence for the next phase

Keep the original broad INT8 recipe/results identifiable. The authorized follow-up
below evaluates a conservative mixed recipe on a larger independent subset;
full-network INT8 recovery and training remain separate work. Phase 5 follows review.

## Authorized recovery plan — 2026-10-15

The user authorized the proposed recovery-first sequence. Before implementation:

- Add one explicit INT8 recipe scope, `linear`: quantize constant Gemm/MatMul
  weights and their activation boundaries, retain Conv and the feature extractor
  in FP32. Keep the existing `all` scope as the default and a comparison control.
  This follows the observed activation sensitivity; it is partial mixed precision,
  not full-network INT8. No layer search or training.
- Use 100 seeded training images for MinMax calibration and 200 stratified
  validation images. Exclude the previous 20 evaluation images by path AND digest;
  preserve that exclusion manifest in the new preparation/cache identity.
- Predeclare a pilot quality limit: at most **2 percentage points** below the
  matching FP32 baseline (at most four net extra errors on 200 images). Compare
  the fixed recipes once; do not tune against this evaluation set.
- Reuse the existing preparation/measurement/reporting tools and standard schema-4
  records, observed integer kernels, actual coverage and estimated-only energy.
  Old recipes, preparations and result bytes remain valid and unchanged.
- Extend typed metadata conservatively. Legacy missing scope means `all`;
  compare decoded compression metadata rather than rewriting historical JSON.
  Add manifest exclusion to reference preparation, without changing default runs.
- Verify numerical/invariant tests, legacy public smoke and the new bounded
  matrix. Record measured CPU cost and gate outcome; quality retention does not
  establish a speedup, population accuracy or Jetson compatibility. Stop for review
  before Phase 5.

Method reference: [ORT quantization debugging](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#quantization-debugging)
describes retaining sensitive computations in floating point.

Larger-subset preparation exposed a FP32 export tolerance failure (3 of 1000
logits on one Small input, violating absolute difference up to 1.08e-4). A paired
check of the existing canonical exports on all 200 new inputs retained **200/200
top-1 parity per model**; maximum absolute differences were Small 1.20e-4 and
Large 2.29e-5. Retain the failed preparation. The quality config explicitly uses
`export_atol=2e-4` with this numerical reason; the default stays 5e-5, rtol stays
1e-4, and exact top-1 equality remains mandatory. This concerns FP32 cross-runtime
export only, not quantized quality or the existing ONNX-to-ONNX parity tolerance.

## Recovery outcome — 2026-10-15

**The mixed `linear` recipe passed the predeclared pilot gate:** 132/200 correct
versus FP32 133/200, a **0.5 percentage-point drop**, below the 2-point limit.
Both fixed matrices gave the same quality; the second ran after regression tests
finished. Repetition does not increase the independent evaluation sample.

| Variant (ORT CPU) | Correct / 200 | Top-1 | File MB (decimal) | Mean / p95 model latency, ms |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 133 | 66.5% | 10.46 | 2.75 / 3.06 |
| INT8 `all`, 100 calibration images | 7 | 3.5% | 2.82 | 6.80 / 9.50 |
| INT8 `linear`, 100 calibration images | 132 | 66.0% | 5.37 | 2.49 / 2.78 |

The mixed file is about 49% smaller. It targets **1,613,824 / 2,525,832 eligible
weight entries**. All 52 Conv weights remain FP32; classifier weights and their
QDQ activation boundaries change. Observed evidence was **200 QGemm execution
events across 200 inputs**, with no quantized Conv events. This is partial kernel
evidence, not proof that both classifier layers always execute integer arithmetic.
The broader recipe still collapses despite the larger calibration.

Latency is descriptive for this CPU/session, not an established speedup or an
energy-saving claim. The sample remains Imagenette, not a deployment quality
guarantee. All energy is estimated. No pruning recovery, retraining or Jetson
validation was performed; original preparations/results were not rewritten.

### Reproduce

After the [CPU setup](cpu_reference.md), from the repository root:

```sh
python -m edge_ai.reference prepare --config configs/experiments/reference_quality.yaml --exclude-manifest artifacts/reference/9029ef52961e20ad/manifest.json
python -m edge_ai.compression.reference prepare --config configs/compression/int8_quality.yaml --reference-config configs/experiments/reference_quality.yaml --reference-prepared artifacts/reference/ec224752df73e653/prepared.json
python -m edge_ai.compression.reference run --config configs/compression/int8_quality.yaml --reference-config configs/experiments/reference_quality.yaml --prepared artifacts/compression/20261015T121525126988Z_cda3a8f856/prepared.json
```

Use the preparation paths printed for your environment; changed versions/settings
produce different identities. Matching reference preparation was verified reusable.
Compression preparation and each measurement reserve fresh directories.
The configs fix the two scopes, 100-image train calibration and 200-image evaluation.
Exclusion is recorded in the reference preparation and its cache identity; image
paths and content hashes cannot overlap the historical twenty-image manifest.

Retained final evidence:

- Report: `results/int8_quality/20261015T121844864837Z_989264f8b3/report.md`.
- Pilot gate: `artifacts/int8_quality_checks/20261015T122004269445Z_e8074c6e6b/quality_gate.json`
  records the threshold, observed difference, run paths/hashes, coverage, dataset
  digests and Git/runtime provenance; completion was written last.
- Regression: **194 passed, 2 Windows symlink skips** in 83.90 s, including legacy
  ordinary/sustained/compression smoke. Follow-up scope guards passed independently.
  Generated checks verify unchanged Conv tensors, actual partial kernels, strict
  scopes, exclusion of renamed image duplicates and explicit tolerance reasons.
  Actual historical schema-2/3/4 records remain readable.
