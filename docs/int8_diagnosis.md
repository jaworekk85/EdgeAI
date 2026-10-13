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

Keep the existing INT8 recipe/results identifiable; they are currently unsuitable
as a quality-preserving option for the future controller. FP32/FP16/INT4 retain
their existing small-fixture observations. Phase 5 remains pending review.

A separate recovery task can test explicit mixed precision around sensitive
activations, calibration strategies that address range imbalance, or later QAT.
Choose a representative evaluation subset and a quality budget before claiming a
successful deployment recipe. This diagnosis does not select or tune such a recipe.
