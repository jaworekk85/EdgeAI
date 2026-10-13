# Phase 4: compression tooling on CPU

Status: user accepted 3A and authorized Phase 4 on 2026-10-13. Architecture
recorded before implementation; CPU tools implemented and verified, awaiting
user review. Stop before Phase 5; 3B remains pending.

## Scope and methods

Implement reusable, lazy compression transforms and a bounded reference
validation using existing MobileNetV3 Small assets. No training, research sweep,
controller or condition experiment. Each transform creates a new exclusive
artifact directory; preserve originals and incomplete work on failure.

| Method | CPU path | Honest limitation |
| --- | --- | --- |
| FP16 | Save a derived FP16 state dict, load a half-precision PyTorch model and cast prepared input | Tensor dtype/storage is verified; internal CPU arithmetic/kernel acceleration is unknown |
| INT8 | ONNX Runtime static QDQ, per-channel weights, MinMax calibration on a disjoint train subset | Conv/Gemm coverage is partial; record transformed graph and actual CPU kernel profile |
| INT4 | ORT blockwise weight-only MatMulNBits; normalize compatible constant-weight Gemm to MatMul/Add | Classifier weights only, activations and convolution stay FP32; not a fully INT4 network |
| Unstructured pruning | Deterministic magnitude masks on ONNX Conv/Gemm weight initializers | Dense storage/execution; nominal dimensions/FLOPs unchanged |
| Channel pruning | Deterministic output-channel weight-row masks | Masks do not physically remove channels; bias is retained; no smaller dense graph claimed |
| 2:4 pruning | Keep two magnitude-ranked entries per four along Conv input channels / Gemm reduction axis | Ineligible layers are unchanged; dense CPU execution, TensorRT kernel eligibility must be rechecked in 3B |

Use existing pinned libraries. Follow official
[ORT quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
and [PyTorch pruning semantics](https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html).
FP16 uses PyTorch rather than assuming CPU ORT FP16 support from
[float16 documentation](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html).
The installed ORT 1.23.2 has `MatMulNBitsQuantizer`; verify its actual API locally.
Fail unsupported/no-op transformations explicitly rather than silently falling back.

## Architecture

- `compression/recipe.py`: small typed recipes, coverage/lineage and kernel evidence.
- `compression/pruning.py`: pure array masks and ONNX weight targeting; stable tie
  handling, explicit axes/eligible weights, before/after zeros and unchanged shape.
- `compression/quantization.py`: exclusive FP16 checkpoints, calibrated QDQ INT8,
  compatible Gemm normalization and INT4 export with recorded settings/versions.
- `compression/verification.py`: graph checks and finite-output probe, ORT CPU
  profiling of the exact derived artifact, hashes and device/runtime evidence.
- `compression/reference.py`: prepare/run commands using verified existing public
  assets, typed YAML matrix, fresh worker per measured configuration, reference
  baselines within each runtime and immutable standard reports.
- Extend artifact roles with `compressed_checkpoint`; original `checkpoint`
  hash-equality invariant remains. Derived artifacts retain canonical source
  checkpoint identity separately from their own content digest.
- Extend compression metadata with typed details. Schema 4 requires complete
  compression/artifact/calibration/kernel evidence for transformed real runs;
  schemas 1–3 remain readable. Keep raw preparation profiles distinct from
  benchmark timing/energy. Private routing includes lineage/calibration paths.

## Provenance and validation

Calibration uses one training image per Imagenette class, seeded and checksummed,
from the already extracted public archive. Evaluation reuses the original twenty
validation images. Enforce disjoint paths and image digests; record calibration
preprocessing, split, sample hashes, selection seed and MinMax settings.

Record parent artifact, recipe, versions, transformed tensor names, eligibility
scope, actual masked/zero counts and observed operator profile. A successful
session load or QDQ graph alone does not prove quantized kernels ran. INT8/INT4
support is scoped to quantized CPU operator families actually observed, with a
hashed raw profile and matching artifact/device/runtime. FP16 internal kernel
precision remains unknown; dense pruned artifacts declare no sparse support.

Preserve nominal dense parameters/FLOPs next to actual file size, RSS, timings,
quality and estimated energy. Root pretrained weights identity stays fixed, so
changing only compression is attributed to compression. Calibration and export
costs are preparation costs, outside benchmark windows. Quantization/pruning may
reduce quality or increase latency; report the observations without guarantees.

## Acceptance and deferred work

- Core/mock imports and tests remain runnable without optional libraries.
- Independent array fixtures verify exact masks/axes/ties, counts, no mutation,
  channel-zero rows, 2:4 validity, and refusal of ineligible/no-op requests.
- Generated CPU models exercise real FP16, INT8 and INT4 transforms/inference;
  fail corrupt identities, overlapping calibration, missing/mismatched evidence,
  invalid schema/counts, incomplete artifacts and overwrite attempts.
- Run a prepared public compression matrix end-to-end (no downloads), verify all
  complete records/reports, plus existing ordinary/sustained CPU regression smoke.
- Check lint/dependencies/docs/whitespace, update status, commit/push and stop for
  review before Phase 5. Keep calibrated reference runs short.

Pending 3B: TensorRT engine/export compatibility, actual FP16/INT8/INT4 operator
support, sparse kernel eligibility/execution, device energy and real efficiency.
General automatic channel graph surgery, retraining and arbitrary model/operator
coverage are not claimed by the initial CPU tooling.
