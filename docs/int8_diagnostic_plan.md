# Phase 4 INT8 diagnostic follow-up

Authorized by the user on 2026-10-14 after the INT4/INT8 scope comparison.
This is bounded CPU pipeline diagnosis, before Phase 5, not a research sweep.

## Proposed approach

1. Verify existing prepared identities, immutable source/evaluation manifests,
   and the disjoint ten-image training calibration manifest.
2. Compare classifier weight-only INT4 with a numerical INT8 rounding control on
   the same normalized MatMul weights and fixed twenty evaluation images. Execute
   the restored INT8 control weights in FP32 to isolate rounding; keep Conv and
   activations FP32, without an INT8 kernel claim. Record exact settings/tensors.
3. Use ORT's QDQ matching tools on the existing FP32 preprocessed and INT8 graphs.
   Stream one image at a time to bound memory. Save per-tensor relative L2 error,
   maximum error and clipping observations, for training calibration and evaluation
   separately. Preserve graph order; distinguish local QDQ error from accumulated
   error relative to FP32.
4. If a concrete faulty transformation or sensitive tensor emerges, run a single
   targeted control to isolate it. Record that decision and its evidence before
   running the control. No calibration grid search, retraining or pruning recovery.

## Architecture and provenance

A standalone CPU diagnostic entry point in `scripts/` reuses the prepared
descriptor validators, manifests, path guards and exclusive artifact directories.
Heavy imports stay inside the diagnostic functions. Augmented graphs, predictions,
summary statistics, source hashes, dataset identities and captured host/Git/runtime
provenance go into a new ignored `artifacts/compression_diagnostics/<run-id>/`
directory. A completion marker is written last. Prior artifacts/results are never
edited. These outputs are diagnostic evidence, not profiling RunRecords; no timing,
energy or deployable low-precision kernel claim is inferred from augmented graphs.

The final report will distinguish established findings from remaining hypotheses.
Any production recipe change needs regression coverage and a fresh prepared run;
the original poor-quality recipe/results remain identifiable and immutable.

## First evidence and targeted control (2026-10-14)

The classifier INT8 rounding control retains 14/20 correct predictions, as does
INT4. Streaming QDQ matching finds the first major accumulated error at the
second Conv (`getitem_3`, first depthwise Conv): mean relative L2 error about
0.45, versus about 0.04 immediately before it. The same pattern occurs on the
training calibration images, so validation-only out-of-range clipping cannot
alone explain it.

Next bounded control: remove activation QDQ while preserving the exact quantized
weight/bias constants, then conversely restore FP32 weights/biases while preserving
activation QDQ. These two numerical controls isolate weight versus activation
contributions in the existing artifact. They are diagnostic FP32 execution graphs,
not new deployment recipes; preserve their exact edits and hashes. Inspect all
dequantized weight/bias errors with integer values promoted before subtraction.

The component controls retained 15/20 correct with only weight/bias QDQ and
0/20 with only activation QDQ. A further isolated first-depthwise control kept
original FP32 inputs/weights/biases and applied only its existing input QDQ in one
copy. It produced about 0.290 relative L2 output error on both splits. This closes
the authorized diagnostic scope; see [findings and reproduction](int8_diagnosis.md).
No production recipe or Phase 5 implementation was added.

Primary API reference: [ORT quantization debugging](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#quantization-debugging).
