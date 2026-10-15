# Current status

Updated 2026-10-15. **Phase 4 CPU tooling and INT8 diagnosis are implemented.
Maintenance is complete and verified; Phase 5 and INT8 recovery await review.**
[Documentation index](docs/README.md) · [Code map](docs/research_framework.md) ·
[Decisions/history](docs/history.md).

## Implemented

- Typed configs/metadata, provenance, immutable/private results and guarded reports.
- PyTorch/ONNX Runtime CPU inference, real timing, sampled RSS and classification.
- Sustained CPU profiling and offline tegrastats parsing/replay.
- FP16, calibrated INT8, classifier weight-only INT4 and dense pruning masks.
- Bounded INT8 diagnostics with retained identities, controls and split statistics.

## Current maintenance

Authorized: simplify code and Markdown before further phase work. The plan was
recorded before implementation and is summarized in history.

- Both CPU paths share experiment construction, isolated workers and baseline
  attribution. Benchmarking and diagnostics share prepared-compression validation.
- Removed repeated orchestration checks; simplified diagnostic aggregation while
  retaining its controls, output meaning, lazy imports and exclusive writes.
- Replaced the long session log and completed standalone plans with concise history
  and fixed Git links to the originals. Added one documentation index and code map.
- Full regression with ordinary/sustained reference and compression smoke:
  **191 passed, 2 Windows symlink skips**, 113.73 s. Focused checks also passed.
- Fresh diagnostic predictions/settings match the previous evidence exactly;
  all tensor/channel statistics agree within `1e-12`. Old/new hashes verified.
- Corrupt-baseline probe rejected before workers; old schema-2/3 runs remain
  readable. Ruff, local links, historical Git targets and whitespace checks passed.

## Known limits

- Existing INT8 gives 1/20 correct versus FP32/FP16/INT4 14/20 on the fixed fixture.
  Activation QDQ alone reproduces the collapse; coarse shared steps erase small
  channels. [Diagnosis](docs/int8_diagnosis.md) is complete; no corrected recipe yet.
- Pruning fixture quality also fell: unstructured 8/20, channel/2:4 0/20. These
  twenty-image observations are pipeline checks, not general accuracy estimates.
- FP16 is slower on this CPU; internal arithmetic is unverified. INT4 covers
  classifier weights; pruning masks retain dense shapes/storage and nominal FLOPs.
- All CPU power/replay energy is estimated. No Jetson performance, TensorRT
  compatibility, sparse acceleration or measured device energy is established.
- Controllers/streaming and perturbation transforms remain unimplemented.
  Distribution license is still undecided.

## Next review and deferred device work

Review maintenance, then choose a separate INT8 recovery task (quality budget and
mixed-precision/calibration approach) or propose Phase 5 controller architecture.
Neither is authorized by this cleanup.

Phase 3B awaits hardware: observe the actual JetPack/L4T stack, review its design
and execute the [hardware acceptance gate](docs/hardware_setup.md).
CPU parts of 5–6 may proceed through their own reviews while 3B remains pending.
