# Current status

Updated 2026-10-15. **Mixed INT8 recovery passed its pilot check.
Ready for review before Phase 5 architecture/implementation.**
[Documentation](docs/README.md) · [Code map](docs/research_framework.md) ·
[History](docs/history.md) · [Polish capabilities guide](docs/mozliwosci_i_eksperymenty.md).

## Implemented

- Typed metadata/configs, provenance, immutable/private results and guarded reports.
- PyTorch/ORT CPU inference, quality, timing, RSS and sustained profiling.
- Offline tegrastats parsing/replay; power remains synthetic.
- FP16, broad INT8, mixed linear INT8, classifier weight-only INT4 and pruning masks.
- INT8 diagnosis, shared CPU orchestration and concise documentation/navigation.

## Latest verified step: INT8 recovery

The user authorized the recovery → 5 → 6 sequence. The predeclared
[recovery plan and results](docs/int8_diagnosis.md) cover this step; phase reviews
still apply.

- Added `int8_scope: linear`: Gemm/MatMul quantization with Conv FP32; default
  `all` and legacy metadata remain supported without rewriting old files.
- Calibrated on 100 train images; evaluated on 200 validation images excluding
  the historical twenty by path and digest. Exclusion participates in cache identity.
- FP32 **133/200 (66.5%)**, mixed INT8 **132/200 (66%)**, broad INT8 **7/200 (3.5%)**.
  Mixed drop **0.5 points** passes the predeclared 2-point pilot limit.
- Mixed artifact **5.37 MB versus 10.46 MB**, targeting 1,613,824 / 2,525,832 eligible
  weights; all 52 Conv weights remain FP32. Hashed evidence observed QGemm events.
- Post-regression model mean/p95: FP32 **2.75/3.06 ms**, mixed **2.49/2.78 ms**.
  Descriptive CPU observations, not a statistically established speedup.
- Full regression with legacy ordinary/sustained/compression smoke:
  **194 passed, 2 Windows symlink skips**, 83.90 s; added scope guards also passed.
- FP32 export required explicit `atol=2e-4` for this larger subset, with recorded
  numerical justification and exact top-1 parity. Default remains 5e-5.
  Failed preparation retained; matching completed preparation reused.
- Old schema-2/3/4 records remain readable. Pilot gate, provenance and run hashes
  are retained; lint, links and whitespace verified.

## Limits and next review

- This is a partial INT8 recipe and a bounded Imagenette pilot, not a deployment
  guarantee or full-network recovery. Broad INT8 still loses quality.
- Pruning quality recovery/training remains separate. Dense masks preserve shapes;
  CPU FP16 was slower, and INT4 covers classifier weights only.
- No measured energy, TensorRT compatibility, sparse acceleration or Jetson
  performance is established. Phase 3B awaits hardware and its
  [acceptance gate](docs/hardware_setup.md).
- Next: review this recovery, then propose Phase 5 controllers/streaming/switching
  costs. Phase 6 perturbations follow their own review. Neither is implemented yet.
- CPU parts of 5–6 can proceed before 3B. Distribution license remains undecided.
