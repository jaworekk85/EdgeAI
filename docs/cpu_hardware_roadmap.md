# CPU development while Jetson validation is pending

The user approved splitting Phase 3 on 2026-10-12 because the device is expected
in about three months. Phase 3B is a separate hardware gate, not a prerequisite
for developing the CPU/mock portions of Phases 4–6. Each phase still needs its
own architecture proposal, user review and status update.

| Stage | Work possible without Jetson | Pending device work |
| --- | --- | --- |
| 3A | Offline telemetry parser/replay, sustained CPU profiling, honest records/reports and hardware acceptance plan | No Jetson execution claimed |
| 4 | Compression transforms/metadata, calibration provenance, quality checks, nominal counts and real CPU benchmarking where supported | TensorRT FP16/INT8/INT4 and sparsity execution/support must be checked for the actual device/runtime; no speedup inferred from sparsity alone |
| 5 | Static/oracle/threshold controllers, CPU streaming, CPU model residency/load/switch measurements, explicitly assumed resource/cost scenarios | Actual Jetson switching time/energy, shared-memory behavior and device budgets |
| 6 | Seeded image perturbations, time-varying schedules, confidence/disagreement hooks and synthetic resource-pressure schedules | Device thermal/power observations and validation of adaptations under actual hardware pressure |
| 3B | Perform after the device is available; bring the previously reviewed CPU features to hardware | Jetson profile, live telemetry, TensorRT, sustained thermal/power measurements, on-device entry points and repeatability checks |

The recommended order is **3A → review → 4 (CPU) → review → 5 (CPU) → review →
6 (CPU) → review**, with **3B when hardware is available**. The device can arrive
at any point; 3B does not need to wait for Phase 6. On 2026-10-13 the user accepted
3A and authorized [Phase 4 CPU work](phase_4_usage.md). Phases 5–6 and research
experiments still require separate authorization; the schedule does not authorize them.

CPU timings describe the CPU host. Image quality tests can be reused but must be
rechecked after backend/precision/export changes. Simulated temperature, memory
budgets or costs belong to explicit scenario inputs, not the automatically
observed `DeviceProfile`. Historical telemetry remains separate from new runs.
Synthetic/replayed watts produce estimated energy only. A credible Jetson
performance/thermal model would need calibration data; none is available yet.

Completion means separately recording **CPU implementation accepted** and
**Jetson validation pending/passed** for each relevant stage. Hardware efficiency,
TensorRT kernel execution and full Phase 3 completion require the real device.
