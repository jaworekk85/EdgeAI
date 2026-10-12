# Edge AI Research Framework — Agent Development Spec

## 0. How to use this document

This document defines **how** the repository is built and **what kind of research it must enable**.
It does **not** define specific papers or experiments to run.

- Read this file before any substantial change.
- Section 3 (research directions) is **context for design decisions only**. Do not implement experiments for these directions unless explicitly asked.
- Work in phases (section 13). **Stop after each phase**, update `STATUS.md`, and wait for review.
- When this spec is ambiguous, prefer the simpler design and record the open question in `STATUS.md`.

---

## 1. Purpose and identity

A general research framework for studying **resource-efficient AI on embedded hardware**:
how much compute, memory and energy a model really needs on a real device, and how an embedded system can adapt its compute budget to changing inputs and conditions.

First platform: NVIDIA Jetson (Orin Nano Super). Later: other Jetson modules, other embedded accelerators, robotics/UAV platforms.

Guiding chain:

```
model compression → real hardware efficiency → adaptive compute → robust edge autonomy
```

The repository is a **measurement and experimentation framework**, not an application and not a demo.

---

## 2. Scope and safety boundaries (hard constraints)

The framework targets broadly applicable, benign workloads:
perception, inspection, mapping, obstacle perception, model monitoring, reliability, adaptive inference, hardware benchmarking.

Do **not** implement, optimize, or add support for:
- target selection or tracking of persons for targeting,
- weapon guidance or attack optimization,
- harmful autonomous decision pipelines.

Code, configs and docs must remain general and publication-friendly.

---

## 3. Research directions (context only)

The abstractions must make the following lines of inquiry easy to set up. They are open questions, not tasks.

1. **Compression vs real efficiency.** When do nominal metrics (parameters, FLOPs, sparsity) fail to predict real latency, memory, power and energy on embedded hardware?
2. **Adaptive compute.** Can a system choose the cheapest sufficient model / precision / input resolution per input or per operating condition — including the real cost of switching?
3. **Operation under changing conditions.** How do compressed models behave under illumination, weather, blur, noise, scene complexity, viewpoint, and thermal/resource pressure? When does the current model become insufficient?
4. **Workload generality.** Vision first, but the framework must not preclude small language or vision-language models running on-device (e.g. validating pruning/specialization methods from other projects on real hardware).

---

## 4. Design principles

1. **Hardware-agnostic core.** No Jetson-specific code outside dedicated backend modules. Never hard-code a single device model.
2. **Strict separation** of training, export, deployment/runtime, profiling, power measurement, and adaptive control.
3. **Runs without the target hardware.** Every component has a CPU or mock implementation so the full pipeline and tests run on a laptop.
4. **Optional heavy dependencies.** TensorRT, Jetson tooling, CUDA, ROS2 are imported lazily; absence must not break import of the core package.
5. **Configs over code.** Experiments are defined by config files; code changes should not be required to run a new combination of model × precision × device × workload.
6. **Measured ≠ estimated.** Every quantity in results carries its provenance (section 9).
7. **Small and boring beats large and clever.** No plugin registries, factories or class hierarchies until at least two concrete implementations need them.

---

## 5. Software stack

- Python, PyTorch.
- ONNX + ONNX Runtime as a portable export/runtime path.
- TensorRT as an optional backend (FP16, INT8, structured 2:4 sparsity where supported).
- OpenCV for video/image pipelines.
- ROS2: not now; keep interfaces compatible with a later adapter.
- Configs: YAML + typed dataclasses (or pydantic). Do not introduce Hydra or similar frameworks unless asked.

---

## 6. Core abstractions

Keep each as a small, serializable, typed object.

### 6.1 `ModelVariant`
```
id
architecture / family
task
weights source (checkpoint id, hash)
precision          (fp32 | fp16 | int8 | int4 | ...)
compression        (see 6.6)
input size / resolution
runtime            (pytorch | onnxruntime | tensorrt | ...)
```

### 6.2 `DeviceProfile` (captured automatically, never typed by hand)
```
device name / module
power mode (e.g. nvpmodel), clock settings, fan mode
memory
OS, JetPack / L4T, CUDA, cuDNN, TensorRT, driver versions
python + package versions
```

### 6.3 `Workload`
```
task
source            (dataset | video file | camera | synthetic)
subset / split
frame rate, resolution
condition / perturbation (see 6.7)
batch size
latency constraint (optional)
```

### 6.4 `RuntimeBackend`
Uniform interface: `load(variant)`, `warmup(n)`, `infer(batch)`, `release()`, plus memory reporting.
Implementations: PyTorch eager, ONNX Runtime, TensorRT (optional), Mock.

### 6.5 `PowerSource`
Uniform interface: `start()`, `stop() -> PowerTrace`.
Implementations:
- `JetsonOnboard` (tegrastats / jtop; rail-level, low sampling rate),
- `ExternalMeter` (pluggable adapter for a later external power meter),
- `Mock` (synthetic trace for tests).

Each source declares: `measured: bool`, `sampling_hz`, `rails`, `resolution`.

### 6.6 `CompressionSpec`
```
method               (none | quantization | structured pruning | unstructured pruning | distillation | smaller architecture)
precision / calibration info
sparsity_level
sparsity_pattern     (unstructured | 2:4 | channel | block | ...)
kernel_support       (whether the runtime actually executes sparse/low-precision kernels — detected or declared)
parameters, FLOPs    (nominal, computed by a single shared utility)
```

### 6.7 `Condition` / perturbation
Composable, seeded transforms describing operating conditions: illumination, blur, noise, weather-like effects, resolution changes, and later thermal/resource pressure.
Must support both static (per-dataset) and time-varying (streaming schedule) application.

### 6.8 `Controller`
Interface: `select(context) -> action`, `update(feedback)`.
- `context`: input difficulty proxies, confidence, resource budget, temperature, energy state, previous action.
- `action`: choice among model variants, precision, input resolution.
- Baselines to provide: static (always one variant), oracle (offline, for analysis), threshold on confidence/entropy.
- The controller runtime must account for **switching cost**: model residency (memory), load/context-switch time and energy.
- Do not start with RL.

### 6.9 `RunRecord`
The single result object written by every run (section 9).

---

## 7. Metrics

Quality: task metric `Q(M, x)` (accuracy, mAP, mIoU, …) via one evaluation module.

Latency: mean, median, p95, p99; warm-up excluded; raw per-inference timings saved.

Throughput: FPS (end-to-end and model-only, labeled separately).

Power: trace `P(t)` per rail when available.

Energy: `E = ∫ P(t) dt` over a defined window.
With low-rate onboard sensors, per-inference energy is computed as **mean power × duration over a long steady-state window ÷ number of inferences**, and labeled with that method. Idle/baseline power is recorded separately so both total and above-idle energy can be reported.

Memory: model file size, peak RAM/VRAM (shared memory on Jetson), runtime allocations.

Thermal: temperature trace, detected throttling events, sustained-run degradation (latency/power drift over time).

---

## 8. Profiling protocol

Every hardware run records: device profile, power mode, clocks, warm-up procedure, run duration, batch size, resolution, precision, runtime backend, number of repetitions, temperature at start/end.

Provide:
- a standard warm-up + steady-state measurement routine,
- a "sustained run" mode (minutes, not seconds) for thermal behavior,
- an idle-power measurement routine,
- `docs/profiling_protocol.md` describing all of the above.

---

## 9. Result storage and invariants enforced in code

Methodological rules must be **enforced by code and tests**, not only stated.

1. **Immutable results.** Each run writes to a new directory `results/<experiment>/<timestamp>_<short-hash>/`. Writing into an existing run directory raises an error.
2. **Provenance in every record:** git commit + dirty flag, full resolved config, seed, device profile, runtime versions, model variant, compression spec, workload, raw timing traces, raw power traces when available.
3. **Measured vs estimated energy.** The schema allows `energy_j` only if the power source has `measured=True`, together with `sampling_hz` and `integration_method`. Anything else goes to `energy_estimate_j` with a mandatory `estimate_method`. Validation fails otherwise.
4. **Comparable settings.** Comparison/aggregation utilities refuse to place runs side by side if resolution, power mode, batch size, or runtime differ — unless called with an explicit `allow_mismatch=True`, in which case the mismatch is printed in the output table/plot.
5. **Sparsity honesty.** Reports of pruned models always show `sparsity_pattern` and `kernel_support`. A pruned model without kernel support is labeled as such.
6. **Nominal vs real.** Parameters/FLOPs are always reported next to measured latency/energy, never instead of them.
7. **Runtime vs algorithm.** A run record states which component changed relative to its baseline (model, compression, runtime, device settings), so runtime changes are not mistaken for algorithmic gains.
8. **Proprietary separation.** Results derived from private data are written under the private tree (section 12), never under public `results/`.

Standard outputs per experiment: `config.yaml`, `device.json`, `metrics.json`, `traces/`, `plots/`.
Standard reporting utilities: a table `variant | quality | latency | memory | power | energy`, and quality-vs-energy / quality-vs-latency Pareto plots.

---

## 10. Repository structure

```
src/edge_ai/
  models/          # model families, variant definitions
  compression/     # quantization, pruning, metadata, nominal FLOPs/params
  export/          # ONNX / TensorRT export
  runtimes/        # RuntimeBackend implementations
  profiling/       # timing, warm-up, sustained runs, memory
  power/           # PowerSource implementations
  devices/         # DeviceProfile capture
  datasets/        # workload sources
  conditions/      # perturbations and schedules
  controllers/     # adaptive compute controllers
  evaluation/      # quality metrics
  records/         # RunRecord schema, validation, result store
  reporting/       # tables, Pareto plots
  utils/

configs/
  devices/  models/  workloads/  compression/  conditions/  experiments/

experiments/
  reference/       # pipeline-validation experiment (section 14)
  # new experiment folders are added by humans as research questions emerge

scripts/
  on_device/       # entry points meant to run on the Jetson

docs/
  research_framework.md
  hardware_setup.md
  profiling_protocol.md
  scope_and_boundaries.md

tests/
results/           # public results (immutable)
private/           # git-ignored: partner data, configs, adapters, results
STATUS.md
```

Do not create experiment folders named after collaborators or specific papers.

---

## 11. Testing

- All tests run on a laptop CPU without Jetson, TensorRT or CUDA (use Mock backends / Mock power).
- Hardware-dependent tests are marked (e.g. `@pytest.mark.jetson`) and skipped elsewhere.
- Required tests: record schema validation (including invariant 3), result-store immutability, comparison mismatch refusal, energy integration on synthetic traces, warm-up exclusion, deterministic perturbations given a seed, controller interface with mock variants.
- A full smoke run of the reference experiment on CPU must finish in under ~2 minutes.

---

## 12. Proprietary / partner integration

- Everything partner-specific lives in `private/` (git-ignored) or in a separate private package loaded through a documented entry point.
- No partner names, platform details, power budgets or data paths in public code, configs or docs.
- Public code must remain fully runnable and testable without the private tree.
- Core scientific results should be reproducible on public data.

---

## 13. Development phases (stop and report after each)

**Phase 0 — architecture proposal.**
Write `docs/research_framework.md` with the proposed module layout, interfaces and dependency plan. No implementation beyond skeleton. Stop for review.

**Phase 1 — core.**
Configs, dataclasses, `RunRecord` schema + validation, result store, provenance capture, Mock backends, test suite.

**Phase 2 — runtimes and profiling on a desktop/laptop.**
PyTorch and ONNX Runtime backends, timing, memory, reporting utilities. Reference experiment runs end-to-end on CPU.

**Phase 3 — Jetson support.**
Device profile capture, onboard power source, thermal traces, sustained-run mode, TensorRT backend, `scripts/on_device/`, `docs/hardware_setup.md`.

Scheduling amendment approved by the user on 2026-10-12: split Phase 3 into
**3A — hardware-independent preparation** (offline telemetry parsing/replay,
sustained CPU profiling and hardware validation plan) and **3B — on-device
implementation and validation** (automatic Jetson observations, live power/thermal
capture, TensorRT and on-device commands). Stop for review after 3A. Phase 3B may
remain pending while CPU/mock portions of Phases 4, 5 and 6 are developed and
reviewed separately. Preserve each phase's pending hardware requirements; do not
mark full Jetson support or hardware efficiency verified before 3B. CPU timings,
synthetic conditions and replay estimates do not predict Jetson performance.

**Phase 4 — compression tooling.**
FP16/INT8 (and INT4 where the runtime supports it), structured and unstructured pruning including 2:4, compression metadata and kernel-support detection.

**Phase 5 — adaptive compute.**
Controller interface, static/oracle/threshold baselines, switching-cost accounting, streaming workloads.

**Phase 6 — changing conditions.**
Perturbation library, time-varying schedules, hooks for model-adequacy signals (confidence, small/large disagreement, input/activation statistics).

---

## 14. Reference experiment (pipeline validation, not research)

Purpose: prove the whole pipeline works end-to-end. Its numbers are not a research result.

```
1 public pretrained model family (2–3 sizes and/or precisions)
1 small public dataset subset or public video
metrics: quality, latency, memory, power, energy
output: standard table + Pareto plot
```

Runs on CPU with Mock power on a laptop, and with real power measurement on the Jetson using the same config (only the device config changes).

---

## 15. Agent working rules

1. Read this spec before substantial changes.
2. Propose architecture before implementing it; stop at phase boundaries.
3. Update `STATUS.md` after each work session: done, open questions, known limitations.
4. Preserve hardware-agnostic abstractions and laptop-runnable tests.
5. Never claim or label energy savings without measured energy.
6. Never overwrite prior measurements.
7. Do not add experiments for the research directions unless asked.
8. Avoid long or expensive runs unless asked.
9. Keep safety boundaries (section 2) and proprietary isolation (section 12).
10. Prefer small, readable, tested modules over general frameworks.
