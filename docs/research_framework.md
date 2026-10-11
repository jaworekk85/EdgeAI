# Proposed research architecture — Phase 0

Status: Phase 0 reviewed against the specification on 2026-10-11; the user
authorized the review and Phase 1 implementation together. The development
specification remains authoritative. The phase table distinguishes implemented
core contracts from planned later backends and experiments.

## Module responsibilities and implementation phases

| Module | Responsibility | First implementation |
| --- | --- | --- |
| `models` | Serializable `ModelVariant`; model families and weights identity | 1 (metadata), 2 (model loading) |
| `compression` | `CompressionSpec`, shared parameter/FLOP accounting; transformations | 1 (metadata), 4 (tooling) |
| `export` | Explicit ONNX / optional TensorRT artifact export | 2 / 3 |
| `runtimes` | `RuntimeBackend` contract, mock, PyTorch, ONNX Runtime, TensorRT | 1 / 2 / 3 |
| `profiling` | Warm-up, synchronized timing, memory, steady-state/sustained runs | 2 / 3 |
| `power` | `PowerSource`, `PowerTrace`, mock, integration, device-specific sources | 1 / 3 |
| `devices` | Automatic host/device profile capture; Jetson-specific capture submodule | 1 / 3 |
| `datasets` | `Workload` metadata, repeatable input sources and dataset subsets | 1 / 2 |
| `conditions` | `Condition` descriptions, seeded transforms and streaming schedules | 1 (metadata), 6 (transforms) |
| `controllers` | Selection/update contract, baselines and switching-cost accounting | 5 |
| `evaluation` | One task-quality evaluation interface, outside model-only timings | 2 |
| `records` | `RunRecord`, validation, immutable store and privacy routing | 1 |
| `reporting` | Comparable-run checks, tables and Pareto plots | 1 (checks), 2 (outputs) |
| `utils` | Small shared helpers such as provenance and deterministic seeding | 1 |

Training is separate from export and measurement. The initial scope does not add
a training subsystem. Future training tools can emit checkpoints consumed by
`models` and `export` without becoming a dependency of profiling or runtimes.

## Data objects

Use standard-library dataclasses, enums and explicit validators. Nested objects
serialize to ordinary JSON/YAML values. Configuration loading will reject unknown
keys and validate values before loading a model. Introduce no factory/registry
framework. Construct the initial mock runtime explicitly; add a small backend
dispatch only when multiple implementations exist.

- `ModelVariant`: stable ID, architecture/family, task, weights source and digest,
  precision, compression, input shape/resolution and runtime.
- `CompressionSpec`: method, calibration/precision metadata, sparsity level and
  pattern, kernel support with detection/declaration provenance, nominal parameters
  and FLOPs from one shared accounting utility. Unknown values remain explicit.
- `DeviceProfile`: automatically captured identity, memory, OS, device software,
  power mode, clocks, fan state, Python and relevant package versions. Unsupported
  observations are `null` with a reason, never guessed or manually entered facts.
- `Workload`: task, source, split/subset, rate, resolution, batch size, conditions,
  optional latency constraint and public/private classification.
- `Condition`: transform kind, parameters, seed and optional time schedule.
- `PowerTrace`: source metadata (`measured`, sampling rate, rails, resolution),
  monotonic timestamps and power samples with units and missing-sample metadata.
- `RunRecord`: schema version, run identity, provenance, resolved config, seed,
  model/compression/workload/device metadata, protocol, raw traces, quality and
  metric summaries, plus baseline identity and changed components.

Unknown quality, power or memory values use `null`, not zero. Nominal compute
metrics state input shape and counting method so FLOP conventions stay visible.

## Proposed runtime and power contracts

`RuntimeBackend` is a small typing protocol, not an inheritance hierarchy:

```text
load(variant: ModelVariant) -> None
warmup(n: int) -> None
infer(batch) -> inference output
memory_report() -> memory summary with provenance
release() -> None
```

Warm-up repeats a representative batch bound during setup. Phase 1 must make that
binding explicit in the mock contract; inference inputs/outputs remain task-aware.
Device synchronization lives in its runtime backend so timing boundaries include
completed computation. Resource release must also run after failed inference.

`PowerSource` declares `measured`, `sampling_hz`, `rails` and `resolution`:

```text
start() -> None
stop() -> PowerTrace
```

The mock returns a deterministic synthetic trace marked `measured=False`.
Jetson tooling lives in dedicated modules under `devices` and `power`, with lazy
imports. A later external meter implements the same contract with a documented
sampling clock and measurement boundary. No hardware imports in package roots.

`Controller` (Phase 5): `select(context) -> action`, `update(feedback) -> None`.
Context includes confidence/difficulty, resource/thermal/energy state and previous
action. Actions choose a declared model variant, precision and resolution. Runtime
execution records residency, loading/context-switch duration and energy separately.
Begin with static, offline oracle and threshold baselines; no RL implementation.

## Configuration and execution flow

YAML fragments live under the six directories in `configs`. A future experiment
entry point resolves selected fragments into a typed config, saves that full
config and never relies on implicit machine-specific defaults. Paths to public
config fragments are relative to the project/config root; private fragments are
loaded only from the private tree. Overrides become part of the resolved config.

```text
resolve and validate config -> capture provenance/device -> prepare workload
-> load runtime -> warm-up -> idle baseline -> timed measurement/power window
-> evaluate quality -> validate RunRecord -> write immutable outputs -> report
```

Model-only timing excludes input preparation and quality evaluation. End-to-end
timing has separate boundaries, includes pipeline overhead and uses a separate
FPS field. Raw per-inference timings and trace windows make each summary auditable.
The reference experiment validates plumbing and is never presented as research.

## Storage and scientific invariants

Reserve a fresh directory with an exclusive create operation:
`results/<experiment>/<UTC timestamp>_<short-hash>/`. Never reuse or overwrite it.
Private runs use `private/results/...`; explicit private classification is checked
before reserving any public output. Failed runs retain an explicit incomplete
status and are excluded from completed-run comparisons.

Standard outputs: `config.yaml`, `device.json`, `metrics.json`, `traces/`, `plots/`.
The validated `RunRecord` is serialized in `metrics.json`; raw traces are embedded
in this self-contained record and mirrored in relative `traces/*.json` files with
units/source metadata. Capture git commit and dirty flag,
runtime versions, model identity and protocol before measurement.

Validation will enforce:

1. `energy_j` requires a measured source, positive `sampling_hz` and an
   `integration_method`; synthetic/estimated values use `energy_estimate_j` and
   mandatory `estimate_method`. Neither field silently substitutes for the other.
2. Comparisons refuse mismatched resolution, power mode, batch size or runtime.
   `allow_mismatch=True` explicitly prints the differing settings in outputs.
3. Pruned models report sparsity pattern and kernel support, including an explicit
   label for pruning without runtime kernel support. Unknown support stays unknown.
4. Tables show nominal parameters/FLOPs alongside real latency and energy, with
   clear measured/estimated labels and unavailable fields where necessary.
5. Baseline comparisons declare changes in model, compression, runtime or device
   settings so gains are attributed to the correct component.
6. Partner-specific runs never write under public `results/`.

Energy integration uses timestamped traces over an explicit window. Low-rate
onboard sources use mean power times steady-state duration divided by inference
count for average inference energy; record idle power separately for total and
above-idle results. Do not suggest per-inference sensor resolution when absent.

## Dependency plan

| Layer | Dependency | Planned introduction |
| --- | --- | --- |
| Build | setuptools; `build` for packaging checks | 0 |
| Core | Python standard library, PyYAML for config I/O | 1 |
| Development | pytest, Ruff | 0 (configuration), 1 (tests) |
| CPU inference | Optional PyTorch, ONNX and CPU ONNX Runtime extras | 2 |
| Workloads | Optional OpenCV for image/video paths | 2 |
| Plots | Optional matplotlib reporting extra | 2 |
| Jetson | Platform-provided CUDA/cuDNN/TensorRT; optional onboard tooling | 3 |
| Compression | Backend-specific tools only when needed | 4 |

Phase 1 adds PyYAML to the core dependencies and CPU/mock tests. Defer hardware extras
and exact backend pins until compatible device software is known. Runtime versions
are captured per run; reproducible CPU and Jetson environments will be documented
separately. No Hydra, ROS2 dependency or plugin registry in the initial framework.

## Verification plan

Phase 0: validate packaging metadata, imports, syntax, whitespace, directory layout
and ignore rules. Do not add a fake experiment or placeholder passing test suite.

Phase 1 tests must cover schema validation (especially energy provenance), immutable
result directories, private output routing, mismatch refusal and visible overrides,
synthetic trace integration, config validation, deterministic mock behavior and
provenance capture. Phase 2 adds warm-up exclusion, runtime cleanup, CPU reference
smoke run under approximately two minutes and reporting. Phase 5 adds controller
contract tests with mock variants; Phase 6 adds seeded perturbation tests.

Hardware tests are marked `jetson` and skipped when hardware is absent. Every
optional backend must have import tests proving its absence does not break core.

## Review questions

- Which distribution license should be used? No license is implied by this scaffold.
- Which public model family/dataset should the Phase 2 reference use? Defer this
  choice; do not download weights/data or create research experiments in Phase 0.
- What JetPack/L4T installation is on the initial device? Capture it in Phase 3
  before selecting compatible hardware packages.

## Phase 1 implementation decisions (2026-10-11)

- Keep metadata in each owning subsystem, with standard-library dataclasses and
  shared strict JSON conversion; reject unknown fields, invalid types, nonfinite
  quantities and unsupported schema versions.
- Use PyYAML only for config I/O. Resolve explicit fragment references within the
  config root, then apply explicit overrides and save the complete resolved config.
  Private fragments require private workload classification; private paths and
  results are confined to the workspace private tree.
- Capture the CPU host profile and Git/runtime provenance automatically. Missing
  Git metadata is an error; unavailable hardware settings carry explicit reasons.
- Bind a representative batch explicitly in the mock runtime. Mock power produces
  deterministic timestamped samples and always declares synthetic provenance.
- Integrate one explicitly selected power rail over a bounded window using the
  trapezoidal rule. Store total energy and optional steady-window per-inference
  energy with their methods; keep idle power separate.
- Validate the complete record before reserving an exclusive directory. Write an
  incomplete marker first and a completion marker last. Comparison reads accept
  only completed records, and visible override output lists setting mismatches.
- Phase 1 implements comparison checks and nominal/sparsity report descriptors;
  plots, profiling, real models, controllers and perturbations retain their later
  phase boundaries. No weights, datasets or research experiments are downloaded.
