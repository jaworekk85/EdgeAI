# Phase 3A: hardware preparation and sustained CPU profiling

Status: scope approved by the user on 2026-10-12, following review of Phase 2;
implemented and verified, awaiting phase review. This architecture was recorded
before implementation. Stop for review after 3A; Phase 3B remains pending.

## Outcome

Develop without a Jetson for the coming months. Exercise telemetry parsing and
repeatable sustained profiling on CPU; reserve hardware execution and validation
for Phase 3B. This is preparation, not a calibrated Jetson performance simulator.

## Architecture and boundaries

| Module | Responsibility |
| --- | --- |
| `devices/tegrastats.py` | Standard-library parser for documented RAM, CPU/GPU, temperature and instantaneous/average rail readings; preserve each original line and unavailable observations |
| `power/replay.py` | Convert explicitly timestamped fixture/log samples into synthetic power traces; selected rail only, no rail summation, no invented timestamps or silent gaps |
| `profiling/sustained.py` | Typed sustained summary, completed-pass stopping rules, latency windows and descriptive drift derived from raw timings |
| `profiling/runner.py` | Repeat complete workload passes until duration or inference cap; warm up once, bound retained outputs and release all resources on failure |
| `records` | Schema 3 only for sustained runs, validate actual counts/summary/window; preserve reading of schemas 1/2 and immutable storage |
| `reference.py` | Explicit sustained CLI overrides reuse verified prepared assets; resolved run config records overrides; subprocess timeout accounts for requested duration |
| `reporting` | Show requested/actual duration, stop reason, completed passes and latency drift; comparisons reject differing profiling protocols |

Use the existing runtime, power, execution and device contracts. CPU profiles
remain observations of the real host. A replay does not create a Jetson profile,
change host settings, or attach a historical temperature to a current CPU run.
Do not implement a live collector, TensorRT backend, thermal simulator, controller,
compression transform or new research experiment in 3A.

## Sustained measurement semantics

`iterations * repetitions` defines one complete pass through the workload.
Sustained mode repeats complete passes, checks its elapsed target after each pass,
and stops before a further pass would exceed `sustained_max_iterations` (default
100000). A pass may exceed the requested duration; actual window and count are
saved. A cap stop before target is explicitly reported, never called a completed
duration target. The cap must permit at least one full pass.

The reference cycles the same verified manifest in a fixed order. Quality counts
every repeated observation and is labeled repeated-subset accuracy, not a larger
independent dataset. In sustained mode score each output immediately after its
timing to avoid retaining an unbounded logits list. Scoring overhead is inside the
active power window, outside model/end-to-end timing. Short-run semantics stay
unchanged. RSS, raw timing and synthetic power storage grow with the bounded run.

Summarize latency in up to four consecutive equal-count windows; save window
positions/counts and mean/p95, and first-to-last mean latency change. These are
descriptive CPU observations, not evidence of thermal throttling. No CPU
temperature or throttling status is inferred from latency changes.

## Telemetry and replay provenance

Parse the documented subset of
[NVIDIA tegrastats output](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html).
Accept optional fields and known frequency formatting alternatives; refuse
malformed recognized fields, duplicates and nonfinite values. Preserve unsupported
text in the raw line; do not claim full coverage of all JetPack versions.

Replay requires caller-supplied strictly increasing relative timestamps, explicit
sampling rate, a source description and a rail present in every sample. Both
instantaneous and average reported mW values remain available; integrate only
instantaneous values converted to W. All replay traces have `measured=False` and
synthetic clocks. A recorded log's origin does not prove synchronization with a new
inference run. Sensor acquisition, reliable timestamp alignment, gaps and measured
energy are Phase 3B work. Fixtures are documented examples, not collected readings.

## Acceptance

- Core imports/tests work without Jetson, CUDA, TensorRT and optional CPU packages.
- Fake-clock tests establish warm-up exclusion, full-pass stopping, target/cap
  outcomes, count/energy consistency, scoring and cleanup on failure.
- Parser/replay tests cover units, format alternatives, optional fields, malformed
  samples, missing rails, timing gaps and estimated-energy enforcement.
- Schema/store/report tests round-trip sustained records and refuse inconsistent
  counts, protocol comparisons or summary claims; schemas 1/2 remain readable.
- Run the existing prepared public CPU smoke and a short sustained validation with
  both CPU backends. Multi-minute trials are available but not run automatically.
- Update status and stop before Phase 4. No automatic implementation of 4–6.

## Deferred hardware gate

Phase 3B needs a real device: automatic JetPack/L4T/CUDA/TensorRT observations,
live onboard power and thermal capture, synchronized sustained measurements,
TensorRT execution, on-device entry points and documented environment validation.
Each later phase has an independent CPU acceptance and pending Jetson validation;
3A completion does not mark Phase 3 or its hardware gate complete.
