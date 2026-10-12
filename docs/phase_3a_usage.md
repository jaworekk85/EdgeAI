# Phase 3A usage: sustained CPU runs and offline telemetry

Use the tested Phase 2 dependencies and prepared assets from
[CPU reference setup](cpu_reference.md). No Jetson, CUDA, TensorRT, extra package
installation or new asset download is needed. See [design](phase_3a_plan.md) and
[CPU/hardware roadmap](cpu_hardware_roadmap.md).

## Sustained CPU validation

```sh
python -m edge_ai.reference run --prepared artifacts/reference/9029ef52961e20ad/prepared.json --sustained-seconds 2 --max-inferences 2000
```

This requests two seconds of active profiling **per model/runtime**, excluding
startup, load, warm-up, idle sampling, reporting and storage. The default matrix
contains four runs. It is a short validation of sustained mode, not a thermal
study. The prepared identity is checked before overrides; the complete resolved
config in each new run records the requested duration and cap. Old verified
Phase 2 preparations with absent new default fields remain usable without edits.

When a longer trial is deliberately needed, `--sustained-seconds 120` requests
two minutes per configuration. Runs finish a complete workload pass before
checking the target. The inference cap (default 100000) may stop a run early;
`inference_cap` is explicit and does not certify the requested duration. Duration
may exceed the target by a full pass; workers also have a duration-aware timeout.
Raw timing/power/RSS storage grows with run duration and capped inference count.

Both CPU backends repeat the same manifest in order. Top-1 accuracy counts repeated
observations of the same subset; repeating it does not enlarge the evaluation
dataset. Sustained scoring occurs after each end-to-end timing, inside the active
power window. Model/end-to-end latency exclude scoring. Short-mode scoring remains
after the active window. Constant mock power always produces energy estimates.

Schema-3 runs store `traces/sustained.json`, the raw traces, complete config and
all standard reports before the completion marker. The sustained report contains
target/actual seconds, actual inference count, complete passes, stop reason and
up to four consecutive equal-count end-to-end latency windows. First-to-last mean
change is descriptive; the framework does not infer temperature or throttling.
Different profiling protocols are refused in comparisons unless mismatch is
explicitly allowed and displayed. Schemas 1/2 remain readable without rewriting.

## Offline telemetry examples

The parser follows the documented subset of
[NVIDIA tegrastats output](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html):
RAM, CPU/GPU utilization/frequencies, temperatures and instantaneous/average power.
It preserves the complete original line, RAM in source MB, temperatures in Celsius,
frequencies in MHz and power converted from mW to W. Missing observations remain
absent; unsupported fields remain in the raw text. This is not exhaustive coverage
of all JetPack formats and has not been verified against this user's device.

```python
from edge_ai.devices.tegrastats import parse_tegrastats
from edge_ai.power.replay import ReplaySample, replay_power
from edge_ai.power.trace import energy_from_trace

# Hand-written format fixtures, not sensor readings.
first = parse_tegrastats("RAM 220/8192MB CPU@45C VDD_IN 5000mW/4000mW")
second = parse_tegrastats("RAM 240/8192MB CPU@46C VDD_IN 7000mW/4500mW")
samples = (ReplaySample(0.0, first), ReplaySample(0.5, second))
trace = replay_power(
    samples, rail="VDD_IN", sampling_hz=2.0, description="hand-written format fixture",
)
energy = energy_from_trace(trace, rail="VDD_IN", start_s=0.0, end_s=0.5)
assert trace.metadata.measured is False
assert energy.energy_j is None
assert energy.energy_estimate_j == 3.0
```

Replay requires explicit relative timestamps and source description. Consecutive
timestamps must match the supplied sampling interval within 1%; absent rails,
duplicates, unordered timestamps or larger gaps fail. This conservative rule
tests clean fixtures; a real collector will need an observed cadence/gap policy.
Only the selected instantaneous rail is integrated. Average readings remain
available in each parsed sample. Never add an aggregate rail to its constituent
rails. Rail meaning and coverage require device-specific validation in 3B.

Replay always uses synthetic clocks and unmeasured power. The API accepts no
switch to promote a replay to measured power. Its samples retain raw lines,
temperatures and timestamps, but cannot identify or replace the CPU host profile
or be synchronized with a current CPU run. Live collection and measured energy
are deferred to 3B; replay is an offline utility, not a runtime power provider.

## Verification

```sh
python -m pytest tests -q
python -m ruff check src tests
python -m pytest tests -q --reference-prepared artifacts/reference/9029ef52961e20ad/prepared.json
```

The opt-in option runs both the original four-worker reference smoke and a
two-second sustained matrix. Tests never download assets or require hardware.
Without CPU extras, those fixtures skip and the core/parser/fake-clock tests run.
