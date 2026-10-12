# Phase 1 core API

The architecture review and core implementation were authorized together on
2026-10-11. The examples below exercise the original contracts with synthetic
data. Real CPU execution, profiling and reports now exist; see
[CPU reference usage](cpu_reference.md) for the Phase 2 extensions.

## Configuration and mocks

From the repository root, after installing `.[dev]`:

```python
from pathlib import Path

from edge_ai.config import load_config
from edge_ai.devices.profile import capture_device_profile
from edge_ai.power.mock import MockPower
from edge_ai.power.trace import energy_from_trace
from edge_ai.runtimes.mock import MockRuntime
from edge_ai.utils.provenance import capture_provenance
from edge_ai.utils.serialization import to_dict

workspace = Path.cwd()
config = load_config(
    Path("configs/experiments/core_mock.yaml"), workspace=workspace,
    overrides={"seed": 12},
)
device = capture_device_profile()
provenance = capture_provenance(workspace)
resolved_config = to_dict(config)

runtime = MockRuntime(seed=config.seed)
runtime.load(config.model)
try:
    runtime.bind_batch([1.0])
    runtime.warmup(config.protocol.warmup_iterations)
    output = runtime.infer([1.0])
    memory = runtime.memory_report()
finally:
    runtime.release()

power = MockPower(seed=config.seed)
power.start()
trace = power.stop()
estimate = energy_from_trace(trace, rail="total", start_s=0.0, end_s=1.0)
assert estimate.energy_j is None
assert estimate.energy_estimate_j is not None
```

Mock output is a deterministic numeric fixture, not a quality prediction. Mock
memory values are unavailable, not zero. Its power clock is synthetic and does
not follow the duration between `start()` and `stop()`. The estimate above must
not be presented as energy consumed by the inference above. In particular, do
not divide it by a real inference count without explicitly describing that
synthetic measurement window. Idle baseline collection is a later profiler task.

Git capture fails if a commit cannot be obtained; it does not invent provenance.
It captures the dirty flag including untracked, nonignored files. Runtime versions
are read from installed package metadata without importing runtime libraries.
The host profile includes memory where queryable and explicit reasons for missing
hardware observations. Requested device settings are separate from observations.

## Records and immutable storage

`edge_ai.records.schema.RunRecord` contains the full `ExperimentConfig`, provenance,
device profile, raw `TimingTrace`, `MemoryReport`, run duration, optional raw power
trace/energy, task-quality metric/value and temperatures. Config includes the seed,
model/compression identity, workload, protocol, baseline ID and changed components.
Unknown metrics remain `None`. Version 1 covers the original core; version 2 adds
CPU execution/artifact/data/memory/idle provenance. Old version 1 files remain readable.

Construct timings from measurement samples only; `warmup_iterations_excluded`
must match the configured warm-up count and the sample count must match
`iterations * repetitions`. Synthetic timings must declare `measured=False` and
`clock="synthetic"`. `summary(batch_size)` derives mean, median, p95, p99 and
explicitly scoped inferences/items per second. End-to-end timing is separate.
Actual timing/synchronization/warm-up exclusion uses the implemented Phase 2 runner.

`energy_from_trace` integrates one explicitly selected rail using trapezoidal
integration with interpolation at window edges. Extrapolation, nonmonotonic
samples, nonfinite/negative power, unknown rails and declared sample loss are
refused. `EnergyResult` stores total measured or estimated joules and the matching
method, window and sampling rate. Optional window-average per-inference energy
retains the same provenance and declares its averaging method. An optional idle
power value is separate; above-idle reports are later reporting work. Record
validation checks energy against the raw trace and validates counted timings
against the energy window when clocks share a basis.

Use `new_run_id(config)` from `edge_ai.records.store` for a UTC timestamp plus
randomized short hash, then build and validate the record. `ResultStore(workspace)`:

- `write(record)` reserves a fresh directory exclusively. Public results go to
  `results/<experiment>/<run_id>/`; private results to
  `private/results/<experiment>/<run_id>/`. Known private paths cannot be declared
  public. Remote/private identifiers still require truthful caller classification.
- Writes `config.yaml`, `device.json`, the complete record as `metrics.json`, raw
  trace JSON under `traces/`, and a `plots/` directory for later reporting.
- Writes `incomplete.json` first and `complete.json` last. A successful run retains
  both; the completion marker is authoritative. Failed reservations remain on disk
  and cannot be reused. API writes never modify an existing run directory.
- `read(run_path)` requires a completion marker and validates schema, directory
  identity and privacy routing. Files are not protected against manual external
  editing by filesystem permissions; immutability is enforced by the write API.

## Comparisons

`edge_ai.reporting.comparison.comparison_rows(records)` validates records and
refuses different resolution, batch size, runtime, power mode or clocks.
`allow_mismatch=True` prints all differing settings and includes them in every row.
Rows keep nominal parameters/FLOPs and their counting provenance beside latency,
memory, power and separate measured/estimated energy fields. Pruning always shows
sparsity pattern, kernel support and a label for unavailable/unknown support.

Baseline attribution is recorded in config using `baseline_id` and
`changed_components` (`model`, `compression`, `runtime`, `device_settings`). When
the referenced baseline is among the compared runs, declared changes must match
the actual metadata differences. A missing baseline cannot be independently
checked. Phase 2 `render_reports` adds Markdown/CSV tables and Pareto plots, and
`load_stored_comparison` checks all referenced peers are complete. Reports are
prepared before writing and stored before the completion marker.
