"""The complete result record and scientific provenance validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, median
from typing import Any, Literal

from edge_ai.config import ExperimentConfig
from edge_ai.devices.profile import DeviceProfile
from edge_ai.power.trace import EnergyResult, PowerTrace, integrate_trace
from edge_ai.profiling.sustained import SustainedSummary, summarize_sustained
from edge_ai.runtimes.base import MemoryReport
from edge_ai.utils.provenance import Provenance
from edge_ai.utils.serialization import from_dict, positive, safe_name, to_dict, validate_types


@dataclass(frozen=True)
class TimingTrace:
    starts_s: tuple[float, ...]
    durations_s: tuple[float, ...]
    warmup_iterations_excluded: int
    measured: bool = True
    clock: Literal["monotonic", "synthetic"] = "monotonic"
    scope: Literal["model", "end_to_end"] = "model"
    unit: Literal["s"] = "s"

    def __post_init__(self) -> None:
        validate_types(self)
        positive(self.warmup_iterations_excluded, "excluded warmup", zero=True)
        if not self.durations_s or len(self.starts_s) != len(self.durations_s):
            raise ValueError("timing trace requires equal, nonempty starts and durations")
        previous_end = -1.0
        for start, duration in zip(self.starts_s, self.durations_s):
            positive(start, "timing start", zero=True)
            positive(duration, "inference duration")
            if start + 1e-12 < previous_end:
                raise ValueError("serial inference timing windows must not overlap")
            previous_end = start + duration
        if self.measured and self.clock != "monotonic":
            raise ValueError("measured timings require a monotonic clock")

    def summary(self, batch_size: int) -> dict[str, float | bool | str]:
        values = sorted(self.durations_s)

        def percentile(fraction: float) -> float:
            index = (len(values) - 1) * fraction
            lower = int(index)
            upper = min(lower + 1, len(values) - 1)
            return values[lower] + (values[upper] - values[lower]) * (index - lower)

        elapsed = self.starts_s[-1] + self.durations_s[-1] - self.starts_s[0]
        scope_duration = sum(self.durations_s) if self.scope == "model" else elapsed
        return {
            "mean_s": mean(values),
            "median_s": median(values),
            "p95_s": percentile(0.95),
            "p99_s": percentile(0.99),
            "inferences_per_s": len(values) / scope_duration,
            "items_per_s": len(values) * batch_size / scope_duration,
            "window_inferences_per_s": len(values) / elapsed,
            "window_items_per_s": len(values) * batch_size / elapsed,
            "scope": self.scope,
            "measured": self.measured,
        }


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    config: ExperimentConfig
    provenance: Provenance
    device: DeviceProfile
    timings: TimingTrace
    memory: MemoryReport
    run_duration_s: float
    power_trace: PowerTrace | None = None
    energy: EnergyResult | None = None
    end_to_end_timings: TimingTrace | None = None
    quality_metric: str | None = None
    quality: float | None = None
    temperature_start_c: float | None = None
    temperature_end_c: float | None = None
    schema_version: Literal[1, 2, 3] = 1
    idle_power_trace: PowerTrace | None = None
    dataset_manifest: dict[str, Any] | None = None
    sustained: SustainedSummary | None = None

    def __post_init__(self) -> None:
        validate_types(self)
        safe_name(self.run_id, "run_id")
        positive(self.run_duration_s, "run_duration_s")
        config = self.config
        if config.model.runtime in ("pytorch", "onnxruntime") and self.schema_version not in (2, 3):
            raise ValueError("real CPU backends require schema version 2 or 3")
        if self.schema_version in (2, 3):
            if any(
                item is None
                for item in (
                    config.artifact,
                    config.execution,
                    config.power,
                    config.workload.preprocessing,
                    config.workload.manifest_sha256,
                    self.dataset_manifest,
                    self.idle_power_trace,
                    self.power_trace,
                    self.energy,
                    self.end_to_end_timings,
                )
            ):
                raise ValueError("schema version 2 requires complete execution/data provenance")
            from edge_ai.datasets.manifest import DatasetManifest
            from edge_ai.utils.artifacts import validate_digest

            manifest = from_dict(DatasetManifest, self.dataset_manifest)
            if manifest.digest() != config.workload.manifest_sha256:
                raise ValueError("embedded dataset manifest identity differs from config")
            if (
                manifest.seed != config.seed
                or manifest.preprocessing != config.workload.preprocessing
            ):
                raise ValueError("dataset selection/preprocessing differs from config")
            validate_digest(config.model.weights_digest)
            if config.workload.conditions:
                raise ValueError("perturbations require a later phase")
            if not self.timings.measured or not self.end_to_end_timings.measured:
                raise ValueError("real CPU runs require measured timing traces")
            if not self.memory.measured or not self.memory.rss_bytes:
                raise ValueError("CPU records require sampled memory provenance")
            if (
                self.quality_metric != "top1_accuracy"
                or self.quality is None
                or not 0 <= self.quality <= 1
            ):
                raise ValueError("CPU reference requires top1_accuracy in [0, 1]")
            if self.idle_power_trace.metadata.measured != self.power_trace.metadata.measured:
                raise ValueError("idle and active power provenance differ")
            if any(trace.metadata.clock != "monotonic" for trace in (
                self.idle_power_trace, self.power_trace,
            )):
                raise ValueError("CPU power windows must share the observed monotonic clock")
            idle = self.idle_power_trace
            idle_mean = integrate_trace(
                idle, self.energy.rail, idle.timestamps_s[0], idle.timestamps_s[-1]
            ) / (idle.timestamps_s[-1] - idle.timestamps_s[0])
            if self.energy.idle_power_w is None or not math.isclose(
                idle_mean,
                self.energy.idle_power_w,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                raise ValueError("idle power differs from raw idle trace")
        if config.baseline_id == self.run_id:
            raise ValueError("a run cannot be its own baseline")
        if config.model.runtime not in self.provenance.runtime_versions:
            raise ValueError("provenance must include the selected runtime version")
        if not self.provenance.runtime_versions[config.model.runtime]:
            raise ValueError("selected runtime version must be known")
        if self.timings.scope != "model":
            raise ValueError("primary timing trace must use model scope")
        if config.protocol.sustained_duration_s is not None:
            if self.schema_version != 3 or self.sustained is None:
                raise ValueError("sustained runs require schema version 3 and a summary")
            if self.energy is None or self.end_to_end_timings is None:
                raise ValueError("sustained summary requires active window and end-to-end trace")
            actual_summary = summarize_sustained(
                config.protocol, self.end_to_end_timings,
                self.energy.window_start_s, self.energy.window_end_s,
            )
            if self.sustained != actual_summary:
                raise ValueError("sustained summary differs from raw timing/window provenance")
            if self.sustained.actual_duration_s > self.run_duration_s + 1e-12:
                raise ValueError("sustained active window exceeds declared run duration")
            expected = self.sustained.inference_count
        else:
            if self.schema_version == 3 or self.sustained is not None:
                raise ValueError("schema version 3 is reserved for sustained runs")
            expected = config.protocol.iterations * config.protocol.repetitions
        for trace in (self.timings, self.end_to_end_timings):
            if trace is None:
                continue
            if len(trace.durations_s) != expected:
                raise ValueError("timing count differs from resolved measurement protocol")
            if trace.warmup_iterations_excluded != config.protocol.warmup_iterations:
                raise ValueError("excluded warmup differs from the protocol")
            elapsed = trace.starts_s[-1] + trace.durations_s[-1] - trace.starts_s[0]
            if elapsed > self.run_duration_s + 1e-12:
                raise ValueError("timing trace exceeds declared run duration")
        if self.end_to_end_timings is not None and self.end_to_end_timings.scope != "end_to_end":
            raise ValueError("end-to-end timing trace must use end_to_end scope")
        if self.quality is not None and not self.quality_metric:
            raise ValueError("quality value requires metric name")
        if self.energy is not None:
            if self.power_trace is None:
                raise ValueError("energy requires a raw power trace")
            energy, power = self.energy, self.power_trace
            if energy.energy_j is not None and not power.metadata.measured:
                raise ValueError("energy_j requires a measured power source")
            if energy.sampling_hz != power.metadata.sampling_hz:
                raise ValueError("energy sampling rate differs from raw trace")
            if energy.inference_count is not None and energy.inference_count != expected:
                raise ValueError("energy inference count differs from measurement protocol")
            for trace in (self.timings, self.end_to_end_timings):
                if trace is not None and power.metadata.clock == trace.clock:
                    first = trace.starts_s[0]
                    last = trace.starts_s[-1] + trace.durations_s[-1]
                    if first < energy.window_start_s or last > energy.window_end_s + 1e-12:
                        raise ValueError("energy window must contain the counted inference timings")
            supported_methods = {"trapezoidal", "synthetic_trace_trapezoidal"}
            method = energy.integration_method or energy.estimate_method
            if method not in supported_methods:
                raise ValueError("unsupported energy method in schema version 1")
            if energy.energy_j is not None and method != "trapezoidal":
                raise ValueError("measured energy must declare the measured integration method")
            actual = integrate_trace(
                power,
                energy.rail,
                energy.window_start_s,
                energy.window_end_s,
            )
            declared = energy.energy_j if energy.energy_j is not None else energy.energy_estimate_j
            if not math.isclose(actual, declared, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("energy is inconsistent with the raw power trace")

    def validate(self) -> None:
        # Reconstruct too: nested mutable mappings must not bypass frozen dataclasses.
        from_dict(RunRecord, to_dict(self))
