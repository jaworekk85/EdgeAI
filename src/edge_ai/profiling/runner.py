"""Serial CPU profiler. Preparation/evaluation scopes stay explicit."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter, sleep
from typing import Callable

from edge_ai.config import ExperimentConfig
from edge_ai.evaluation.classification import numpy_logits, top1_correct
from edge_ai.power.trace import EnergyResult, PowerTrace, energy_from_trace, integrate_trace
from edge_ai.power.window_mock import WindowMockPower
from edge_ai.profiling.memory import RSSSampler
from edge_ai.profiling.sustained import SustainedSummary, summarize_sustained
from edge_ai.records.schema import TimingTrace
from edge_ai.runtimes.base import MemoryReport, RuntimeBackend


@dataclass(frozen=True)
class ProfileResult:
    timings: TimingTrace
    end_to_end_timings: TimingTrace
    power_trace: PowerTrace
    idle_power_trace: PowerTrace
    energy: EnergyResult
    memory: MemoryReport
    quality: float
    run_duration_s: float
    sustained: SustainedSummary | None = None


def profile_cpu(
    runtime: RuntimeBackend,
    config: ExperimentConfig,
    batch_source: Callable,
    *,
    clock: Callable = perf_counter,
    pause: Callable = sleep,
    sampler=None,
    materialize: Callable = numpy_logits,
    score: Callable = top1_correct,
) -> ProfileResult:
    if config.execution is None or config.power is None:
        raise ValueError("profiler requires explicit execution and power settings")
    if config.workload.batch_size != 1 or config.workload.conditions:
        raise ValueError("Phase 2 profiler supports batch 1 without perturbations")
    sampler = sampler or RSSSampler(config.execution.memory_interval_s)
    active = WindowMockPower(config.power.assumed_watts, config.power.sampling_hz, clock=clock)
    idle = WindowMockPower(config.power.assumed_idle_watts, config.power.sampling_hz, clock=clock)
    sampling = active_started = idle_started = False
    run_start = clock()
    try:
        sampler.start()
        sampling = True
        runtime.load(config.model)
        first, _ = batch_source(0)
        runtime.bind_batch(runtime.prepare_batch(first))
        runtime.warmup(config.protocol.warmup_iterations)
        runtime.synchronize()
        idle.start()
        idle_started = True
        pause(config.power.idle_duration_s)
        idle_started = False
        idle_trace = idle.stop()
        starts, durations, e2e_starts, e2e_durations = [], [], [], []
        outputs = []
        pass_size = config.protocol.iterations * config.protocol.repetitions
        sustained = config.protocol.sustained_duration_s is not None
        count = correct = 0
        active.start()
        active_started = True
        active_start = clock()
        while True:
            for index in range(pass_size):
                e2e_start = clock()
                batch, label = batch_source(index)
                native = runtime.prepare_batch(batch)
                runtime.synchronize()
                model_start = clock()
                output = runtime.infer(native)
                runtime.synchronize()
                model_end = clock()
                logits = materialize(output)
                e2e_end = clock()
                starts.append(model_start)
                durations.append(model_end - model_start)
                e2e_starts.append(e2e_start)
                e2e_durations.append(e2e_end - e2e_start)
                if sustained:
                    correct += score(logits, label)
                else:
                    outputs.append((logits, label))
                count += 1
            if (
                not sustained
                or clock() - active_start >= config.protocol.sustained_duration_s
                or count + pass_size > config.protocol.sustained_max_iterations
            ):
                break
        active_started = False
        trace = active.stop()
        if not sustained:
            correct = sum(score(logits, label) for logits, label in outputs)
        native_memory = runtime.memory_report()
        sampling = False
        memory = sampler.stop(native_memory.model_file_bytes)
        idle_mean = integrate_trace(
            idle_trace, "total", idle_trace.timestamps_s[0], idle_trace.timestamps_s[-1]
        ) / (idle_trace.timestamps_s[-1] - idle_trace.timestamps_s[0])
        timings = TimingTrace(tuple(starts), tuple(durations), config.protocol.warmup_iterations)
        e2e_timings = TimingTrace(
            tuple(e2e_starts), tuple(e2e_durations), config.protocol.warmup_iterations,
            scope="end_to_end",
        )
        summary = summarize_sustained(
            config.protocol, e2e_timings, trace.timestamps_s[0], trace.timestamps_s[-1]
        ) if sustained else None
        return ProfileResult(
            timings,
            e2e_timings,
            trace,
            idle_trace,
            energy_from_trace(
                trace,
                rail="total",
                start_s=trace.timestamps_s[0],
                end_s=trace.timestamps_s[-1],
                inference_count=count,
                idle_power_w=idle_mean,
            ),
            memory,
            correct / count,
            clock() - run_start,
            summary,
        )
    finally:
        # Attempt every cleanup even if an earlier cleanup raises.
        try:
            if active_started:
                active.stop()
        finally:
            try:
                if idle_started:
                    idle.stop()
            finally:
                try:
                    if sampling:
                        sampler.stop()
                finally:
                    runtime.release()
