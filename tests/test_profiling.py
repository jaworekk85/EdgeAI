from dataclasses import replace

import pytest

from edge_ai.config import ExecutionSettings, PowerSettings
from edge_ai.power.window_mock import WindowMockPower
from edge_ai.profiling.runner import profile_cpu
from edge_ai.runtimes.base import MemoryReport


class Clock:
    def __init__(self):
        self.time = 100.0

    def __call__(self):
        return self.time

    def advance(self, seconds):
        self.time += seconds


class FakeRuntime:
    def __init__(self, clock, failure=None):
        self.clock, self.failure = clock, failure
        self.inferences = self.warmups = 0
        self.released = False

    def load(self, variant):
        if self.failure == "load":
            raise RuntimeError("load failure")
        self.clock.advance(2)

    def prepare_batch(self, batch):
        self.clock.advance(0.1)
        return batch

    def bind_batch(self, batch):
        pass

    def warmup(self, n):
        self.warmups = n
        self.clock.advance(n * 10)

    def synchronize(self):
        pass

    def infer(self, batch):
        if self.failure == "infer":
            raise RuntimeError("inference failure")
        self.inferences += 1
        self.clock.advance(0.2)
        return batch

    def memory_report(self):
        return MemoryReport(model_file_bytes=42, measured=True, method="fixture")

    def release(self):
        self.released = True


class FakeSampler:
    def __init__(self, failure=None):
        self.started = self.stopped = False
        self.failure = failure

    def start(self):
        self.started = True

    def stop(self, size=None):
        self.stopped = True
        if self.failure:
            raise RuntimeError("sampler failure")
        return MemoryReport(
            model_file_bytes=size,
            peak_ram_bytes=123,
            measured=True,
            method="fixture",
            sample_times_s=(1.0,),
            rss_bytes=(123,),
        )


def test_profiler_excludes_warmup_and_keeps_scopes_and_energy_aligned(config):
    config = replace(config, execution=ExecutionSettings(), power=PowerSettings())
    clock, sampler = Clock(), FakeSampler()
    runtime = FakeRuntime(clock)

    def source(index):
        clock.advance(0.3)
        return [index], 0

    def convert(output):
        clock.advance(0.4)
        return output

    result = profile_cpu(
        runtime,
        config,
        source,
        clock=clock,
        pause=clock.advance,
        sampler=sampler,
        materialize=convert,
        score=lambda output, label: 1,
    )
    assert runtime.warmups == 2 and runtime.inferences == 3
    assert result.timings.durations_s == pytest.approx((0.2,) * 3)
    assert result.end_to_end_timings.durations_s == pytest.approx((1.0,) * 3)
    assert result.energy.energy_estimate_j == pytest.approx(15)
    assert result.energy.per_inference_estimate_j == pytest.approx(5)
    assert result.power_trace.timestamps_s[0] > 120
    assert result.energy.energy_j is None
    assert result.energy.idle_power_w == pytest.approx(1)
    assert result.quality == 1 and runtime.released and sampler.stopped


@pytest.mark.parametrize("failure", ["load", "infer", "sampler", "source", "score"])
@pytest.mark.parametrize("sustained", [False, True])
def test_profiler_cleans_up_after_failure(config, failure, sustained):
    config = replace(config, execution=ExecutionSettings(), power=PowerSettings())
    if sustained:
        config = replace(config, protocol=replace(config.protocol, sustained_duration_s=1))
    clock, sampler = Clock(), FakeSampler("sampler" if failure == "sampler" else None)
    runtime = FakeRuntime(clock, failure)

    def source(index):
        if failure == "source":
            raise RuntimeError("source failure")
        clock.advance(0.1)
        return [1], 0

    def score(output, label):
        if failure == "score":
            raise RuntimeError("score failure")
        return 1

    with pytest.raises(RuntimeError):
        profile_cpu(
            runtime,
            config,
            source,
            clock=clock,
            pause=clock.advance,
            sampler=sampler,
            materialize=lambda x: x,
            score=score,
        )
    assert runtime.released and sampler.stopped


def test_window_mock_short_window_and_lifecycle():
    clock = Clock()
    source = WindowMockPower(5, 100, clock=clock)
    source.start()
    clock.advance(0.001)
    trace = source.stop()
    assert len(trace.timestamps_s) == 2 and trace.timestamps_s[-1] == clock()
    assert not trace.metadata.measured and trace.metadata.clock == "monotonic"
    with pytest.raises(RuntimeError):
        source.stop()


@pytest.mark.parametrize("target,cap,count,reason", [
    (4, 100, 6, "duration_reached"),
    (30, 5, 3, "inference_cap"),
    (30, 6, 6, "inference_cap"),
    (0.1, 3, 3, "duration_reached"),
])
def test_sustained_complete_passes_actual_counts_and_estimates(config, target, cap, count, reason):
    from edge_ai.profiling.sustained import summarize_sustained

    config = replace(
        config, execution=ExecutionSettings(), power=PowerSettings(),
        protocol=replace(
            config.protocol, sustained_duration_s=target, sustained_max_iterations=cap,
        ),
    )
    clock, sampler = Clock(), FakeSampler()
    runtime = FakeRuntime(clock)
    order, scoring_times = [], []

    def source(index):
        order.append(index)
        clock.advance(0.3)
        return [index], 0

    def convert(output):
        clock.advance(0.4)
        return output

    def score(output, label):
        scoring_times.append(clock())
        return output[0] == 0

    result = profile_cpu(
        runtime, config, source, clock=clock, pause=clock.advance, sampler=sampler,
        materialize=convert, score=score,
    )
    assert runtime.warmups == 2 and runtime.inferences == count
    assert order == [0] + [0, 1, 2] * (count // 3)
    assert result.quality == pytest.approx(1 / 3)
    assert len(scoring_times) == count and scoring_times[0] < scoring_times[-1]
    assert result.sustained.completed_passes == count // 3
    assert result.sustained.stop_reason == reason
    assert result.sustained.actual_duration_s == pytest.approx(count)
    assert result.sustained.latency_change_fraction == pytest.approx(0)
    assert result.energy.inference_count == count and result.energy.energy_j is None
    assert result.energy.energy_estimate_j == pytest.approx(5 * count)
    assert result.energy.per_inference_estimate_j == pytest.approx(5)
    assert all(time >= result.energy.window_start_s for time in scoring_times)
    assert result.sustained == summarize_sustained(
        config.protocol, result.end_to_end_timings,
        result.energy.window_start_s, result.energy.window_end_s,
    )
    assert runtime.released and sampler.stopped


def test_sustained_summary_refuses_early_stops_partial_passes_and_scope():
    from edge_ai.profiling.protocol import ProfilingProtocol
    from edge_ai.profiling.sustained import summarize_sustained
    from edge_ai.records.schema import TimingTrace

    protocol = ProfilingProtocol(0, 2, 1, 10, 10)
    timings = TimingTrace((0, 1), (0.5, 0.5), 0, scope="end_to_end")
    with pytest.raises(ValueError, match="before either"):
        summarize_sustained(protocol, timings, 0, 2)
    with pytest.raises(ValueError, match="complete passes"):
        summarize_sustained(protocol, replace(timings, starts_s=(0,), durations_s=(0.5,)), 0, 2)
    with pytest.raises(ValueError, match="end-to-end"):
        summarize_sustained(protocol, replace(timings, scope="model"), 0, 2)
    with pytest.raises(ValueError, match="after a completed pass"):
        summarize_sustained(
            replace(protocol, sustained_duration_s=1),
            replace(timings, starts_s=(0, 1, 2, 3), durations_s=(0.5,) * 4), 0, 4,
        )
