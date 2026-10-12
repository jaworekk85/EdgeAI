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
def test_profiler_cleans_up_after_failure(config, failure):
    config = replace(config, execution=ExecutionSettings(), power=PowerSettings())
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
