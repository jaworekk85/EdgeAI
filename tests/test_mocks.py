import pytest

from edge_ai.power.mock import MockPower
from edge_ai.runtimes.base import RuntimeBackend
from edge_ai.runtimes.mock import MockRuntime


def test_runtime_lifecycle_determinism_and_warmup_accounting(config):
    runtime = MockRuntime(seed=7)
    assert isinstance(runtime, RuntimeBackend)
    with pytest.raises(RuntimeError, match="no model"):
        runtime.infer([1])
    runtime.load(config.model)
    with pytest.raises(RuntimeError, match="representative"):
        runtime.warmup(2)
    runtime.bind_batch([1, 2])
    runtime.warmup(2)
    assert runtime.warmup_calls == 2 and runtime.inference_calls == 0
    first = runtime.infer([1, 2])
    assert runtime.infer([1, 2]) == first
    assert runtime.inference_calls == 2
    memory = runtime.memory_report()
    assert memory.peak_ram_bytes is None and not memory.measured
    with pytest.raises(RuntimeError, match="release"):
        runtime.load(config.model)
    runtime.release()
    runtime.release()
    with pytest.raises(RuntimeError):
        runtime.infer([1])
    runtime.load(config.model)
    assert runtime.infer([1, 2]) == first
    other = MockRuntime(seed=8)
    other.load(config.model)
    assert other.infer([1, 2]) != first


@pytest.mark.parametrize("batch", [[], [True], [float("nan")], ["1"], None])
def test_mock_rejects_invalid_batches(config, batch):
    runtime = MockRuntime()
    runtime.load(config.model)
    with pytest.raises(ValueError):
        runtime.infer(batch)


def test_power_lifecycle():
    power = MockPower()
    with pytest.raises(RuntimeError, match="not started"):
        power.stop()
    power.start()
    with pytest.raises(RuntimeError, match="already started"):
        power.start()
    power.stop()


def test_mock_accepts_signed_normalized_inputs(config):
    runtime = MockRuntime(seed=7)
    runtime.load(config.model)
    runtime.bind_batch([-1.0, 0.0, 1.0])
    runtime.warmup(1)
    assert len(runtime.infer([-1.0, 0.0, 1.0])) == 3
