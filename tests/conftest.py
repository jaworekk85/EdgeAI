from dataclasses import replace
from pathlib import Path

import pytest

from edge_ai.config import load_config
from edge_ai.devices.profile import capture_device_profile
from edge_ai.power.mock import MockPower
from edge_ai.power.trace import energy_from_trace
from edge_ai.records.schema import RunRecord, TimingTrace
from edge_ai.runtimes.base import MemoryReport
from edge_ai.utils.provenance import Provenance

WORKSPACE = Path(__file__).resolve().parents[1]


def pytest_addoption(parser):
    parser.addoption(
        "--reference-prepared",
        default=None,
        help="Run the public four-worker smoke using this prepared.json path",
    )


@pytest.fixture
def config():
    return load_config(Path("configs/experiments/core_mock.yaml"), workspace=WORKSPACE)


@pytest.fixture
def record(config):
    power = MockPower(seed=config.seed)
    power.start()
    trace = power.stop()
    return RunRecord(
        run_id="20261011T100000000000Z_0123456789",
        config=config,
        provenance=Provenance(
            git_commit="a" * 40,
            git_dirty=False,
            captured_at_utc="2026-10-11T10:00:00+00:00",
            runtime_versions={"mock": "1"},
        ),
        device=capture_device_profile(),
        timings=TimingTrace(
            starts_s=(0.0, 0.1, 0.2),
            durations_s=(0.01, 0.02, 0.03),
            warmup_iterations_excluded=2,
            measured=False,
            clock="synthetic",
        ),
        memory=MemoryReport(method="synthetic fixture; unavailable"),
        run_duration_s=1.0,
        power_trace=trace,
        energy=energy_from_trace(trace, rail="total", start_s=0, end_s=1, inference_count=3),
    )


@pytest.fixture
def second_record(record):
    return replace(record, run_id="20261011T100001000000Z_abcdef0123")
