from dataclasses import replace

import pytest

from edge_ai.power.base import PowerSource
from edge_ai.power.mock import MockPower
from edge_ai.power.trace import (
    EnergyResult,
    PowerMetadata,
    PowerTrace,
    energy_from_trace,
    integrate_trace,
)


def test_mock_is_deterministic_and_always_estimated():
    source = MockPower(seed=4, noise_w=1)
    assert isinstance(source, PowerSource)
    source.start()
    first = source.stop()
    source.start()
    assert source.stop() == first
    result = energy_from_trace(first, rail="total", start_s=0, end_s=1, inference_count=10)
    assert result.energy_j is None and result.integration_method is None
    assert result.energy_estimate_j > 0 and result.estimate_method
    assert result.per_inference_estimate_j == result.energy_estimate_j / 10
    assert not first.metadata.measured


def test_analytic_ramp_and_interpolated_window():
    trace = PowerTrace(
        metadata=PowerMetadata("test", False, 1, ("total",), None, "synthetic", "fixture"),
        timestamps_s=(0.0, 1.0, 2.0), watts={"total": (0.0, 2.0, 4.0)},
    )
    assert integrate_trace(trace, "total", 0, 2) == pytest.approx(4)
    assert integrate_trace(trace, "total", 0.5, 1.5) == pytest.approx(2)


def test_measured_trace_retains_measurement_method(record):
    trace = replace(record.power_trace, metadata=replace(
        record.power_trace.metadata, source="sensor-fixture", measured=True, clock="monotonic",
        resolution_w=0.01, description="Synthetic test fixture of a measured schema",
    ))
    energy = energy_from_trace(trace, rail="total", start_s=0, end_s=1, inference_count=3)
    measured = replace(record, power_trace=trace, energy=energy)
    assert measured.energy.energy_j == 5
    assert measured.energy.energy_estimate_j is None
    assert measured.energy.integration_method == "trapezoidal"


@pytest.mark.parametrize("start,end", [(-1, 1), (0, 2), (0.5, 0.5), (1, 0)])
def test_no_extrapolation_or_invalid_windows(record, start, end):
    with pytest.raises(ValueError):
        integrate_trace(record.power_trace, "total", start, end)


def test_multirail_selection_does_not_double_count(record):
    trace = replace(record.power_trace,
                    metadata=replace(record.power_trace.metadata, rails=("total", "cpu")),
                    watts={"total": (5.0,) * 11, "cpu": (2.0,) * 11})
    assert integrate_trace(trace, "total", 0, 1) == pytest.approx(5)
    assert integrate_trace(trace, "cpu", 0, 1) == pytest.approx(2)
    with pytest.raises(ValueError, match="absent"):
        integrate_trace(trace, "all", 0, 1)


@pytest.mark.parametrize("kwargs", [
    {"timestamps_s": (0.0, 0.0)}, {"timestamps_s": (0.0, float("inf"))},
    {"watts": {"total": (5.0,)}}, {"watts": {"total": (-1.0,) * 11}},
    {"watts": {"other": (5.0,) * 11}},
])
def test_invalid_traces_rejected(record, kwargs):
    with pytest.raises(ValueError):
        replace(record.power_trace, **kwargs)


def test_missing_samples_are_not_silently_integrated(record):
    trace = replace(record.power_trace, missing_samples=1)
    with pytest.raises(ValueError, match="missing samples"):
        integrate_trace(trace, "total", 0, 1)


def test_energy_requires_method_and_correct_average():
    with pytest.raises(ValueError, match="method"):
        EnergyResult("total", 0, 1, 10, energy_estimate_j=5)
    with pytest.raises(ValueError, match="inconsistent"):
        EnergyResult("total", 0, 1, 10, energy_estimate_j=5, estimate_method="fixture",
                     inference_count=2, per_inference_estimate_j=5, per_inference_method="fixture")
