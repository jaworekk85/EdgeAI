from dataclasses import replace

import pytest

from edge_ai.records.schema import RunRecord, TimingTrace
from edge_ai.utils.serialization import from_dict, to_dict


def test_complete_record_roundtrips(record):
    restored = from_dict(RunRecord, to_dict(record))
    assert restored == record
    restored.validate()


def test_synthetic_power_cannot_be_promoted_to_measured_energy(record):
    fake = replace(
        record.energy, energy_j=5, energy_estimate_j=None, integration_method="trapezoidal",
        estimate_method=None, per_inference_j=5 / 3, per_inference_estimate_j=None,
    )
    with pytest.raises(ValueError, match="measured power source"):
        replace(record, energy=fake)


@pytest.mark.parametrize("field,value", [
    ("schema_version", 2), ("schema_version", True), ("unknown", "bad"),
    ("run_id", "../escape"), ("quality", float("nan")), ("quality", 0.5),
    ("run_duration_s", 0.1), ("power_trace", None),
])
def test_invalid_records_rejected(record, field, value):
    data = to_dict(record)
    data[field] = value
    with pytest.raises(ValueError):
        from_dict(RunRecord, data)


def test_energy_must_match_trace_and_inference_count(record):
    wrong = replace(record.energy, energy_estimate_j=6, per_inference_estimate_j=2)
    with pytest.raises(ValueError, match="inconsistent with the raw"):
        replace(record, energy=wrong)
    wrong = replace(record.energy, inference_count=1, per_inference_estimate_j=5)
    with pytest.raises(ValueError, match="inference count"):
        replace(record, energy=wrong)
    with pytest.raises(ValueError, match="sampling rate"):
        replace(record, energy=replace(record.energy, sampling_hz=100))


def test_energy_without_trace_is_invalid_but_missing_both_is_explicit(record):
    empty = replace(record, power_trace=None, energy=None)
    assert empty.energy is None
    with pytest.raises(ValueError, match="raw power trace"):
        replace(record, power_trace=None)


def test_energy_window_contains_counted_timings(record):
    from edge_ai.power.trace import energy_from_trace

    window = energy_from_trace(
        record.power_trace, rail="total", start_s=0.5, end_s=1, inference_count=3,
    )
    with pytest.raises(ValueError, match="window must contain"):
        replace(record, energy=window)


def test_self_baseline_is_rejected(record):
    with pytest.raises(ValueError, match="own baseline"):
        replace(record, config=replace(record.config, baseline_id=record.run_id))


def test_nested_mutation_cannot_bypass_save_validation(record):
    record.provenance.runtime_versions["mock"] = None
    with pytest.raises(ValueError, match="version must be known"):
        record.validate()


def test_timing_summary_uses_measurement_samples_and_labels(record):
    summary = record.timings.summary(record.config.workload.batch_size)
    assert summary["mean_s"] == pytest.approx(0.02)
    assert summary["median_s"] == pytest.approx(0.02)
    assert summary["p95_s"] == pytest.approx(0.029)
    assert summary["p99_s"] == pytest.approx(0.0298)
    assert summary["items_per_s"] == pytest.approx(3 / 0.23)
    assert not summary["measured"] and summary["scope"] == "model"


def test_warmup_and_count_provenance_must_match_protocol(record):
    with pytest.raises(ValueError, match="excluded warmup"):
        replace(record, timings=replace(record.timings, warmup_iterations_excluded=0))
    trace = replace(record.timings, starts_s=(0.0,), durations_s=(0.01,))
    with pytest.raises(ValueError, match="timing count"):
        replace(record, timings=trace)


def test_timing_validation_and_end_to_end_scope(record):
    with pytest.raises(ValueError, match="overlap"):
        TimingTrace((0.0, 0.01), (0.1, 0.1), 0)
    with pytest.raises(ValueError, match="monotonic"):
        replace(record.timings, measured=True)
    with pytest.raises(ValueError, match="end_to_end scope"):
        replace(record, end_to_end_timings=record.timings)
