from dataclasses import replace

import pytest

from edge_ai.compression.spec import CompressionSpec
from edge_ai.reporting.comparison import comparison_rows


@pytest.mark.parametrize("setting", ["resolution", "power_mode", "batch_size", "runtime", "clocks"])
def test_mismatch_requires_override_and_is_visible(record, second_record, setting, capsys):
    config = second_record.config
    device = second_record.device
    provenance = second_record.provenance
    if setting == "resolution":
        config = replace(config, model=replace(config.model, resolution=(16, 16)),
                         workload=replace(config.workload, resolution=(16, 16)))
    elif setting == "batch_size":
        config = replace(config, workload=replace(config.workload, batch_size=2))
    elif setting == "runtime":
        config = replace(config, model=replace(config.model, runtime="other-fixture"))
        provenance = replace(provenance, runtime_versions={"other-fixture": "1"})
    else:
        device = replace(device, **{setting: "other-fixture"})
    changed = replace(second_record, config=config, device=device, provenance=provenance)
    with pytest.raises(ValueError, match=setting):
        comparison_rows([record, changed])
    rows = comparison_rows([record, changed], allow_mismatch=True)
    assert setting in capsys.readouterr().out
    assert setting in rows[0]["settings_mismatch"]
    assert rows[0]["settings_mismatch"] == rows[1]["settings_mismatch"]


def test_nominal_sparsity_and_estimate_labels_survive_reporting(record):
    compression = CompressionSpec(
        method="unstructured_pruning", sparsity_level=0.5, sparsity_pattern="unstructured",
        kernel_support=False, kernel_support_method="declared fixture",
        parameters=100, flops=200, counting_method="fixture", counting_input_shape=(1, 3, 8, 8),
    )
    changed = replace(record, config=replace(
        record.config, model=replace(record.config.model, compression=compression),
    ))
    row = comparison_rows([changed])[0]
    assert row["parameters"] == 100 and row["flops"] == 200
    assert "without kernel support" in row["sparsity_label"]
    assert row["kernel_support"] is False and row["sparsity_pattern"] == "unstructured"
    assert row["energy_j"] is None and row["energy_estimate_j"] == pytest.approx(5)
    assert row["latency"]["measured"] is False and row["power_measured"] is False


def test_baseline_changes_are_checked(record, second_record):
    config = replace(second_record.config,
                     baseline_id=record.run_id,
                     model=replace(second_record.config.model, id="different-model"))
    changed = replace(second_record, config=config)
    with pytest.raises(ValueError, match="baseline differences"):
        comparison_rows([record, changed])
    changed = replace(changed, config=replace(config, changed_components=("model",)))
    assert comparison_rows([record, changed])[1]["changed_components"] == ("model",)


def test_empty_or_duplicate_comparisons_fail(record):
    with pytest.raises(ValueError, match="at least one"):
        comparison_rows([])
    with pytest.raises(ValueError, match="duplicate"):
        comparison_rows([record, record])


@pytest.mark.parametrize("kwargs", [
    {"method": "unstructured_pruning"}, {"sparsity_level": 2},
    {"kernel_support": True}, {"parameters": 100}, {"flops": float("inf")},
])
def test_compression_provenance_is_required(kwargs):
    with pytest.raises(ValueError):
        CompressionSpec(**kwargs)
