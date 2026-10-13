"""Comparable-setting guards and explicit scientific report descriptors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from edge_ai.records.schema import RunRecord
from edge_ai.utils.serialization import safe_name, to_dict


def _settings(record: RunRecord) -> dict[str, Any]:
    return {
        "resolution": record.config.workload.resolution,
        "power_mode": record.device.power_mode,
        "clocks": record.device.clocks,
        "batch_size": record.config.workload.batch_size,
        "runtime": record.config.model.runtime,
        "manifest": record.config.workload.manifest_sha256,
        "preprocessing": to_dict(record.config.workload.preprocessing),
        "execution": to_dict(record.config.execution),
        "quality_metric": record.quality_metric,
        "profiling_protocol": to_dict(record.config.protocol),
        "workload": {
            "task": record.config.workload.task,
            "source": record.config.workload.source,
            "source_id": record.config.workload.source_id,
            "split": record.config.workload.split,
            "subset": record.config.workload.subset,
        },
    }


def baseline_changes(baseline: RunRecord, record: RunRecord) -> set[str]:
    old, new = to_dict(baseline.config.model), to_dict(record.config.model)
    changes = set()
    old_compression, new_compression = old.pop("compression"), new.pop("compression")
    if old_compression.get("details") is not None or new_compression.get("details") is not None:
        # A derived variant's label/precision is compression, not new pretrained weights.
        old.pop("id")
        new.pop("id")
        if old.pop("precision") != new.pop("precision"):
            changes.add("compression")
    # Nominal counts describe the model, not an applied compression transformation.
    for key in (
        "parameters",
        "flops",
        "counting_method",
        "counting_input_shape",
        "counting_exclusions",
    ):
        old_compression.pop(key)
        new_compression.pop(key)
    for item in (old_compression, new_compression):
        details = item.pop("details", None)
        if details is not None:
            item["transformation"] = {
                key: details[key] for key in (
                    "recipe", "changed_tensors", "target_elements", "total_weight_elements",
                    "precision_scope", "zeros_before", "zeros_after", "masked_elements",
                    "skipped_tensors", "calibration_manifest",
                )
            }
    if old_compression != new_compression:
        changes.add("compression")
    if old.pop("runtime") != new.pop("runtime"):
        changes.add("runtime")
    if old != new:
        changes.add("model")
    if (
        baseline.device != record.device
        or baseline.config.device != record.config.device
        or baseline.config.execution != record.config.execution
    ):
        changes.add("device_settings")
    return changes


def comparison_rows(
    records: list[RunRecord],
    *,
    allow_mismatch: bool = False,
) -> list[dict[str, Any]]:
    """Return descriptors and print differing settings for explicit overrides.

    Table rendering and Pareto plots are Phase 2. Each row already keeps nominal
    metrics, measured quantities, estimates and sparsity support side by side.
    """
    if not records:
        raise ValueError("comparison requires at least one record")
    for record in records:
        record.validate()
    if len({record.run_id for record in records}) != len(records):
        raise ValueError("comparison contains duplicate run ids")
    settings = [_settings(record) for record in records]
    differing = [
        key for key in settings[0] if any(item[key] != settings[0][key] for item in settings[1:])
    ]
    if differing and not allow_mismatch:
        raise ValueError(f"incomparable settings: {', '.join(differing)}")
    mismatch = {
        key: {record.run_id: setting[key] for record, setting in zip(records, settings)}
        for key in differing
    }
    if mismatch:
        print(
            "Comparison settings mismatch (allow_mismatch=True): "
            + json.dumps(to_dict(mismatch), sort_keys=True)
        )
    by_id = {record.run_id: record for record in records}
    rows = []
    for record in records:
        if record.config.baseline_id in by_id:
            actual = baseline_changes(by_id[record.config.baseline_id], record)
            if set(record.config.changed_components) != actual:
                raise ValueError(f"changed_components do not match baseline differences: {actual}")
        compression = record.config.model.compression
        latency = record.timings.summary(record.config.workload.batch_size)
        energy = record.energy
        average_power = None
        if energy is not None:
            total = energy.energy_j if energy.energy_j is not None else energy.energy_estimate_j
            average_power = total / (energy.window_end_s - energy.window_start_s)
        rows.append(
            {
                "run_id": record.run_id,
                "variant": record.config.model.id,
                "quality_metric": record.quality_metric,
                "quality": record.quality,
                "sustained": to_dict(record.sustained),
                "parameters": compression.parameters,
                "flops": compression.flops,
                "counting_method": compression.counting_method,
                "counting_input_shape": compression.counting_input_shape,
                "counting_exclusions": compression.counting_exclusions,
                "latency": latency,
                "memory": to_dict(record.memory),
                "power_w": average_power,
                "power_measured": record.power_trace.metadata.measured
                if record.power_trace
                else None,
                "energy_j": energy.energy_j if energy else None,
                "energy_estimate_j": energy.energy_estimate_j if energy else None,
                "per_inference_j": energy.per_inference_j if energy else None,
                "per_inference_estimate_j": energy.per_inference_estimate_j if energy else None,
                "above_idle_j": (
                    energy.energy_j
                    - energy.idle_power_w * (energy.window_end_s - energy.window_start_s)
                    if energy and energy.energy_j is not None and energy.idle_power_w is not None
                    else None
                ),
                "above_idle_estimate_j": (
                    energy.energy_estimate_j
                    - energy.idle_power_w * (energy.window_end_s - energy.window_start_s)
                    if energy
                    and energy.energy_estimate_j is not None
                    and energy.idle_power_w is not None
                    else None
                ),
                "energy_method": (
                    (energy.integration_method or energy.estimate_method) if energy else None
                ),
                "sparsity_pattern": compression.sparsity_pattern,
                "kernel_support": compression.kernel_support,
                "sparsity_label": compression.sparsity_label(),
                "precision": compression.precision,
                "compression_method": compression.method,
                "compression_details": to_dict(compression.details),
                "model_file_bytes": record.memory.model_file_bytes,
                "baseline_id": record.config.baseline_id,
                "changed_components": record.config.changed_components,
                "settings_mismatch": to_dict(mismatch),
            }
        )
    return rows


def load_stored_comparison(
    store, run: Path, *, allow_mismatch: bool = False
) -> list[dict[str, Any]]:
    """Refuse partially stored groups even when one individual record completed."""
    record = store.read(run)
    data = json.loads((run / "comparison.json").read_text(encoding="utf-8"))
    identifiers = data.get("run_ids")
    if not isinstance(identifiers, list) or not identifiers or record.run_id not in identifiers:
        raise ValueError("comparison must reference its own run and peer identities")
    for identifier in identifiers:
        safe_name(identifier, "comparison run id")
    try:
        records = [store.read(run.parent / identifier) for identifier in identifiers]
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError("comparison group is not completely stored") from exc
    return comparison_rows(records, allow_mismatch=allow_mismatch)
