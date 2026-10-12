"""Headless, provenance-aware reports prepared before immutable storage."""

from __future__ import annotations

import csv
import io
import json
import math

from edge_ai.records.schema import RunRecord
from edge_ai.reporting.comparison import comparison_rows
from edge_ai.utils.serialization import to_dict


def pareto_indices(costs: list[float], qualities: list[float]) -> list[int]:
    if len(costs) != len(qualities) or not all(math.isfinite(x) for x in costs + qualities):
        raise ValueError("Pareto coordinates must be equal-length finite arrays")
    return [
        index
        for index, (cost, quality) in enumerate(zip(costs, qualities))
        if not any(
            other_cost <= cost
            and other_quality >= quality
            and (other_cost < cost or other_quality > quality)
            for other_cost, other_quality in zip(costs, qualities)
        )
    ]


def render_reports(records: list[RunRecord], *, allow_mismatch: bool = False) -> dict[str, bytes]:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    rows = comparison_rows(records, allow_mismatch=allow_mismatch)
    mismatch = rows[0]["settings_mismatch"]
    mismatch_text = json.dumps(mismatch, sort_keys=True) if mismatch else "none"
    columns = [
        "variant",
        "runtime",
        "quality",
        "latency_mean_ms",
        "latency_p95_ms",
        "ram_peak_bytes",
        "memory_method",
        "power_w",
        "power_measured",
        "energy_j",
        "energy_estimate_j",
        "per_inference_j",
        "per_inference_estimate_j",
        "above_idle_j",
        "above_idle_estimate_j",
        "energy_method",
        "parameters",
        "flops",
        "counting_method",
        "counting_exclusions",
        "sparsity_pattern",
        "kernel_support",
        "sparsity_label",
        "settings_mismatch",
    ]
    flat_rows = []
    for record, row in zip(records, rows):
        flat = {key: row.get(key) for key in columns}
        flat.update(
            {
                "runtime": record.config.model.runtime,
                "latency_mean_ms": row["latency"]["mean_s"] * 1000,
                "latency_p95_ms": row["latency"]["p95_s"] * 1000,
                "ram_peak_bytes": row["memory"]["peak_ram_bytes"],
                "memory_method": row["memory"]["method"],
                "counting_exclusions": ", ".join(row["counting_exclusions"]),
                "settings_mismatch": mismatch_text,
            }
        )
        flat_rows.append(flat)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    writer.writerows(flat_rows)
    title = "Pipeline validation; synthetic power/energy estimates are not device measurements"
    summary_columns = [
        "Variant",
        "Runtime",
        "Top-1",
        "Latency mean / p95 (ms)",
        "Sampled peak RSS (MiB)",
        "Power (W)",
        "Energy / inference (J)",
        "Parameters",
        "Partial FLOPs",
        "Sparsity / kernels",
    ]
    markdown = [
        f"# {title}",
        "",
        f"Settings mismatch: {mismatch_text}",
        "",
        "| " + " | ".join(summary_columns) + " |",
        "| " + " | ".join(["---"] * len(summary_columns)) + " |",
    ]

    def display(value, *, digits: int = 4):
        return "unavailable" if value is None else f"{value:.{digits}g}"

    for row in flat_rows:
        average = row["per_inference_j"]
        energy_label = "measured"
        if average is None:
            average, energy_label = row["per_inference_estimate_j"], "estimate"
        power_label = "measured" if row["power_measured"] else "estimate"
        values = [
            row["variant"],
            row["runtime"],
            display(row["quality"]),
            f"{display(row['latency_mean_ms'])} / {display(row['latency_p95_ms'])}",
            display(row["ram_peak_bytes"] / 2**20 if row["ram_peak_bytes"] is not None else None),
            f"{display(row['power_w'])} ({power_label})",
            f"{display(average)} ({energy_label})",
            display(row["parameters"], digits=8),
            display(row["flops"], digits=8),
            row["sparsity_label"],
        ]
        markdown.append("| " + " | ".join(str(value).replace("|", "/") for value in values) + " |")
    markdown.extend(
        [
            "",
            "Full metrics and provenance fields are in report.csv and metrics.json.",
            "RSS includes libraries, inputs and runtime; sampling can miss short peaks.",
            "",
        ]
    )
    for row in flat_rows:
        markdown.append(
            f"- {row['variant']}: {row['counting_method']}; excluded: "
            f"{row['counting_exclusions']}. Memory: {row['memory_method']}. "
            f"Energy method: {row['energy_method']}."
        )
    reports = {
        "report.csv": output.getvalue().encode(),
        "report.md": ("\n".join(markdown) + "\n").encode(),
        "comparison.json": json.dumps(
            {
                "rows": to_dict(rows),
                "run_ids": [record.run_id for record in records],
                "settings_mismatch": mismatch,
                "purpose": "pipeline_validation",
            },
            indent=2,
        ).encode(),
    }
    for kind in ("latency", "energy"):
        figure = Figure(figsize=(9, 6))
        canvas = FigureCanvasAgg(figure)
        axis = figure.add_subplot(111)
        points = []
        for record, row in zip(records, rows):
            if kind == "latency":
                cost, label = row["latency"]["mean_s"] * 1000, "measured latency (ms)"
                if not row["latency"]["measured"]:
                    label = "synthetic latency (ms)"
            else:
                cost = row["per_inference_j"]
                label = "measured energy (J / inference)"
                if cost is None:
                    cost = row["per_inference_estimate_j"]
                    label = "synthetic energy estimate (J / inference)"
            if cost is None or row["quality"] is None:
                continue
            points.append((cost, row["quality"], label))
            axis.scatter(
                cost, row["quality"], label=f"{row['variant']} / {record.config.model.runtime}"
            )
        if points:
            # Separate measured/estimated coordinates into independent fronts.
            for provenance in sorted({point[2] for point in points}):
                subset = [(x, y) for x, y, label in points if label == provenance]
                front = pareto_indices([x for x, _ in subset], [y for _, y in subset])
                ordered = sorted(subset[index] for index in front)
                axis.plot([x for x, _ in ordered], [y for _, y in ordered], linestyle="--")
            axis.set_xlabel(" / ".join(sorted({point[2] for point in points})))
            axis.legend()
        else:
            axis.text(
                0.5, 0.5, "No compatible available metrics", ha="center", transform=axis.transAxes
            )
        axis.set_ylabel("top-1 accuracy (full ImageNet-1K output)")
        axis.set_ylim(0, 1.05)
        axis.set_title("Pipeline validation — SYNTHETIC ENERGY, no measured energy savings")
        if mismatch:
            figure.text(0.02, 0.01, "Settings mismatch: " + mismatch_text, fontsize=6, wrap=True)
        figure.tight_layout(rect=(0, 0.15 if mismatch else 0, 1, 1))
        png = io.BytesIO()
        canvas.print_png(
            png, metadata={"Description": title + "; Settings mismatch: " + mismatch_text}
        )
        reports[f"plots/quality_vs_{kind}.png"] = png.getvalue()
        figure.clear()
    return reports
