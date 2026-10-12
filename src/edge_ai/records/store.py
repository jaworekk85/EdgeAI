"""Exclusive result directories and private routing. Existing runs are immutable."""

from __future__ import annotations

import hashlib
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path

from edge_ai.records.schema import RunRecord
from edge_ai.utils.serialization import from_dict, safe_name, to_dict


def new_run_id(config: object) -> str:
    encoded = json.dumps(to_dict(config), sort_keys=True, allow_nan=False).encode()
    digest = hashlib.sha256(encoded + secrets.token_bytes(16)).hexdigest()[:10]
    return f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_{digest}"


def _json(path: Path, data: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(to_dict(data), stream, indent=2, allow_nan=False)
        stream.write("\n")


class ResultStore:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve(strict=True)

    def _root(self, privacy: str) -> Path:
        public = (self.workspace / "results").resolve()
        private = (self.workspace / "private").resolve()
        private_results = (self.workspace / "private" / "results").resolve()
        if not all(
            path.is_relative_to(self.workspace)
            for path in (
                public,
                private,
                private_results,
            )
        ):
            raise ValueError("result root escapes workspace")
        if public.is_relative_to(private) or private.is_relative_to(public):
            raise ValueError("public and private result trees must not overlap")
        if not private_results.is_relative_to(private):
            raise ValueError("private result root escapes private/")
        return private_results if privacy == "private" else public

    def write(self, record: RunRecord, *, reports: dict[str, bytes] | None = None) -> Path:
        import yaml

        record.validate()
        reports = reports or {}
        allowed_reports = {
            "report.md",
            "report.csv",
            "comparison.json",
            "plots/quality_vs_latency.png",
            "plots/quality_vs_energy.png",
        }
        if set(reports) - allowed_reports or any(
            type(value) is not bytes for value in reports.values()
        ):
            raise ValueError("unsupported report artifact name or content")
        if sum(len(value) for value in reports.values()) > 50 * 1024 * 1024:
            raise ValueError("report artifacts exceed size limit")
        if record.schema_version in (2, 3) and set(reports) != allowed_reports:
            raise ValueError("schema version 2/3 requires complete reports before storage")
        safe_name(record.config.experiment, "experiment")
        root = self._root(record.config.workload.privacy)
        private_root = (self.workspace / "private").resolve()
        sources = [record.config.workload.source_id]
        if record.config.artifact is not None:
            sources.append(record.config.artifact.path)
        if record.dataset_manifest is not None:
            sources.extend(item["path"] for item in record.dataset_manifest["samples"])
        if record.config.workload.privacy != "private" and any(
            (self.workspace / source).resolve().is_relative_to(private_root) for source in sources
        ):
            raise ValueError("private source cannot be written to public results")
        parent = root / record.config.experiment
        parent.mkdir(parents=True, exist_ok=True)
        if not parent.resolve().is_relative_to(root):
            raise ValueError("experiment directory escapes the result root")
        run = parent / record.run_id
        run.mkdir(exist_ok=False)
        _json(
            run / "incomplete.json",
            {"status": "incomplete", "schema_version": record.schema_version},
        )
        # Keep the reservation and incomplete marker on failure; never retry into it.
        (run / "traces").mkdir()
        (run / "plots").mkdir()
        with (run / "config.yaml").open("x", encoding="utf-8", newline="\n") as stream:
            yaml.safe_dump(to_dict(record.config), stream, sort_keys=True)
        _json(run / "device.json", record.device)
        _json(run / "traces" / "timings.json", record.timings)
        if record.end_to_end_timings is not None:
            _json(run / "traces" / "end_to_end_timings.json", record.end_to_end_timings)
        if record.power_trace is not None:
            _json(run / "traces" / "power.json", record.power_trace)
        if record.idle_power_trace is not None:
            _json(run / "traces" / "idle_power.json", record.idle_power_trace)
        if record.sustained is not None:
            _json(run / "traces" / "sustained.json", record.sustained)
        if record.memory.rss_bytes:
            _json(run / "traces" / "memory.json", record.memory)
        _json(run / "metrics.json", record)
        for relative, content in reports.items():
            with (run / relative).open("xb") as stream:
                stream.write(content)
        _json(
            run / "complete.json", {"status": "complete", "schema_version": record.schema_version}
        )
        # Both markers remain: complete.json supersedes the reservation marker.
        return run

    def read(self, run: Path) -> RunRecord:
        run = run.resolve(strict=True)
        allowed = [self._root("public"), self._root("private")]
        if not any(run.is_relative_to(root) for root in allowed):
            raise ValueError("run is outside this result store")
        if not (run / "complete.json").is_file():
            raise ValueError("incomplete run cannot be read as a completed result")
        with (run / "complete.json").open(encoding="utf-8") as stream:
            marker = json.load(stream)
        if (
            not isinstance(marker, dict)
            or type(marker.get("schema_version")) is not int
            or marker
            not in [{"status": "complete", "schema_version": version} for version in (1, 2, 3)]
        ):
            raise ValueError("invalid completion marker")
        with (run / "metrics.json").open(encoding="utf-8") as stream:
            record = from_dict(RunRecord, json.load(stream))
        if marker["schema_version"] != record.schema_version:
            raise ValueError("completion marker version differs from record")
        if run.name != record.run_id or run.parent.name != record.config.experiment:
            raise ValueError("record identity differs from its directory")
        expected = self._root(record.config.workload.privacy)
        if not run.is_relative_to(expected):
            raise ValueError("stored privacy classification differs from its result tree")
        return record
