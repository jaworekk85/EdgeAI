import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

import edge_ai.records.store as store_module
from edge_ai.records.store import ResultStore, new_run_id


def test_store_writes_standard_outputs_and_preserves_record(tmp_path, record):
    store = ResultStore(tmp_path)
    run = store.write(record)
    assert run.parent.parent == tmp_path / "results"
    for name in ("config.yaml", "device.json", "metrics.json", "traces", "plots", "complete.json"):
        assert (run / name).exists()
    assert (run / "traces" / "timings.json").is_file()
    assert (run / "traces" / "power.json").is_file()
    assert store.read(run) == record
    before = (run / "metrics.json").read_bytes()
    with pytest.raises(FileExistsError):
        store.write(record)
    assert (run / "metrics.json").read_bytes() == before


def test_concurrent_writers_only_one_can_reserve(tmp_path, record):
    store = ResultStore(tmp_path)

    def attempt():
        try:
            return store.write(record)
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = list(pool.map(lambda _: attempt(), range(2)))
    assert sum(path is not None for path in paths) == 1
    assert store.read(next(path for path in paths if path is not None)) == record


def test_failed_write_remains_incomplete_and_cannot_be_reused(tmp_path, record, monkeypatch):
    original = store_module._json

    def fail_metrics(path, data):
        if path.name == "metrics.json":
            raise OSError("simulated interrupted write")
        original(path, data)

    monkeypatch.setattr(store_module, "_json", fail_metrics)
    store = ResultStore(tmp_path)
    with pytest.raises(OSError, match="interrupted"):
        store.write(record)
    run = tmp_path / "results" / record.config.experiment / record.run_id
    assert (run / "incomplete.json").exists()
    assert not (run / "complete.json").exists()
    with pytest.raises(ValueError, match="incomplete"):
        store.read(run)
    with pytest.raises(FileExistsError):
        store.write(record)


def test_private_runs_never_create_public_result_tree(tmp_path, record):
    private = replace(record, config=replace(
        record.config, workload=replace(record.config.workload, privacy="private"),
    ))
    store = ResultStore(tmp_path)
    run = store.write(private)
    assert run.is_relative_to(tmp_path / "private" / "results")
    assert not (tmp_path / "results").exists()
    assert store.read(run) == private


def test_validation_precedes_directory_reservation(tmp_path, record):
    record.provenance.runtime_versions["mock"] = None
    with pytest.raises(ValueError):
        ResultStore(tmp_path).write(record)
    assert not (tmp_path / "results").exists()


def test_direct_private_source_cannot_leak_into_public_tree(tmp_path, record):
    bad = replace(record, config=replace(record.config, workload=replace(
        record.config.workload, source_id="private/data/input",
    )))
    with pytest.raises(ValueError, match="private source"):
        ResultStore(tmp_path).write(bad)
    assert not (tmp_path / "results").exists()


def test_read_rejects_wrong_identity_and_invalid_marker(tmp_path, record):
    store = ResultStore(tmp_path)
    run = store.write(record)
    (run / "complete.json").write_text('{"status": "incomplete"}', encoding="utf-8")
    with pytest.raises(ValueError, match="completion marker"):
        store.read(run)
    (run / "complete.json").write_text(
        json.dumps({"status": "complete", "schema_version": 1}), encoding="utf-8",
    )
    data = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    data["run_id"] = "other-id"
    (run / "metrics.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="identity differs"):
        store.read(run)


def test_new_ids_are_unique_and_encode_utc(config):
    first, second = new_run_id(config), new_run_id(config)
    assert first != second
    assert "Z_" in first and len(first.split("_")[1]) == 10


def link_or_skip(link, target):
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable on this host: {exc}")


def test_private_result_symlink_cannot_redirect_to_public(tmp_path, record):
    (tmp_path / "private").mkdir()
    (tmp_path / "results").mkdir()
    link_or_skip(tmp_path / "private" / "results", tmp_path / "results")
    private = replace(record, config=replace(
        record.config, workload=replace(record.config.workload, privacy="private"),
    ))
    with pytest.raises(ValueError, match="escapes private"):
        ResultStore(tmp_path).write(private)
    assert not list((tmp_path / "results").iterdir())


def test_result_root_symlink_cannot_escape_workspace(tmp_path, record):
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    link_or_skip(workspace / "results", outside)
    with pytest.raises(ValueError, match="escapes workspace"):
        ResultStore(workspace).write(record)
    assert not list(outside.iterdir())
