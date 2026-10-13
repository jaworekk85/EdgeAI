import subprocess
import sys
from pathlib import Path
from time import perf_counter

import pytest

from edge_ai.records.store import ResultStore
from edge_ai.reporting.comparison import load_stored_comparison


@pytest.mark.reference
def test_prepared_public_compression_matrix(request):
    prepared = request.config.getoption("--compression-prepared")
    if prepared is None:
        pytest.skip("opt in with --compression-prepared after separate compression preparation")
    workspace = Path(__file__).resolve().parents[1]
    started = perf_counter()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "edge_ai.compression.reference",
            "run",
            "--workspace",
            str(workspace),
            "--prepared",
            prepared,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    paths = [
        Path(line)
        for line in completed.stdout.splitlines()
        if line.startswith(str(workspace / "results"))
    ]
    assert len(paths) == 8 and len(set(paths)) == 8
    store = ResultStore(workspace)
    records = [store.read(path) for path in paths]
    compressed = [record for record in records if record.schema_version == 4]
    assert {record.config.model.compression.details.recipe.method for record in compressed} == {
        "fp16",
        "int8",
        "int4",
        "unstructured",
        "channel",
        "2:4",
    }
    for path, record in zip(paths, records):
        assert len(record.timings.durations_s) == 20 and record.energy.energy_j is None
        assert record.memory.model_file_bytes > 0 and record.memory.peak_ram_bytes > 0
        peers = load_stored_comparison(store, path)
        assert len(peers) == (2 if record.config.model.runtime == "pytorch" else 6)
        if record.schema_version == 4:
            assert record.config.changed_components == ("compression",)
            assert (path / "compression.json").is_file()
            if record.config.model.precision in ("int8", "int4"):
                assert (path / "traces/compression_kernel_profile.json").is_file()
    assert perf_counter() - started < 120
