import subprocess
import sys
from pathlib import Path
from time import perf_counter

import pytest

from edge_ai.records.store import ResultStore
from edge_ai.reporting.comparison import load_stored_comparison


@pytest.mark.reference
def test_prepared_public_reference_full_cli_under_two_minutes(request):
    prepared = request.config.getoption("--reference-prepared")
    if prepared is None:
        pytest.skip("opt in with --reference-prepared after separate asset preparation")
    workspace = Path(__file__).resolve().parents[1]
    started = perf_counter()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "edge_ai.reference",
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
    elapsed = perf_counter() - started
    assert completed.returncode == 0, completed.stderr
    paths = [
        Path(line)
        for line in completed.stdout.splitlines()
        if line.startswith(str(workspace / "results"))
    ]
    assert len(paths) == 4 and len(set(paths)) == 4
    store = ResultStore(workspace)
    records = [store.read(path) for path in paths]
    assert {
        (record.config.model.architecture, record.config.model.runtime) for record in records
    } == {
        (architecture, backend)
        for architecture in ("mobilenet_v3_small", "mobilenet_v3_large")
        for backend in ("pytorch", "onnxruntime")
    }
    for path, record in zip(paths, records):
        assert len(record.timings.durations_s) == 20
        assert record.timings.warmup_iterations_excluded == 5
        assert record.energy.energy_j is None and record.energy.energy_estimate_j is not None
        assert record.timings.measured and len(load_stored_comparison(store, path)) == 2
    assert elapsed < 120
    print(f"Full public reference CLI: {elapsed:.3f}s; four complete runs")
