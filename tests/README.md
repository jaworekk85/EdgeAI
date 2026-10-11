# CPU/mock tests

Run `python -m pytest -q` and `python -m ruff check src tests` from the repository
after installing `.[dev]`. Tests need no Jetson, accelerator libraries, pretrained
weights or datasets. All generated records use temporary directories.

Coverage includes strict config/schema validation, energy provenance, analytic
trace integration and bounded windows, concurrent directory reservation, failed
writes, private output routing, comparison mismatch refusal/visible overrides,
baseline attribution, mock determinism/lifecycle, automatic CPU/Git metadata and
imports with optional dependencies explicitly blocked. Symlink confinement tests
skip when the operating system denies creation of directory symlinks.

Measured-schema fixtures are generated test data, not physical sensor readings.
Phase 1 checks declared warm-up exclusion and mock call accounting. Actual profiler
warm-up exclusion and the CPU reference smoke run arrive in Phase 2; controller
contracts in Phase 5 and deterministic perturbations in Phase 6. Hardware tests
will use the `jetson` marker and skip on unsupported hosts.
