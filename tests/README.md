# CPU/mock tests

Run `python -m pytest tests -q` and `python -m ruff check src tests scripts` from the repository
after installing `.[dev]`. Tests need no Jetson or accelerator libraries. Default
tests use generated fixtures in temporary directories; the opt-in public smoke
uses separately prepared pretrained assets and writes new ignored result directories.

Coverage includes strict config/schema validation, energy provenance, analytic
trace integration and bounded windows, concurrent directory reservation, failed
writes, private output routing, comparison mismatch refusal/visible overrides,
baseline attribution, mock determinism/lifecycle, automatic CPU/Git metadata and
imports with optional dependencies explicitly blocked. Symlink confinement tests
skip when the operating system denies creation of directory symlinks.

Measured-schema fixtures are generated test data, not physical sensor readings.
Phase 1 checks declared warm-up exclusion and mock call accounting. Phase 2 adds
fake-clock profiling/cleanup tests, offline PyTorch/ONNX fixtures, schema-v1/v2
compatibility, deterministic checked manifests, full-class label mapping and
report/group validation. Controller
contracts in Phase 5 and deterministic perturbations in Phase 6. Hardware tests
will use the `jetson` marker and skip on unsupported hosts.

For the actual public pretrained smoke, prepare assets separately, then run:

```sh
python -m pytest tests -q --reference-prepared <prepared.json path>
```

This opt-in test checks the full four-worker CLI, immutable standard outputs and
the approximately two-minute budget. It downloads nothing; without the option it
skips explicitly. Real adapter fixtures skip when the optional CPU/export extras
are absent. Phase 2 acceptance requires running both the offline suite and this
prepared public smoke; a skip alone is not evidence of pipeline completion.

Phase 3A extends the opt-in smoke with a two-second sustained matrix using the same
assets and both CPU backends. It checks schema 3, actual counts, complete passes,
stop reason, estimated energy and reports. Parser/replay fixtures are hand-written
format examples, not hardware data. Fake-clock tests exercise duration/cap stops
and cleanup; sustained record validation recomputes summaries from raw traces.

Phase 4 adds independent mask/axis/tie checks, generated FP16/INT8/INT4 CPU
transforms, calibration isolation, actual operator profile evidence, immutable
compression records, private-lineage routing and compression-only attribution.
`--compression-prepared <prepared.json>` enables the separately prepared public
eight-run matrix; it creates fresh ignored results and performs no downloads.
