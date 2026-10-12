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
