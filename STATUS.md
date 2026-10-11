# Status

## 2026-10-11 — Phase 1

**State: architecture reviewed and core implemented; awaiting user review before
Phase 2.**

### Authorization and architecture review

- The user requested both the architecture review and Phase 1 implementation in
  one session. Reviewed the proposal against the authoritative specification;
  no structural conflict was found. Recorded concrete core decisions in
  `docs/research_framework.md` before implementation.
- Kept controllers, perturbation transforms, real runtimes/profiling, hardware
  support, compression transformations and reference experiments in later phases.

### Done

- Added strict typed configs and metadata for models, compression, workloads,
  conditions and measurement protocol; finite JSON conversion and schema versioning.
- Added safe YAML loading, confined fragment references, explicit overrides,
  unknown-field/duplicate-key rejection and private-fragment classification.
- Added automatic CPU device observations, installed-package metadata and Git
  commit/dirty-state capture without importing heavy optional packages.
- Added runtime/power protocols and deterministic CPU-only mocks with explicit
  representative warm-up batches and synthetic power/timing provenance.
- Added raw timing/power schemas, bounded single-rail trapezoidal integration,
  total/window-average energy metadata and measured-versus-estimated enforcement.
- Added complete `RunRecord` validation and exclusive public/private result
  storage, failure/completion markers, standard artifacts and validated reads.
- Added comparison guards with visible mismatch overrides, nominal/sparsity
  descriptors and validation of declared changes against an available baseline.
- Added CPU/mock tests, mock YAML contract fixtures, core API usage documentation,
  updated setup instructions and installed local development dependencies in `.venv`.
- Prepared Phase 1 for a local Git commit at the user's request; Phase 2 remains
  outside this work session.

### Verification

- Python 3.10 CPU suite: 84 passed; 2 symlink tests skipped because Windows denied
  directory symlink creation. Tests include concurrent/failed writes, private
  routing, schema/config rejection, analytic energy and provenance checks.
- Ruff checks passed for `src` and `tests`; Git whitespace checks passed.
- All core modules imported in a subprocess that explicitly blocks PyTorch,
  ONNX/ONNX Runtime, TensorRT, CUDA, OpenCV, jtop, NumPy, matplotlib and PyYAML.
- Reinstalled the editable package with its declared PyYAML dependency;
  `pip check` passed and the documented core API example executed successfully.
- Specification unchanged; no weights/datasets downloaded or public measurement
  runs created. All test records are synthetic fixtures in temporary directories.

### Open questions

- Review Phase 1 before authorizing Phase 2.
- Choose a distribution license; none has been selected or implied.
- Select the public pretrained model family and workload for Phase 2.
- Capture installed JetPack/L4T before Phase 3 compatibility choices.

### Known limitations

- No real inference backend, profiling runner, reference experiment, rendered
  report/Pareto plot, controller or perturbation transformation yet.
- Mock power is an independently generated synthetic trace, not a measurement of
  time spent executing mock inference. No energy savings are claimed.
- Warm-up metadata and mock accounting are validated; actual timing exclusion and
  synchronization await the Phase 2 profiler.
- Unknown CPU-host hardware settings carry reasons; requested settings are not
  applied. Sensor loss is rejected during integration until a gap policy exists.
- Storage immutability is enforced by the API, not filesystem write protection.
  Source privacy classification remains the caller's responsibility for remote
  identifiers; known private paths/fragments are checked automatically.
- Baseline attribution is independently checked only when the referenced baseline
  is included in the comparison. The development version remains `0.0.0`.

### Next phase (after review)

Phase 2: PyTorch/ONNX Runtime CPU backends, actual timing/warm-up exclusion,
memory profiling, rendered reports/Pareto plots and the public reference pipeline.

## 2026-10-10 — Phase 0

**State: architecture proposal complete; awaiting review before Phase 1.**

### Done

- Read the development specification and preserved it unchanged.
- Proposed module responsibilities, interfaces, dependency plan, storage rules,
  measurement boundaries and staged verification in `docs/research_framework.md`.
- Created the `src/edge_ai` package skeleton and all specified subsystem directories.
- Added config, reference experiment, script, test and public-result scaffolding.
- Added package metadata, development tool settings, README and agent instructions.
- Added Git ignore/line-ending rules, including complete private-tree exclusion.
- Added draft profiling, scope and hardware setup documents with implementation
  status made explicit.
- Configured local `main` and `origin` pointing to `https://github.com/jaworekk85/EdgeAI.git`.

### Verification

- Parsed `pyproject.toml` and verified metadata and version consistency.
- Parsed Python source and imported all 15 package/subsystem modules on Python 3.10
  without installing PyTorch, CUDA, TensorRT or other runtime dependencies.
- Verified package discovery and the required source/config directory layout.
- Verified ignore rules for private configs/results, generated public runs, secrets
  and model artifacts; scaffold documentation and config placeholders remain tracked.
- Checked local documentation links and staged changes for whitespace errors.
- Initialized the local Git repository on `main`; the existing GitHub repository
  `jaworekk85/EdgeAI` was confirmed empty before preparing the initial commit.

### Open questions

- Choose a license before distributing/reusing under defined license terms.
- Review the architecture before authorizing Phase 1 implementation.
- Select the public reference workload in Phase 2.
- Capture the installed JetPack/L4T stack before Phase 3 package compatibility choices.

### Known limitations

- No runtime, configuration loader, record validator, measurement, reporting,
  controller, compression tooling or experiment implementation yet.
- No test suite yet. CPU/mock scientific-invariant tests begin in Phase 1.
- Heavy optional dependencies and runtime environment pins are intentionally deferred.
- The package version `0.0.0` denotes the scaffold, not a research-capable release.

### Next phase (after review)

Phase 1: typed configs/dataclasses, record schema/validation, immutable/private result
store, provenance capture, mock runtime/power, comparison validation and CPU tests.
