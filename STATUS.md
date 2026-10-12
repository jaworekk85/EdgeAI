# Status

## 2026-10-12 — Phase 2

**State: approved CPU/reference plan implemented and verified; ready for user
review before Phase 3.**

### Authorization and scope

- The user explicitly approved `docs/phase_2_plan.md` and requested continuation.
- Implemented CPU runtimes, profiling, reporting and the reference pipeline only.
  Hardware support, compression transforms, controllers and condition experiments
  remain in their later phases; the authoritative specification is unchanged.

### Done

- Added lazy PyTorch/ONNX Runtime CPU adapters with native input preparation,
  explicit thread/provider settings, warm-up, synchronization contract and release.
- Added offline verified model loading, exclusive ONNX export, export identities,
  shared partial Conv2d/Linear FLOP accounting and full parameter counts.
- Added typed reference config, deterministic stratified Imagenette selection,
  checked ImageNet-1K labels, shared Pillow/NumPy preprocessing and image hashes.
- Added separate bounded/checksummed public preparation and no-download benchmark
  commands; matching caches are verified and failed preparations never overwritten.
- Added serial model/end-to-end timing, correct scoped throughput, sampled native
  process RSS, observed-window synthetic active/idle power and energy estimates.
- Added schema version 2 for execution/data/artifact/RSS/idle provenance while
  preserving reading of version 1 files without rewriting them.
- Added concise Markdown/full CSV tables and headless Pareto plots, visible
  mismatch overrides, semantic model/runtime baseline attribution, private-path
  checks and comparison-group completeness validation.
- Reports are prepared before exclusive result storage and written before the
  completion marker; no completed run directory is extended or overwritten.
- Added offline adapter, profiler/cleanup, manifest/extraction, report/privacy,
  compatibility and opt-in full public smoke tests; documented usage and versions.

### Verification

- Full CPU suite with prepared public smoke: **112 passed, 2 skipped** (Windows
  denied directory symlink creation). No hardware was required.
- Core-only emulation blocking optional inference/export/image/memory/plot imports:
  **99 passed, 15 skipped** (optional CPU fixtures, opt-in smoke and symlink tests).
- The first four-worker public CLI completed in **16.795 seconds**, including
  startup, loading, preprocessing, warm-up, inference, reporting and storage.
  The opt-in test independently passed the approximately two-minute budget and
  validated all four complete records/groups.
- Both exported models passed full-logit checks and identical top-1 predictions
  on all 20 selected images. Recorded tolerance: rtol=1e-4, atol=5e-5; actual max
  absolute errors: Small 4.63e-5, Large 1.83e-5. The initial atol=1e-5 rejected
  one near-zero difference of 1.78e-5; the change is explicit in export metadata.
- Verified pinned PyTorch 2.9.1+cpu/Torchvision 0.24.1+cpu and ONNX exporter stack
  on Python 3.10.11 Windows; ONNX IR 1.0 incompatibility resolved with 0.1.12.
- Ruff, dependency checks, editable package installation, local documentation
  links and Git whitespace checks passed. Visually inspected a generated plot.
- Assets and generated immutable runs remain ignored; none will be committed.
  Validation runs captured Git dirty=True honestly while implementation was pending.

### Known limitations

- Real reference adapters currently cover FP32 MobileNetV3 classification, batch 1,
  a small Imagenette validation subset and CPU execution. No Jetson/thermal/sustained
  run, quantization, pruning, controller or perturbation implementation yet.
- Synthetic watts/energy validate plumbing only; no measured energy or savings.
  Download/export setup is outside the prepared benchmark's two-minute budget.
- FLOPs are partial Conv2d/Linear accounting with explicit exclusions; RSS is a
  sampled process peak including libraries/runtime and may miss transient peaks.
- Storage is immutable through the API, not filesystem write protection. Private
  remote-source classification remains the caller's responsibility.
- The exact tested CPU dependency stack is pinned; other OS/Python combinations
  are not claimed verified. Version remains unreleased development `0.0.0`.

### Open questions / next phase

- Review Phase 2 before authorizing Phase 3.
- Choose a repository distribution license; no asset redistribution is implied.
- Capture the Jetson's installed JetPack/L4T stack before hardware compatibility
  decisions, then propose Phase 3 device/power/thermal/TensorRT architecture.

## 2026-10-11 — Phase 2 planning

**State: Phase 1 accepted for the next planning step; Phase 2 proposal prepared
for review. Implementation has not started.**

### Done

- The user agreed to the proposed next step: prepare a Phase 2 plan before
  implementation. Re-read the specification, status, architecture and core APIs.
- Confirmed Phase 1 commit `a68189e` is on `main`/`origin/main`, with a clean
  working tree before preparing this proposal.
- Wrote `docs/phase_2_plan.md`: CPU runtime/export contracts, timing and RSS
  boundaries, provenance/schema evolution, immutable reporting and validation gates.
- Proposed MobileNetV3 Small/Large FP32 V1 weights with a seeded 20-image
  Imagenette validation subset and both PyTorch/ONNX Runtime CPU backends.
- Checked current primary documentation for weight preprocessing, dataset mapping,
  ONNX export and ORT CPU/session settings. Kept exact dependency pins pending
  compatibility checks during implementation.
- Identified two integration requirements: power estimates must cover the observed
  run window, and report files must be written before completion to preserve
  immutable results. Recorded solutions in the proposal.

### Verification

- Inspected implemented Phase 1 contracts against the proposed extension points.
- Confirmed local Python 3.10.11 and absence of all optional Phase 2 libraries.
- Checked documentation links and Git whitespace for the documentation change.
- No runtime code/config/test changes, dependency installations, asset downloads
  or benchmark executions in this planning session.

### Open questions / review

- Review the proposed model/data choice, timing boundaries, schema version 2 and
  prepared-asset smoke definition before implementation.
- Exact compatible Python 3.10 Windows CPU dependency versions remain to verify.
- The approximately two-minute smoke target is an acceptance requirement, not
  verified performance; first downloads/export are separate setup work.
- Repository distribution license and future JetPack/L4T observations remain open.

### Next step

Implement the reviewed Phase 2 proposal, run both offline CPU tests and the actual
prepared public reference smoke, then stop before Phase 3.

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
