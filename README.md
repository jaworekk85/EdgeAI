# EdgeAI

Research framework for measuring resource-efficient AI on embedded hardware.
First target: NVIDIA Jetson Orin Nano Super, with a hardware-independent core
and CPU/mock paths for development on a laptop.

**Current stage: Phase 1 — core implemented; awaiting review before Phase 2.**
Typed configurations, validated records, immutable result storage, CPU provenance
capture and deterministic runtime/power mocks are implemented. Real inference,
profiling runners and the reference experiment belong to subsequent phases.

## Start here

- [Development specification](EDGE_AI_AGENT_RESEARCH_SPEC.md): scope and mandatory rules.
- [Proposed architecture](docs/research_framework.md): modules, interfaces and dependencies.
- [Status](STATUS.md): completed work, questions and phase boundary.
- [Core API usage](docs/core_usage.md): configuration, mocks and result contracts.
- [Profiling protocol draft](docs/profiling_protocol.md).
- [Hardware setup plan](docs/hardware_setup.md).
- [Scope and boundaries](docs/scope_and_boundaries.md).

## Local setup

Python 3.10 or newer. The core uses the standard library and PyYAML.

```sh
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell,
or `source .venv/bin/activate` on Linux/macOS. Then:

```sh
python -m pip install -e ".[dev]"
python -c "import edge_ai; print(edge_ai.__version__)"
```

The import prints the unreleased development version `0.0.0`. Run the CPU/mock
checks with:

```sh
python -m pytest -q
python -m ruff check src tests
```

No experiment command exists yet. Heavy hardware dependencies remain optional;
core imports do not load PyTorch, ONNX Runtime, CUDA, TensorRT or even PyYAML.
PyYAML is loaded only when reading/writing configuration or results.

## Repository layout

```text
src/edge_ai/   Core package and dedicated subsystem packages
configs/       Typed YAML fragments and a mock contract fixture
experiments/   Reference pipeline validation only, until research is requested
scripts/       Future entry points, including on-device tools
docs/          Architecture, measurement protocol and setup guidance
tests/         CPU/mock scientific-invariant tests
results/       Immutable public run directories (generated runs ignored)
private/       Local partner data/configs/adapters/results; entirely git-ignored
STATUS.md      Work-session log and pending review
```

All reported quantities must distinguish measurements from estimates.
Mock power must never produce a measured `energy_j` value. Private data must
never enter public result directories. See the specification for the complete
invariants and phase plan.

## Development

Phase 0 was reviewed and Phase 1 authorized on 2026-10-11. Review Phase 1 before
starting Phase 2. Read the specification before substantial changes and update
`STATUS.md` after each work session.

The remote repository is <https://github.com/jaworekk85/EdgeAI>.
No distribution license has been chosen yet.
