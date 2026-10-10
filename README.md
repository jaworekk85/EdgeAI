# EdgeAI

Research framework for measuring resource-efficient AI on embedded hardware.
First target: NVIDIA Jetson Orin Nano Super, with a hardware-independent core
and CPU/mock paths for development on a laptop.

**Current stage: Phase 0 — architecture proposal and repository skeleton.**
Runtime backends, measurement routines and experiments are not implemented yet.

## Start here

- [Development specification](EDGE_AI_AGENT_RESEARCH_SPEC.md): scope and mandatory rules.
- [Proposed architecture](docs/research_framework.md): modules, interfaces and dependencies.
- [Status](STATUS.md): completed work, questions and phase boundary.
- [Profiling protocol draft](docs/profiling_protocol.md).
- [Hardware setup plan](docs/hardware_setup.md).
- [Scope and boundaries](docs/scope_and_boundaries.md).

## Local setup

Python 3.10 or newer. The skeleton has no runtime dependencies.

```sh
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell,
or `source .venv/bin/activate` on Linux/macOS. Then:

```sh
python -m pip install -e ".[dev]"
python -c "import edge_ai; print(edge_ai.__version__)"
```

The import prints `0.0.0`. There is no experiment command or test suite yet;
these are scheduled for subsequent phases. Heavy hardware dependencies will
be optional and loaded only by their dedicated backends.

## Repository layout

```text
src/edge_ai/   Core package and dedicated subsystem packages
configs/       Future typed YAML configuration inputs
experiments/   Reference pipeline validation only, until research is requested
scripts/       Future entry points, including on-device tools
docs/          Architecture, measurement protocol and setup guidance
tests/         CPU/mock tests planned for Phase 1 onward
results/       Future immutable public run directories (generated runs ignored)
private/       Local partner data/configs/adapters/results; entirely git-ignored
STATUS.md      Work-session log and pending review
```

All reported quantities must distinguish measurements from estimates.
Mock power must never produce a measured `energy_j` value. Private data must
never enter public result directories. See the specification for the complete
invariants and phase plan.

## Development

Review Phase 0 before starting Phase 1. Read the specification before substantial
changes and update `STATUS.md` after each work session.

The remote repository is <https://github.com/jaworekk85/EdgeAI>.
No distribution license has been chosen yet.
