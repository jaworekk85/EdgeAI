# EdgeAI

Research framework for measuring resource-efficient AI on embedded hardware.
Develop on a laptop with CPU/mock backends; the first hardware target is
NVIDIA Jetson Orin Nano Super.

CPU inference, compression tools and immutable reports are implemented.
Jetson support is pending. See [current status](STATUS.md) for limitations and
the next review, and the [documentation index](docs/README.md) for task guides.

## Local setup

Python 3.10 or newer; the core uses the standard library and PyYAML.

```sh
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` on Linux/macOS, then:

```sh
python -m pip install -e ".[dev]"
python -m pytest tests -q
python -m ruff check src tests scripts
```

Model execution needs the optional dependencies documented in
[CPU setup and reference commands](docs/cpu_reference.md).
Core imports do not load inference, hardware or YAML libraries.

## Find your way

- [Możliwości repo i pomysły eksperymentów — po polsku](docs/mozliwosci_i_eksperymenty.md).
- [Code map and execution flow](docs/research_framework.md).
- [Documentation by task](docs/README.md), including compression and diagnostics.
- [Development decisions and verification history](docs/history.md).
- [Authoritative specification](EDGE_AI_AGENT_RESEARCH_SPEC.md): scope, invariants
  and phase review rules. Read before substantial changes.

Generated data, weights, artifacts and public results are ignored. Partner data,
configs and results belong in ignored `private/`. CPU mock power produces
estimated energy only; CPU timings do not predict Jetson performance.
The small public reference validates the pipeline, not research conclusions.

Development version: `0.0.0`. No distribution license has been selected.
