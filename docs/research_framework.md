# Code map and execution flow

This describes the current implementation. The
[specification](../EDGE_AI_AGENT_RESEARCH_SPEC.md) is authoritative; see
[status](../STATUS.md) for open work and [history](history.md) for design decisions.

## Where to change things

Paths below are relative to `src/edge_ai/`.

| Area | Files / responsibility |
| --- | --- |
| Configuration | `config.py`, `utils/serialization.py`: typed YAML/JSON, strict validation |
| Models and data | `models/`, `datasets/`: model identity/loading, image manifests and preprocessing |
| Export | `export/onnx.py`: exclusive ONNX export and FP32 output parity |
| Compression | `compression/recipe.py`, `spec.py`, `accounting.py`: recipes, evidence and nominal counts; `quantization.py`, `pruning.py`: transforms; `verification.py`: artifact/profile checks |
| CPU entry points | `reference.py`: original preparation, shared experiment construction, isolated workers and baseline attribution; `compression/reference.py`: compression preparation and shared prepared-matrix validation |
| Execution | `runtimes/`: small backend protocol, mock, PyTorch and ORT adapters |
| Measurement | `profiling/`: warm-up, timing scopes, RSS, sustained summaries; `power/`: synthetic sources, trace integration and replay |
| Device observations | `devices/`: host identity and offline tegrastats parser |
| Quality and output | `evaluation/`, `reporting/`: classification metrics, guarded comparisons, tables and plots; `records/`: schemas and immutable storage |
| Future work | `controllers/` is a placeholder; `conditions/` has metadata only, no perturbation transforms |

Outside the library: `scripts/diagnose_int8.py` contains the bounded numerical
diagnosis, `configs/` the YAML inputs, and `tests/` invariant checks and CPU fixtures.
Weights/data/prepared artifacts live in ignored `artifacts/` and `data/`.
Generated public runs use `results/`; private inputs/adapters/results use `private/`.

## Ordinary and compression runs

```text
config + verified prepared assets
  -> resolve ExperimentConfig for each model/runtime
  -> fresh worker process captures provenance and host
  -> runtime load -> warm-up -> idle baseline -> timed inference
  -> quality + validated RunRecord
  -> baseline attribution -> prepare group reports
  -> exclusive ResultStore write -> complete.json last
```

Preparation/download/export is separate from measurement. Both CPU CLIs share
`reference_experiment`, `run_worker` and `with_baseline`; they differ in model
selection and comparison grouping. Compression benchmarking and diagnosis share
`load_compression_prepared`, which verifies source identity, config, manifest,
recipe coverage, lineage and kernel evidence before executing work.

Runtime implementations expose load, native batch preparation, representative-batch
binding, warm-up, synchronization, inference and release. Optional libraries are
imported inside the relevant operations. Training, hardware collection and model
selection do not belong inside the profiling loop.

## Contracts to preserve

- Typed metadata belongs to its subsystem. Prefer explicit functions/dataclasses
  to registries, inheritance layers or generic pipeline frameworks.
- Canonical pretrained checkpoint identity is separate from derived deployment
  artifact hashes. A runtime/precision change must retain its correct attribution.
- Record schemas 1–4 remain readable; 2 adds real CPU provenance, 3 sustained runs,
  4 transformed-model evidence. Never rewrite old runs or append reports to them.
- Device observations are captured, never inferred from requested settings.
  Unknown values remain unavailable with reasons.
- No measured energy without measured power. Replay and CPU mock watts always
  produce estimates. Detailed timing/energy semantics live in the
  [profiling protocol](profiling_protocol.md).
- Public paths are confined; private lineage cannot enter public output. Validate
  before reserving output and preserve incomplete runs on failure.

Current APIs and commands are in the [documentation index](README.md).
