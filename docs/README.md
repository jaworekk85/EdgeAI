# Documentation

Start with [current status](../STATUS.md) and the [code map](research_framework.md).
The [specification](../EDGE_AI_AGENT_RESEARCH_SPEC.md) defines mandatory scope,
result invariants and phase gates; other documents describe their implementation.

| Task | Read |
| --- | --- |
| Install CPU dependencies, prepare models, run the reference | [CPU reference](cpu_reference.md) |
| Use configs, mocks, records and comparisons from Python | [Core API](core_usage.md) |
| Run longer CPU measurements or replay telemetry | [Sustained CPU and replay](phase_3a_usage.md) |
| Prepare/run FP16, INT8, INT4 and pruning masks | [CPU compression](phase_4_usage.md) |
| Reproduce the INT8 quality diagnosis | [INT8 findings and command](int8_diagnosis.md) |
| Understand timing, memory, power and energy boundaries | [Profiling protocol](profiling_protocol.md) |
| Plan device validation and later CPU work | [CPU/hardware roadmap](cpu_hardware_roadmap.md), [Jetson acceptance](hardware_setup.md) |
| Review approvals, design choices and past verification | [History](history.md) |
| Check allowed workloads | [Scope and boundaries](scope_and_boundaries.md) |

Use [test instructions](../tests/README.md) for offline/opt-in checks.
[Configs](../configs/README.md), [scripts](../scripts/README.md) and
[reference validation](../experiments/reference/README.md) have short local indexes.
Completed plans are summarized in history; their full original text remains linked
at fixed Git revisions. Keep current instructions here and append decisions to
history rather than growing the status page into a session log.
