# Entry points

The shared CPU reference CLI is `python -m edge_ai.reference`; Phase 3A adds
sustained overrides documented in [3A usage](../docs/phase_3a_usage.md).
The `on_device` directory is reserved for Phase 3B Jetson entry points. These should
call library interfaces rather than duplicate profiling or result-storage logic.
