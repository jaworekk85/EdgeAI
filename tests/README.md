# Test plan

Phase 0 contains no test suite. See `docs/research_framework.md` for the staged
verification plan and specification section 11 for mandatory scientific tests.

Phase 1 tests run on CPU with deterministic mocks and enforce record validation,
energy provenance, immutable/private storage, trace integration and comparable
settings. Later phases add warm-up exclusion, the CPU reference smoke run, controller
contract checks and deterministic perturbations. Hardware-dependent tests use the
configured `jetson` marker and must be skipped on unsupported hosts.
