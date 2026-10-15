# Decisions and verification history

This is the compact history. [Status](../STATUS.md) holds current work and limits;
[the code map](research_framework.md) describes implemented responsibilities.
Original session logs and complete proposals remain in
[the snapshot before cleanup](https://github.com/jaworekk85/EdgeAI/tree/1d958ab).

## Approved milestones

| Date | Milestone | Commit | Verification at that boundary |
| --- | --- | --- | --- |
| 2026-10-10 | Phase 0: package skeleton and architecture only | [e8d9812](https://github.com/jaworekk85/EdgeAI/commit/e8d9812) | Syntax/imports, packaging and ignore rules; no implementation tests |
| 2026-10-11 | Phase 1: core contracts and scientific invariants | [a68189e](https://github.com/jaworekk85/EdgeAI/commit/a68189e) | 84 passed, 2 Windows symlink skips |
| 2026-10-11/12 | Phase 2: CPU reference and profiling | [d40f9df](https://github.com/jaworekk85/EdgeAI/commit/d40f9df) | 112 passed, 2 skips; core-only 99 passed, 15 skips |
| 2026-10-12 | Phase 3A: offline telemetry and sustained CPU | [a0e7546](https://github.com/jaworekk85/EdgeAI/commit/a0e7546) | 159 passed, 2 skips; core-only 143 passed, 18 skips |
| 2026-10-13 | Phase 4: CPU compression tooling | [f09fe64](https://github.com/jaworekk85/EdgeAI/commit/f09fe64) | 188 passed, 2 skips; core-only 152 passed, 38 skips |
| 2026-10-14 | Authorized INT8 quality diagnosis | [1d958ab](https://github.com/jaworekk85/EdgeAI/commit/1d958ab) | 191 passed, 2 skips, including 3 independent numerical diagnostic checks |

Core-only verification blocks optional inference/export/image/memory/plot/hardware
imports; skips reflect optional fixtures and opt-in public runs. Full boundary
checks included the relevant prepared public smoke. These counts describe their
Git revisions, not a promise about other environments.

## Decisions that still matter

**Core (Phase 1).** Use standard-library dataclasses, strict serialization and
explicit constructors. Reject unknown/nonfinite metadata. Capture Git commit and
dirty state, runtime versions and host observations automatically. Keep private
routing, measured/estimated energy, comparison guards and exclusive writes in code.
Write incomplete first and complete last; API immutability is not filesystem
write protection. No license was selected.

**CPU reference (Phase 2).** Use MobileNetV3 Small/Large, explicit ImageNet V1
weights and twenty seeded Imagenette validation images. Labels remain ImageNet-1K
indices, not a ten-class remapping. Prepare assets separately; run each config in
a fresh CPU worker with explicit settings. The first four-run CLI took 16.795 s,
including startup/reports/storage but excluding downloads/export. This validates
plumbing, not research accuracy or hardware efficiency.

FP32 export parity requires identical top-1 and `rtol=1e-4, atol=5e-5`. The initial
`atol=1e-5` rejected a near-zero 1.78e-5 discrepancy; the change was explicit.
Observed maximum absolute errors were Small 4.63e-5, Large 1.83e-5.
The pinned Python 3.10 Windows CPU stack and setup are in
[CPU usage](cpu_reference.md). ONNX IR 1.0 incompatibility was resolved with 0.1.12.

**Deferred hardware (3A/3B).** The user approved the split on 2026-10-12 while
waiting for a device. The specification was amended: independently reviewed CPU
parts of 4–6 can proceed before 3B. Replay always uses synthetic power/clocks;
sustained CPU latency drift carries no thermal attribution. Short two-second
public checks validate sustained stopping/counting, not minute-scale thermals.
See the [roadmap](cpu_hardware_roadmap.md) and [device gate](hardware_setup.md).

**Compression (Phase 4).** Preserve canonical checkpoint identity and add derived
artifact hashes, calibration provenance and hashed CPU kernel evidence. INT8 uses
separate training calibration; INT4 covers classifier MatMul weights only.
Channel/2:4/unstructured pruning use dense masks with unchanged shapes/biases.
FP16 tensor dtype does not prove native CPU half arithmetic. ORT preprocessing
uses its file-path API after a ModelProto call exposed external-initializer errors;
failed preparations were retained. The first eight-run compression CLI took
23.046 s. Usage/limitations are maintained in [compression guidance](phase_4_usage.md).

**INT8 diagnosis.** Architecture and targeted controls were recorded before each
diagnostic extension. No scales were refitted and no training/search was performed.
Activation QDQ alone reproduced the quality collapse; coarse shared activation
steps lose small channel signals at the first depthwise input. See
[the retained evidence and reproduction](int8_diagnosis.md). No corrected recipe
was selected.

## Original proposals

These are fixed historical documents; current commands are in the usage guides.

- [Phase 0 architecture](https://github.com/jaworekk85/EdgeAI/blob/e8d9812/docs/research_framework.md).
- [Phase 2 proposal](https://github.com/jaworekk85/EdgeAI/blob/1d958ab/docs/phase_2_plan.md).
- [Phase 3A proposal](https://github.com/jaworekk85/EdgeAI/blob/1d958ab/docs/phase_3a_plan.md).
- [Phase 4 proposal](https://github.com/jaworekk85/EdgeAI/blob/1d958ab/docs/phase_4_plan.md).
- [INT8 diagnostic plan](https://github.com/jaworekk85/EdgeAI/blob/1d958ab/docs/int8_diagnostic_plan.md).
- [Full session log through the diagnosis](https://github.com/jaworekk85/EdgeAI/blob/1d958ab/STATUS.md).

## 2026-10-15 — maintenance

The user requested code and Markdown cleanup before further phase work. The plan
was recorded in status before implementation: reuse worker/config/baseline and
prepared-matrix validation; simplify diagnostic aggregation; consolidate completed
plans and session history; keep one current status page and a documentation index.
No recipe, schema, hardware gate or phase authorization changed.

Verification: 191 passed, 2 Windows symlink skips in 113.73 s, including public
ordinary/sustained/compression smoke. Fresh diagnostic evidence is in ignored
`artifacts/compression_diagnostics/20261014T232246821113Z_b26ef67848/`:
predictions/settings match the retained diagnosis exactly and all tensor/channel
statistics agree within `1e-12`. Source/output hashes, old schema-2/3 reads,
corrupt-baseline refusal, lint and documentation links were verified.

The user subsequently requested a clear capabilities/experiments guide. Added
[a Polish overview](mozliwosci_i_eksperymenty.md) distinguishing implemented CPU
tools, small extensions, planned phases 5/6 and the device-dependent 3B gate.
Experiment ideas describe questions, comparisons and measurements only; no
research runs or new functionality were added.
Checked the guide against current status/configs/usage, local links and Git whitespace.
