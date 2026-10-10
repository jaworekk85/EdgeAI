# Working in EdgeAI

Read `EDGE_AI_AGENT_RESEARCH_SPEC.md` before any substantial change; it is the
authoritative development specification. Read `STATUS.md` to identify the
current phase and outstanding review.

- Propose architecture before implementation; stop at phase boundaries and
  update `STATUS.md`. Do not advance without the user's review.
- Research directions are design context, not authorization to add experiments.
- Keep the core hardware-independent and runnable on CPU with mock backends.
- Import hardware and other heavy optional dependencies lazily.
- Preserve all result/provenance invariants in specification section 9.
- Never overwrite run directories or label estimated energy as measured.
- Keep partner-specific files in ignored `private/` or a separate private package.
- Respect scope and safety boundaries in specification section 2.
- Prefer small typed objects and readable modules over registries and factories.

Phase 0 contains a package skeleton and architecture documents only. Do not
mistake planned interfaces in the docs for implemented APIs.
