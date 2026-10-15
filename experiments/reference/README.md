# Reference pipeline validation — CPU implementation

Phase 2 implements `python -m edge_ai.reference` using MobileNetV3 Small/Large,
explicit ImageNet V1 FP32 weights and 20 seeded Imagenette validation images.
Both CPU backends produce immutable quality/timing/memory records, synthetic
energy estimates, Markdown/CSV tables and Pareto plots. Values are pipeline
checks, not research results. See [CPU usage](../../docs/cpu_reference.md).

CPU runs use mock power and therefore report only labeled energy estimates.
Phase 3B will implement and validate measured device power; changing a CPU config
alone does not enable hardware support. The CPU smoke run should finish in under approximately
two minutes. Preparation downloads/checksums public assets separately from runs.
Weights, data, prepared artifacts and generated results are ignored by Git.
