# Reference pipeline validation — planned

Implementation starts in Phase 2. Use one public pretrained model family with
two or three sizes/precisions and a small public dataset subset or video. Collect
quality, latency, memory, power and energy, then emit the standard table and Pareto
plots. Values are pipeline checks, not research results.

CPU runs use mock power and therefore report only labeled energy estimates.
Jetson runs use measured power. Keep the experiment config the same apart from
the device selection. The CPU smoke run should finish in under approximately
two minutes. No weights/data downloads or extra research experiments in Phase 0.
