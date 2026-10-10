# Scope and boundaries

EdgeAI is a measurement and experimentation framework for resource-efficient AI,
initially on Jetson hardware. General workloads include perception, inspection,
mapping, obstacle perception, model monitoring, reliability, adaptive inference
and hardware benchmarking.

Do not implement or optimize selection/tracking of people for targeting, weapon
guidance, attack optimization or harmful autonomous decision pipelines. Public
code, configs and documentation must remain general and publication-friendly.

Partner data, configs, adapters and derived results belong entirely in ignored
`private/` or a separately maintained private package. Public code must run without
either. Do not embed partner names, platform details, power budgets or private data
paths in public assets. Reproducible core scientific results use public data.

Measurements and estimates must remain distinguishable. Never claim measured
energy savings from synthetic power, overwrite prior runs or treat a reference
pipeline smoke run as a research result.
