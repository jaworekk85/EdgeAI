# Phase 1 configuration

The directories `devices`, `models`, `workloads`, `compression`, `conditions` and
`experiments` contain or reserve typed YAML fragments. `experiments/core_mock.yaml`
is a core contract fixture; it does not run a reference or research experiment.

Load it with `edge_ai.config.load_config(path, workspace=repository_path)`.
References are workspace-relative YAML paths in the following positions:
`model`, `workload`, `device`, `protocol`, `model.compression` and elements of
`workload.conditions`. Inline mappings are also accepted. References resolve
before explicit recursive overrides are applied; overrides must contain resolved
values, not further references. Unknown fields, duplicate YAML keys, unsafe YAML
tags, invalid quantities and paths outside `configs/` or `private/` are rejected.

The complete resolved experiment config is stored with every run. Hardware
observations are captured automatically; a device config describes requested
settings and is not a handwritten `DeviceProfile`. Private configuration belongs
only in the ignored private tree. Loading any private fragment or private local
source requires `workload.privacy: private`.
