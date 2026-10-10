# Configuration skeleton

The directories `devices`, `models`, `workloads`, `compression`, `conditions` and
`experiments` are reserved for typed YAML fragments. Loading, composition and
validation begin in Phase 1. There are no executable experiment configs yet.

The resolved experiment config will be stored with every run. Hardware observations
are captured automatically; a device config is not a handwritten `DeviceProfile`.
Private configuration belongs only in the ignored private tree.
