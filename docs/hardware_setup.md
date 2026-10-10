# Hardware setup — Phase 3 plan

Initial target: NVIDIA Jetson Orin Nano Super. The core must also support a laptop
CPU and later device adapters. No Jetson support is implemented in Phase 0.

Before hardware profiling, automatically capture device identity, OS, JetPack/L4T,
CUDA, cuDNN, TensorRT and driver versions, memory, power mode, clocks and fan state.
The device profile records observations, not values copied from a config by hand.

Phase 3 will document a compatible environment, permissions needed for onboard
telemetry, onboard rail definitions and sampling limits, optional TensorRT setup,
thermal/sustained-run procedures and `scripts/on_device/` commands. Platform
dependencies remain confined to dedicated modules and do not affect CPU imports.

No setup command here changes power mode, clocks or fan behavior. Concrete hardware
instructions will be written after the installed software and device are known.
