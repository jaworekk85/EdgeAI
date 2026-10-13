# Hardware setup — Phase 3B deferred

Initial target: NVIDIA Jetson Orin Nano Super. The core must also support a laptop
CPU and later device adapters. Phase 3A prepares offline telemetry parsing/replay
and sustained CPU profiling. Live Jetson support and validation remain Phase 3B.
See the [3A usage](phase_3a_usage.md) and [revised roadmap](cpu_hardware_roadmap.md).

Before hardware profiling, automatically capture device identity, OS, JetPack/L4T,
CUDA, cuDNN, TensorRT and driver versions, memory, power mode, clocks and fan state.
The device profile records observations, not values copied from a config by hand.

Phase 3B will document a compatible environment, permissions needed for onboard
telemetry, onboard rail definitions and sampling limits, optional TensorRT setup,
thermal/sustained-run procedures and `scripts/on_device/` commands. Platform
dependencies remain confined to dedicated modules and do not affect CPU imports.

No setup command here changes power mode, clocks or fan behavior. Concrete hardware
instructions will be written after the installed software and device are known.

## Hardware acceptance gate

After acquiring the device, propose and review the 3B architecture using its
observed software stack. Verify at least the following before calling it supported:

1. Automatically capture identity, package/runtime versions, memory, power mode,
   clocks and fan state with observation sources and explicit unavailable reasons.
2. Validate rail names/coverage, sensor units, cadence, timestamp alignment and
   sample-loss policy on real telemetry. Preserve raw lines and thermal traces.
3. Implement lazy TensorRT loading/export and check output quality against the
   CPU baseline. Record target/device/runtime-dependent engine identities.
4. Run warm-up, idle and sustained profiling on the device; separate model and
   end-to-end timing, preserve raw timings, and derive energy only from measured
   power over covered windows. Observe temperature and throttling evidence rather
   than infer throttling from latency drift alone.
5. Recheck compression kernel support, controller switching costs and condition
   behavior introduced during CPU development. CPU acceptance does not substitute
   for Jetson validation or establish hardware energy savings.
   CPU Phase 4 INT4 is classifier weight-only, and channel pruning uses masks.
   Do not assume TensorRT compatibility, native arithmetic or sparse acceleration
   from CPU artifact creation or operator-event evidence.
6. Document the supported environment and on-device commands, run the hardware
   reference, update status and stop for review.

The 3A parser is based on documented
[tegrastats formats](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html),
not a verified local device capture. No compatibility version is selected before
the installed JetPack/L4T is known.
