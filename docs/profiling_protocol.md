# Profiling protocol — design draft

This document describes intended measurement boundaries. No profiler exists yet.
Implementation begins in Phase 2; Jetson-specific methods arrive in Phase 3.

Each run will record the automatic device profile, power mode, clock/fan settings,
runtime and package versions, model/compression identity, batch size, input
resolution, precision, seed, repetitions, warm-up count, measurement duration and
start/end temperatures when observable. Unsupported observations remain explicit.

1. Load inputs/model and capture provenance before the timed window.
2. Warm up the selected backend on representative batches. Exclude these samples
   from reported timings, FPS and measurement-window energy.
3. Capture a separate idle-power baseline with its duration and device state.
4. Synchronize through the backend around completed inference. Save every timing;
   report mean, median, p95 and p99. Label model-only and end-to-end FPS separately.
5. Collect power samples with timestamps, rail/unit metadata and source properties.
   Integrate over an explicit window. With low-rate sensors, use mean power times
   window duration divided by completed inference count for average inference
   energy. Label the method and report total/above-idle energy separately.
6. Report model file size, peak RAM/VRAM and runtime allocations with their methods.
   Distinguish shared Jetson memory from independent discrete GPU memory.
7. A sustained run lasts minutes and records temperature, throttling and changes
   in latency/power over time. Long hardware runs require a specific request.
8. Validate and store immutable outputs, then report nominal and measured metrics
   together. Synthetic power can only produce explicitly labeled energy estimates.

Trace clock alignment, device synchronization, idle/sustained default durations and
sampling-loss handling will be made concrete and tested with their implementations.
