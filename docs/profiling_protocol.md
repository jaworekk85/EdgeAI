# Profiling protocol — CPU implementation and hardware plan

Phase 2 implements short serial CPU runs, with separate model/end-to-end timing,
sampled process RSS, and synthetic active/idle power. Phase 3A adds bounded sustained
CPU profiling and offline telemetry parsing/replay. Jetson-specific live power,
thermal observation and TensorRT execution remain Phase 3B.

Each run records the automatic device profile, power mode, clock/fan settings,
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
7. Sustained CPU mode repeats complete passes until its duration target or inference
   cap. Save actual count/duration/stop reason and descriptive end-to-end latency
   windows. Multi-minute operation is opt-in. Jetson temperature, throttling and
   power drift need a later real-device run, not inference from CPU latency changes.
8. Validate and store immutable outputs, then report nominal and measured metrics
   together. Synthetic power can only produce explicitly labeled energy estimates.

CPU configurations run in fresh subprocesses to isolate allocator/library state.
RSS sampling starts before model load at a recorded interval (default 10 ms).
The sampled process peak includes libraries, inputs, warm-up and native runtime
state and may miss short peaks. CPU VRAM/allocator totals remain unavailable.

Warm-up occurs before opening the active window. End-to-end timing begins before
image decoding/preprocessing/native conversion and ends after materializing logits.
Model-only timing wraps prepared native input through completed CPU inference.
Quality aggregation occurs after closing the power window. Model throughput uses
the sum of inference durations; end-to-end throughput uses the observed trace span.
Separate window throughput fields preserve between-call overhead.

In sustained mode score each materialized output immediately after its timing;
this bounds retained logits. Scoring overhead belongs to the active energy window,
not model/end-to-end latency. Quality counts repeated observations of the same
verified subset. One pass has `iterations * repetitions` inferences; the reference
cycles manifest indices. A time target is checked after a pass, so a full-pass
overshoot is possible. An inference-cap stop before target is explicitly labeled.
Raw trace retention remains proportional to actual run length and the count cap.

The window mock creates assumed constant watts on the observed monotonic interval,
using a configured sampling grid plus exact endpoints. Its values are always
`measured=False`; a real clock does not turn modeled power into a sensor reading.
Idle power has its own raw synthetic trace. Per-inference estimates use the actual
inference count; above-idle estimates preserve their provenance and signed values.
Traces with declared sample loss are refused until a gap policy is implemented.

Schema-v2 records embed the verified image manifest and artifact identities.
Reports are generated before storage and written before the completion marker;
completed run directories are never extended. `load_stored_comparison` also checks
all referenced peer runs are complete. Default reports compare Small/Large within
one runtime; cross-runtime overrides show mismatches in tables and plots.

The CPU smoke budget includes workers, local loading, warm-up, inference, reports
and storage; first downloads/export are separate setup. See
[CPU reference usage](cpu_reference.md) for tested commands and numerical details.

Schema-3 sustained records also require a summary recomputed from raw end-to-end
timings and the active window; both timing scopes and energy use actual counts.
Up to four equal-count windows report their offsets/counts/mean/p95. The first-to-last
mean change is descriptive and carries no thermal attribution. Schema-1/2 files
remain readable. Different profiling protocols require visible comparison overrides.
See [3A usage](phase_3a_usage.md) for commands and replay provenance.
