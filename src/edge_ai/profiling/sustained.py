"""Descriptive sustained latency summaries; no inferred thermal causes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from edge_ai.profiling.protocol import ProfilingProtocol
from edge_ai.utils.serialization import finite, positive, validate_types

if TYPE_CHECKING:
    from edge_ai.records.schema import TimingTrace


@dataclass(frozen=True)
class LatencyWindow:
    start_offset_s: float
    end_offset_s: float
    inference_count: int
    mean_s: float
    p95_s: float

    def __post_init__(self) -> None:
        validate_types(self)
        positive(self.start_offset_s, "window start", zero=True)
        positive(self.end_offset_s, "window end")
        if self.end_offset_s <= self.start_offset_s:
            raise ValueError("latency window must have positive duration")
        for name in ("inference_count", "mean_s", "p95_s"):
            positive(getattr(self, name), name)


@dataclass(frozen=True)
class SustainedSummary:
    target_duration_s: float
    actual_duration_s: float
    inference_count: int
    completed_passes: int
    stop_reason: Literal["duration_reached", "inference_cap"]
    windows: tuple[LatencyWindow, ...]
    latency_change_fraction: float
    scope: Literal["end_to_end"] = "end_to_end"
    method: Literal["equal_count_windows_first_to_last_mean"] = (
        "equal_count_windows_first_to_last_mean"
    )

    def __post_init__(self) -> None:
        validate_types(self)
        for name in (
            "target_duration_s", "actual_duration_s", "inference_count", "completed_passes",
        ):
            positive(getattr(self, name), name)
        finite(self.latency_change_fraction, "latency change")
        if not 1 <= len(self.windows) <= 4:
            raise ValueError("sustained summary requires one to four windows")
        if sum(window.inference_count for window in self.windows) != self.inference_count:
            raise ValueError("window counts differ from sustained inference count")


def summarize_sustained(
    protocol: ProfilingProtocol,
    timings: TimingTrace,
    window_start_s: float,
    window_end_s: float,
) -> SustainedSummary:
    """Recompute from raw timings, enforcing the complete-pass stopping protocol."""
    from edge_ai.records.schema import TimingTrace

    if protocol.sustained_duration_s is None:
        raise ValueError("sustained summary requires a duration target")
    timings.__post_init__()
    if timings.scope != "end_to_end":
        raise ValueError("sustained windows require end-to-end timing scope")
    positive(window_start_s, "active window start", zero=True)
    positive(window_end_s, "active window end")
    count = len(timings.durations_s)
    pass_size = protocol.iterations * protocol.repetitions
    if count % pass_size or count > protocol.sustained_max_iterations:
        raise ValueError("sustained count must comprise complete passes within the cap")
    if (
        window_start_s > timings.starts_s[0]
        or window_end_s < timings.starts_s[-1] + timings.durations_s[-1] - 1e-12
    ):
        raise ValueError("sustained active window must contain raw timings")
    elapsed = window_end_s - window_start_s
    # Completing an earlier pass after the target should have stopped the run.
    previous_pass_end = count - pass_size - 1
    if previous_pass_end >= 0 and (
        timings.starts_s[previous_pass_end] + timings.durations_s[previous_pass_end]
        - window_start_s >= protocol.sustained_duration_s
    ):
        raise ValueError("sustained run continued after a completed pass reached the target")
    reached = elapsed >= protocol.sustained_duration_s
    if not reached and count + pass_size <= protocol.sustained_max_iterations:
        raise ValueError("sustained run stopped before either target or cap")
    window_count = min(4, count)
    windows = []
    for index in range(window_count):
        first, last = index * count // window_count, (index + 1) * count // window_count
        trace = TimingTrace(
            timings.starts_s[first:last], timings.durations_s[first:last],
            timings.warmup_iterations_excluded, measured=timings.measured,
            clock=timings.clock, scope=timings.scope,
        )
        summary = trace.summary(1)
        windows.append(LatencyWindow(
            trace.starts_s[0] - window_start_s,
            trace.starts_s[-1] + trace.durations_s[-1] - window_start_s,
            last - first, summary["mean_s"], summary["p95_s"],
        ))
    return SustainedSummary(
        protocol.sustained_duration_s, elapsed, count, count // pass_size,
        "duration_reached" if reached else "inference_cap", tuple(windows),
        windows[-1].mean_s / windows[0].mean_s - 1,
    )
