"""Sampled process RSS including native libraries; no claim of allocator peak."""

from threading import Event, Thread
from time import perf_counter

from edge_ai.runtimes.base import MemoryReport
from edge_ai.utils.serialization import positive


class RSSSampler:
    def __init__(self, interval_s: float = 0.01):
        positive(interval_s, "RSS interval")
        self.interval_s = interval_s
        self._stop = Event()
        self._thread = None
        self._samples = []
        self._error = None

    def start(self):
        import psutil

        if self._thread is not None:
            raise RuntimeError("RSS sampler already started")
        process = psutil.Process()
        self._samples, self._error = [], None
        self._stop.clear()

        def sample():
            self._samples.append((perf_counter(), process.memory_info().rss))

        def poll():
            try:
                while not self._stop.wait(self.interval_s):
                    sample()
            except Exception as exc:
                self._error = exc

        sample()
        self._thread = Thread(target=poll, daemon=True)
        self._thread.start()
        self._process = process

    def stop(self, model_file_bytes: int | None = None) -> MemoryReport:
        if self._thread is None:
            raise RuntimeError("RSS sampler not started")
        self._stop.set()
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise RuntimeError("RSS sampler failed to stop")
        self._thread = None
        if self._error is not None:
            raise RuntimeError("RSS sampling failed") from self._error
        self._samples.append((perf_counter(), self._process.memory_info().rss))
        times, values = zip(*self._samples)
        return MemoryReport(
            model_file_bytes=model_file_bytes,
            peak_ram_bytes=max(values),
            measured=True,
            method=(
                "sampled_process_rss_peak; includes libraries/inputs/runtime; may miss short peaks"
            ),
            sampling_interval_s=self.interval_s,
            sample_times_s=tuple(times),
            rss_bytes=tuple(values),
            unavailable={
                "peak_vram_bytes": "CPU execution",
                "runtime_allocation_bytes": "No supported CPU allocator measurement",
            },
        )
