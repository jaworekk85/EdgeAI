"""Automatic CPU-host observations; no Jetson imports."""

from __future__ import annotations

import ctypes
import os
import platform
from dataclasses import dataclass, field
from importlib import metadata

from edge_ai.utils.serialization import nonempty, positive, validate_types


@dataclass(frozen=True)
class DeviceProfile:
    name: str
    module: str
    os: str
    python: str
    memory_bytes: int | None
    power_mode: str | None
    clocks: str | None
    fan_mode: str | None
    versions: dict[str, str | None]
    unavailable: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_types(self)
        for name in ("name", "module", "os", "python"):
            nonempty(getattr(self, name), name)
        if self.memory_bytes is not None:
            positive(self.memory_bytes, "memory_bytes")
        for name in ("memory_bytes", "power_mode", "clocks", "fan_mode"):
            if getattr(self, name) is None and not self.unavailable.get(name):
                raise ValueError(f"missing observation requires unavailable reason: {name}")
        for name, version in self.versions.items():
            if version is None and not self.unavailable.get(f"versions.{name}"):
                raise ValueError(f"missing version requires unavailable reason: {name}")


def package_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _host_memory() -> int | None:
    try:
        if os.name == "nt":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                    ("total", ctypes.c_ulonglong), ("available", ctypes.c_ulonglong),
                    ("page_total", ctypes.c_ulonglong), ("page_available", ctypes.c_ulonglong),
                    ("virtual_total", ctypes.c_ulonglong),
                    ("virtual_available", ctypes.c_ulonglong),
                    ("extended_available", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.total)
        elif hasattr(os, "sysconf"):
            return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        pass
    return None


def capture_device_profile() -> DeviceProfile:
    memory = _host_memory()
    versions = package_versions(("edge-ai-research", "PyYAML"))
    versions.update(dict.fromkeys(("jetpack", "l4t", "cuda", "cudnn", "tensorrt", "driver")))
    unavailable = {
        name: "Not captured by the CPU host adapter; requires a hardware adapter"
        for name in ("power_mode", "clocks", "fan_mode")
    }
    if memory is None:
        unavailable["memory_bytes"] = "Host memory query unavailable"
    for name, version in versions.items():
        if version is None:
            unavailable[f"versions.{name}"] = "Unavailable to the CPU host adapter"
    return DeviceProfile(
        name=platform.node() or "unnamed-host",
        module=platform.machine() or "unknown-architecture",
        os=platform.platform(), python=platform.python_version(), memory_bytes=memory,
        power_mode=None, clocks=None, fan_mode=None, versions=versions, unavailable=unavailable,
    )
