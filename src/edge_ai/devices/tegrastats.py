"""Offline parsing of a documented tegrastats subset, without hardware imports.

Examples are format fixtures. Parsing text does not establish device identity,
acquisition timestamps, measurement provenance or thermal throttling.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from edge_ai.utils.serialization import finite, nonempty, positive, validate_types

_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"


@dataclass(frozen=True)
class RailReading:
    instantaneous_w: float
    average_w: float

    def __post_init__(self) -> None:
        validate_types(self)
        positive(self.instantaneous_w, "instantaneous rail power", zero=True)
        positive(self.average_w, "average rail power", zero=True)


@dataclass(frozen=True)
class TegrastatsSample:
    raw_line: str
    ram_used_mb: int | None = None
    ram_total_mb: int | None = None
    cpu_utilization_pct: tuple[int | None, ...] = ()
    cpu_frequency_mhz: tuple[int | None, ...] = ()
    gpu_utilization_pct: int | None = None
    gpu_frequency_mhz: tuple[int, ...] = ()
    temperatures_c: dict[str, float] = field(default_factory=dict)
    rails: dict[str, RailReading] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_types(self)
        nonempty(self.raw_line, "raw telemetry line")
        if (self.ram_used_mb is None) != (self.ram_total_mb is None):
            raise ValueError("RAM observation requires used and total values")
        if self.ram_total_mb is not None:
            positive(self.ram_total_mb, "total RAM")
            positive(self.ram_used_mb, "used RAM", zero=True)
            if self.ram_used_mb > self.ram_total_mb:
                raise ValueError("RAM used exceeds reported total")
        if len(self.cpu_utilization_pct) != len(self.cpu_frequency_mhz):
            raise ValueError("CPU usage/frequency counts differ")
        for value in (*self.cpu_utilization_pct, self.gpu_utilization_pct):
            if value is not None and not 0 <= value <= 100:
                raise ValueError("processor utilization must be in [0, 100]")
        for value in (*self.cpu_frequency_mhz, *self.gpu_frequency_mhz):
            if value is not None:
                positive(value, "processor frequency", zero=True)
        for name, temperature in self.temperatures_c.items():
            nonempty(name, "temperature sensor")
            finite(temperature, "temperature")
            if temperature < -273.15:
                raise ValueError("temperature is below absolute zero")
        for name in self.rails:
            nonempty(name, "power rail")


def parse_tegrastats(line: str) -> TegrastatsSample:
    """Preserve unsupported fields, reject malformed recognized fields.

    RAM is retained in the source's MB unit; power mW is converted to W. Rail
    averages are preserved but never substituted for instantaneous readings.
    """
    if type(line) is not str or "\n" in line or "\r" in line:
        raise ValueError("expected one telemetry line")
    nonempty(line, "telemetry line")
    fields = {}
    remainder = line

    def take(pattern: str, marker: str):
        nonlocal remainder
        matches = list(re.finditer(pattern, remainder))
        if len(matches) > 1:
            raise ValueError(f"duplicate {marker} observation")
        if not matches:
            if re.search(rf"\b{marker}(?=\s|$)", remainder):
                raise ValueError(f"malformed {marker} observation")
            return None
        match = matches[0]
        remainder = remainder[:match.start()] + " " + remainder[match.end():]
        if re.search(rf"\b{marker}(?=\s|$)", remainder):
            raise ValueError(f"duplicate or malformed {marker} observation")
        return match

    ram = take(r"\bRAM\s+(\d+)/(\d+)MB\b", "RAM")
    if ram:
        fields.update(ram_used_mb=int(ram[1]), ram_total_mb=int(ram[2]))
    cpu = take(r"\bCPU\s+\[([^\]]+)\](?:@(\d+))?(?=\s|$)", "CPU")
    if cpu:
        usage, frequencies = [], []
        for core in cpu[1].split(","):
            core = core.strip()
            if core == "off":
                usage.append(None)
                frequencies.append(None)
                continue
            match = re.fullmatch(r"(\d+)%(?:@(\d+))?", core)
            if match is None or (match[2] is not None and cpu[2] is not None):
                raise ValueError("malformed CPU core observation")
            usage.append(int(match[1]))
            frequency = match[2] or cpu[2]
            frequencies.append(int(frequency) if frequency is not None else None)
        fields.update(cpu_utilization_pct=tuple(usage), cpu_frequency_mhz=tuple(frequencies))
    gpu = take(
        r"\bGR3D_FREQ\s+(?:(\d+)%)?(?:\s*@(?:(\d+)|\[([\d,\s]+)\]))?(?=\s|$)",
        "GR3D_FREQ",
    )
    if gpu:
        if remainder[gpu.start():].lstrip().startswith("@"):
            raise ValueError("malformed GPU frequency observation")
        if not any(gpu[index] for index in (1, 2, 3)):
            raise ValueError("malformed GR3D_FREQ observation")
        fields["gpu_utilization_pct"] = int(gpu[1]) if gpu[1] else None
        if gpu[2]:
            fields["gpu_frequency_mhz"] = (int(gpu[2]),)
        elif gpu[3]:
            try:
                fields["gpu_frequency_mhz"] = tuple(int(item.strip()) for item in gpu[3].split(","))
            except ValueError as exc:
                raise ValueError("malformed GPU frequencies") from exc
    temperatures, rails = {}, {}
    for token in remainder.split():
        if "@" in token and token.endswith("C"):
            match = re.fullmatch(rf"([A-Za-z][A-Za-z0-9_]*)@({_NUMBER})C", token)
            if match is None or match[1] in temperatures:
                raise ValueError("malformed or duplicate temperature observation")
            temperatures[match[1]] = float(match[2])
    # Only name + mW/mW tokens are power observations; preserve unrelated text.
    tokens = remainder.split()
    for index, token in enumerate(tokens):
        if "mW" not in token:
            continue
        match = re.fullmatch(rf"({_NUMBER})mW/({_NUMBER})mW", token)
        if (
            index == 0 or match is None
            or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", tokens[index - 1])
        ):
            raise ValueError("malformed rail power observation")
        name = tokens[index - 1]
        if name in rails:
            raise ValueError("duplicate rail power observation")
        rails[name] = RailReading(float(match[1]) / 1000, float(match[2]) / 1000)
    if not fields and not temperatures and not rails:
        raise ValueError("no supported telemetry observations")
    return TegrastatsSample(line, **fields, temperatures_c=temperatures, rails=rails)
