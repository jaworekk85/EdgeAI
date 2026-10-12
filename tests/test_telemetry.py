from dataclasses import replace

import pytest

from edge_ai.devices.tegrastats import parse_tegrastats
from edge_ai.power.replay import ReplaySample, replay_power
from edge_ai.power.trace import energy_from_trace
from edge_ai.utils.serialization import from_dict, to_dict

# Hand-written format examples, not hardware observations. No device provenance.
EXAMPLE = (
    "RAM 220/8192MB (lfb 847x4MB) CPU [1%@102,off,5%@203] EMC_FREQ 95%@1600 "
    "GR3D_FREQ 99%@[1098,1098] CPU@45C GPU@46.5C "
    "VDD_IN 5000mW/4000mW VDD_CPU_GPU_CV 2000mW/1900mW"
)


def test_documented_subset_preserves_raw_units_and_both_rail_values():
    sample = parse_tegrastats(EXAMPLE)
    assert sample.raw_line == EXAMPLE
    assert (sample.ram_used_mb, sample.ram_total_mb) == (220, 8192)
    assert sample.cpu_utilization_pct == (1, None, 5)
    assert sample.cpu_frequency_mhz == (102, None, 203)
    assert sample.gpu_utilization_pct == 99 and sample.gpu_frequency_mhz == (1098, 1098)
    assert sample.temperatures_c == {"CPU": 45, "GPU": 46.5}
    assert sample.rails["VDD_IN"].instantaneous_w == 5
    assert sample.rails["VDD_IN"].average_w == 4
    assert from_dict(type(sample), to_dict(sample)) == sample


@pytest.mark.parametrize("text,usage,frequencies", [
    ("CPU [10%,off,0%]@1000", (10, None, 0), (1000, None, 1000)),
    ("CPU [10%,0%]", (10, 0), (None, None)),
])
def test_cpu_frequency_variants(text, usage, frequencies):
    sample = parse_tegrastats(text)
    assert sample.cpu_utilization_pct == usage and sample.cpu_frequency_mhz == frequencies
    assert sample.ram_total_mb is None and not sample.rails


@pytest.mark.parametrize("text,usage,frequencies", [
    ("GR3D_FREQ 95%@1000", 95, (1000,)),
    ("GR3D_FREQ 95%", 95, ()),
    ("GR3D_FREQ @1000", None, (1000,)),
    ("GR3D_FREQ @[1000, 900]", None, (1000, 900)),
])
def test_gpu_frequency_variants(text, usage, frequencies):
    sample = parse_tegrastats(text)
    assert sample.gpu_utilization_pct == usage and sample.gpu_frequency_mhz == frequencies


@pytest.mark.parametrize("text", [
    "", "unrecognized only", "CPU [bad]", "CPU [101%@10]", "CPU [1%@10]@100",
    "RAM 90/20MB", "RAM bad", "RAM 1/20MB RAM 2/20MB",
    "RAM 1/20MB RAM bad", "CPU [10%] CPU bad", "GR3D_FREQ 50% GR3D_FREQ bad",
    "GR3D_FREQ 200%", "GR3D_FREQ 50% @bad", "GR3D_FREQ @[1,]",
    "CPU@nanC", "CPU@-300C", "CPU@30C CPU@35C",
    "VDD_IN -1mW/1mW", "VDD_IN nanmW/1mW", "VDD_IN 1mW/1mW VDD_IN 2mW/2mW",
    "RAM 1/20MB\nCPU [1%]",
])
def test_invalid_or_ambiguous_telemetry_is_rejected(text):
    with pytest.raises(ValueError):
        parse_tegrastats(text)


def test_replay_integrates_selected_instantaneous_rail_and_keeps_estimate_provenance():
    samples = tuple(ReplaySample(time, parse_tegrastats(text)) for time, text in (
        (0, EXAMPLE), (0.5, EXAMPLE.replace("5000mW", "7000mW")), (1, EXAMPLE),
    ))
    trace = replay_power(samples, rail="VDD_IN", sampling_hz=2, description="format fixture")
    assert trace.watts == {"VDD_IN": (5, 7, 5)}
    assert not trace.metadata.measured and trace.metadata.clock == "synthetic"
    energy = energy_from_trace(trace, rail="VDD_IN", start_s=0, end_s=1)
    assert energy.energy_j is None and energy.energy_estimate_j == pytest.approx(6)
    assert "fixture" in trace.metadata.description
    with pytest.raises(ValueError, match="monotonic"):
        replace(trace.metadata, measured=True)
    with pytest.raises(ValueError, match="cannot declare measured"):
        replace(trace.metadata, measured=True, clock="monotonic")
    with pytest.raises(ValueError, match="synthetic clock"):
        replace(trace.metadata, clock="monotonic")


@pytest.mark.parametrize("times", [(0, 1), (0, 0), (1, 0), (0, 0.6)])
def test_replay_refuses_gaps_irregular_or_nonincreasing_timestamps(times):
    samples = tuple(ReplaySample(time, parse_tegrastats(EXAMPLE)) for time in times)
    with pytest.raises(ValueError, match="irregular"):
        replay_power(samples, rail="VDD_IN", sampling_hz=2, description="fixture")


def test_replay_refuses_missing_rail_or_source_provenance():
    samples = (ReplaySample(0, parse_tegrastats(EXAMPLE)),
               ReplaySample(0.5, parse_tegrastats("CPU@40C")))
    with pytest.raises(ValueError, match="present in every"):
        replay_power(samples, rail="VDD_IN", sampling_hz=2, description="fixture")
    with pytest.raises(ValueError, match="empty"):
        replay_power(samples, rail="VDD_IN", sampling_hz=2, description="")
