from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from edge_ai.config import ExperimentConfig, load_config
from edge_ai.utils.serialization import from_dict, to_dict


def test_resolved_config_roundtrip_and_overrides(config):
    workspace = Path(__file__).resolve().parents[1]
    changed = load_config(
        Path("configs/experiments/core_mock.yaml"), workspace=workspace,
        overrides={"seed": 19, "protocol": {"iterations": 4}},
    )
    assert changed.seed == 19 and changed.protocol.iterations == 4
    assert changed.protocol.warmup_iterations == config.protocol.warmup_iterations
    assert from_dict(ExperimentConfig, to_dict(changed)) == changed
    assert isinstance(to_dict(changed)["model"], dict)


@pytest.mark.parametrize("field,value", [
    ("seed", True), ("seed", -1), ("seed", 1.5), ("seed", "1"),
    ("experiment", "../escape"), ("unknown", 1),
])
def test_config_rejects_invalid_fields(config, field, value):
    data = to_dict(config)
    data[field] = value
    with pytest.raises(ValueError):
        from_dict(ExperimentConfig, data)


@pytest.mark.parametrize("section,field,value", [
    ("model", "unknown", 1), ("workload", "batch_size", True),
    ("workload", "frame_rate", float("nan")), ("workload", "resolution", [8, 0]),
    ("protocol", "iterations", 0), ("device", "adapter", "jetson"),
])
def test_nested_validation(config, section, field, value):
    data = to_dict(config)
    data[section][field] = value
    with pytest.raises(ValueError):
        from_dict(ExperimentConfig, data)


def write_config(workspace, data, relative="configs/test.yaml"):
    path = workspace / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_private_fragment_cannot_be_reclassified_public(tmp_path, config):
    path = write_config(tmp_path, to_dict(config), "private/config.yaml")
    with pytest.raises(ValueError, match="private fragments"):
        load_config(path, workspace=tmp_path)
    private = load_config(path, workspace=tmp_path, overrides={"workload": {"privacy": "private"}})
    assert private.workload.privacy == "private"


def test_private_source_requires_classification(tmp_path, config):
    data = to_dict(config)
    data["workload"]["source_id"] = "private/data/input"
    path = write_config(tmp_path, data)
    with pytest.raises(ValueError, match="source in private"):
        load_config(path, workspace=tmp_path)


def test_fragment_cannot_escape_config_tree(tmp_path, config):
    data = to_dict(config)
    data["model"] = "outside.yaml"
    (tmp_path / "outside.yaml").write_text("id: outside", encoding="utf-8")
    path = write_config(tmp_path, data)
    with pytest.raises(ValueError, match="escapes"):
        load_config(path, workspace=tmp_path)


def test_duplicate_yaml_keys_and_unsafe_tags_rejected(tmp_path):
    path = tmp_path / "configs" / "test.yaml"
    path.parent.mkdir()
    path.write_text("seed: 1\nseed: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_config(path, workspace=tmp_path)
    path.write_text("!!python/object:builtins.object {}", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        load_config(path, workspace=tmp_path)


def test_model_workload_and_baseline_must_agree(config):
    with pytest.raises(ValueError, match="tasks differ"):
        replace(config, workload=replace(config.workload, task="other"))
    with pytest.raises(ValueError, match="resolutions differ"):
        replace(config, workload=replace(config.workload, resolution=(16, 16)))
    with pytest.raises(ValueError, match="baseline_id"):
        replace(config, changed_components=("runtime",))
