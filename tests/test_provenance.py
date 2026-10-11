import pkgutil
import subprocess
import sys
from dataclasses import replace

import pytest

import edge_ai
from edge_ai.devices.profile import capture_device_profile
from edge_ai.utils.provenance import capture_provenance


def test_capture_git_commit_dirty_flag_and_versions(tmp_path):
    def git(*arguments):
        return subprocess.run(
            ["git", "-c", f"safe.directory={tmp_path.as_posix()}", "-C", str(tmp_path),
             *arguments], check=True, capture_output=True, text=True,
        ).stdout.strip()

    git("init")
    (tmp_path / "data.txt").write_text("fixture", encoding="utf-8")
    git("add", "data.txt")
    git("-c", "user.name=Test Fixture", "-c", "user.email=test@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "-m", "fixture")
    clean = capture_provenance(tmp_path)
    assert clean.git_commit == git("rev-parse", "HEAD")
    assert not clean.git_dirty and clean.runtime_versions["mock"] == "1"
    (tmp_path / "untracked.txt").write_text("untracked", encoding="utf-8")
    assert capture_provenance(tmp_path).git_dirty
    with pytest.raises(ValueError):
        replace(clean, git_commit="unknown")
    with pytest.raises(ValueError, match="UTC"):
        replace(clean, captured_at_utc="2026-10-11T12:00:00+02:00")


def test_missing_repository_fails_instead_of_inventing_provenance(tmp_path):
    with pytest.raises(RuntimeError, match="Git provenance"):
        capture_provenance(tmp_path)


def test_cpu_profile_has_explicit_unknowns():
    profile = capture_device_profile()
    assert profile.name and profile.os and profile.python
    assert profile.power_mode is None and profile.unavailable["power_mode"]
    assert profile.versions["cuda"] is None and profile.unavailable["versions.cuda"]
    with pytest.raises(ValueError, match="unavailable reason"):
        replace(profile, unavailable={})


def test_all_core_modules_import_without_optional_runtime_dependencies():
    names = [module.name for module in pkgutil.walk_packages(edge_ai.__path__, "edge_ai.")]
    code = '''
import importlib, importlib.abc, sys
class BlockHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'torch', 'onnx', 'onnxruntime', 'tensorrt', 'cuda', 'cv2', 'jtop',
            'numpy', 'matplotlib', 'yaml'
        }:
            raise RuntimeError('Unexpected optional import: ' + fullname)
sys.meta_path.insert(0, BlockHeavy())
for name in sys.argv[1:]:
    importlib.import_module(name)
'''
    completed = subprocess.run(
        [sys.executable, "-c", code, *names], capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
