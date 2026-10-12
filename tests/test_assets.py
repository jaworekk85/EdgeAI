import io
import tarfile
from dataclasses import replace

import pytest

from edge_ai.datasets.manifest import IMAGENETTE_LABELS, select_manifest
from edge_ai.datasets.preprocessing import PreprocessingSpec
from edge_ai.reference import ReferenceConfig
from edge_ai.utils.downloads import extract_images


def archive_with_member(path, name, content=b"fixture", *, link=None):
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo(name)
        if link is not None:
            member.type, member.linkname = tarfile.SYMTYPE, link
            archive.addfile(member)
        else:
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))


def test_extraction_reuses_matching_files_and_never_overwrites_different_data(tmp_path):
    archive = tmp_path / "fixture.tgz"
    root = tmp_path / "data"
    archive_with_member(archive, "images/input", b"correct")
    extract_images(archive, root)
    extract_images(archive, root)
    file = root / "images" / "input"
    assert file.read_bytes() == b"correct"
    file.write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        extract_images(archive, root)
    assert file.read_bytes() == b"changed"


@pytest.mark.parametrize("name,link", [("../escape", None), ("linked", "../outside")])
def test_extraction_refuses_traversal_and_links(tmp_path, name, link):
    archive = tmp_path / "fixture.tgz"
    archive_with_member(archive, name, link=link)
    with pytest.raises(ValueError, match="escapes"):
        extract_images(archive, tmp_path / "data")
    assert not (tmp_path / "escape").exists()


def test_manifest_selection_is_stratified_seeded_and_verified(tmp_path):
    archive = tmp_path / "archive.tgz"
    archive.write_bytes(b"fixture archive identity")
    images = tmp_path / "images"
    for wnid in IMAGENETTE_LABELS:
        folder = images / wnid
        folder.mkdir(parents=True)
        for number in range(6):
            (folder / f"{number}.JPEG").write_bytes(f"{wnid}:{number}".encode())
    arguments = {
        "archive_url": "https://example.invalid/fixture",
        "per_class": 2,
        "preprocessing": PreprocessingSpec(),
    }
    first = select_manifest(tmp_path, images, archive, seed=7, **arguments)
    second = select_manifest(tmp_path, images, archive, seed=7, **arguments)
    other = select_manifest(tmp_path, images, archive, seed=8, **arguments)
    assert first == second and first.digest() != other.digest()
    assert len(first.samples) == 20
    first.verify(tmp_path)
    (tmp_path / first.samples[0].path).write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        first.verify(tmp_path)


def test_reference_rejects_unsupported_settings():
    from edge_ai.config import DeviceSettings
    from edge_ai.profiling.protocol import ProfilingProtocol

    sustained = replace(ReferenceConfig(), protocol=ProfilingProtocol(5, 20, 1, 60))
    assert sustained.protocol.sustained_duration_s == 60
    with pytest.raises(ValueError, match="complete workload pass"):
        ProfilingProtocol(5, 20, 1, 60, sustained_max_iterations=19)
    with pytest.raises(ValueError, match="CPU adapter"):
        replace(ReferenceConfig(), device=DeviceSettings(adapter="mock"))
    with pytest.raises(ValueError, match="subset"):
        replace(ReferenceConfig(), per_class=3)
