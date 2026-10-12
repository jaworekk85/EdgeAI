"""Bounded exclusive downloads and extraction confined to a named data root."""

import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path

from edge_ai.utils.artifacts import sha256_file


def download_public(
    url: str, destination: Path, *, md5: str | None = None, sha256_prefix: str | None = None
) -> Path:
    if not url.startswith("https://"):
        raise ValueError("public assets require HTTPS")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            length = response.headers.get("Content-Length")
            print(
                f"Asset download: {destination.name}; advertised bytes: {length or 'unknown'}",
                flush=True,
            )
            if length is not None and int(length) > 450 * 1024 * 1024:
                raise ValueError("asset exceeds reference download limit")
            downloaded = 0
            with destination.open("xb") as stream:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    downloaded += len(chunk)
                    if downloaded > 450 * 1024 * 1024:
                        raise ValueError("asset exceeds reference download limit")
                    stream.write(chunk)
            if length is not None and downloaded != int(length):
                raise ValueError("asset download length differs from advertised size")
    if md5 is not None:
        checksum = hashlib.md5()  # upstream archive identity, additionally record SHA-256
        with destination.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(chunk)
        if checksum.hexdigest() != md5:
            raise ValueError("archive checksum mismatch; cached files are never overwritten")
    if sha256_prefix is not None and not sha256_file(destination).startswith(sha256_prefix):
        raise ValueError("checkpoint checksum mismatch; cached files are never overwritten")
    return destination


def extract_images(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve(strict=True)
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            target = (root / member.name).resolve()
            if not target.is_relative_to(root) or member.issym() or member.islnk():
                raise ValueError("archive member escapes extraction root")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                if target.exists():
                    with source.extractfile(member) as data:
                        expected = hashlib.sha256()
                        for chunk in iter(lambda: data.read(1024 * 1024), b""):
                            expected.update(chunk)
                    if sha256_file(target) != expected.hexdigest():
                        raise ValueError("existing extracted file differs from verified archive")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as data, target.open("xb") as output:
                    shutil.copyfileobj(data, output)
            else:
                raise ValueError("unsupported archive member type")
