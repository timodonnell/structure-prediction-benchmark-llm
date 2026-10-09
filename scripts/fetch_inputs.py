"""Download and verify the immutable source-input release before an offline build."""

import argparse
import hashlib
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from pdbthink.util import REPO_ROOT, read_json, sha256_file


def install(archive: Path, destination: Path, manifest: dict):
    if sha256_file(archive) != manifest["sha256"]:
        raise ValueError("Input archive hash mismatch; do not change the frozen source hashes")
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        if len({m.name for m in members}) != len(members):
            raise ValueError("Duplicate archive member")
        if {m.name for m in members} != set(manifest["files"]):
            raise ValueError("Archive does not match the frozen file list")
        for member in members:
            relative = Path(member.name)
            if not member.isfile() or relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe archive member")
            target = destination / relative
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError("Archive destination escapes the repository")
            data = tar.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != manifest["files"][member.name]:
                raise ValueError(f"Source hash mismatch: {member.name}")
            if target.exists():
                if sha256_file(target) != manifest["files"][member.name]:
                    raise ValueError(f"Refusing to overwrite different local data: {target}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use an already-downloaded release asset")
    args = parser.parse_args()
    manifest = read_json(REPO_ROOT / "data/provenance/input_release.json")
    with tempfile.TemporaryDirectory(prefix="structure-prediction-inputs-") as temp:
        archive = args.archive or Path(temp) / "input-sources-v1.tar.gz"
        if args.archive is None:
            print("Downloading", manifest["url"], flush=True)
            with urllib.request.urlopen(manifest["url"], timeout=60) as response, archive.open("wb") as out:
                shutil.copyfileobj(response, out)
        install(archive, REPO_ROOT, manifest)
    print(f"Verified {len(manifest['files'])} source files")


if __name__ == "__main__":
    main()
