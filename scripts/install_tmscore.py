"""Install the pinned official TMscore source without silently accepting upstream changes."""

import hashlib
import json
import subprocess
from pathlib import Path

SOURCE = "https://zhanggroup.org/TM-score/TMscore.cpp"
SHA256 = "02d24a35857b1076a432e97c4d63cad7993ac2cc6585a1b85a7c33031c84f8b1"


def main():
    directory = Path(__file__).resolve().parents[1] / "data/tools"
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "TMscore.cpp"
    data = (Path(__file__).resolve().parents[1] / "third_party/TMscore.cpp").read_bytes()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise RuntimeError("TMscore source hash mismatch; do not change scoring without a version bump")
    source.write_bytes(data)
    subprocess.run(["g++", "-O3", "-o", str(directory / "TMscore"), str(source)], check=True)
    (directory / "TMscore.build.json").write_text(
        json.dumps(
            {
                "source_url": SOURCE,
                "source_sha256": SHA256,
                "executable_sha256": hashlib.sha256((directory / "TMscore").read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Installed verified TMscore in {directory}")


if __name__ == "__main__":
    main()
