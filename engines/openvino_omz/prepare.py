"""Run with project Python: creates an isolated OMZ environment under var."""

import hashlib
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "var/openvino-migration"
COMMIT = "37f60eb7fe1dcdedc552b2fb184d646723ed5e80"


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    repo = BASE / "omz"
    if not repo.exists():
        subprocess.run(
            ["git", "clone", "https://github.com/openvinotoolkit/open_model_zoo.git", str(repo)],
            check=True,
        )
    subprocess.run(["git", "-C", str(repo), "checkout", COMMIT], check=True)
    assert not subprocess.check_output(["git", "-C", str(repo), "diff", "HEAD"])
    env = BASE / "venv"
    subprocess.run(
        ["uv", "venv", "--python", "3.11.16", str(env)], check=True
    ) if not env.exists() else None
    py = env / "bin/python"
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(py),
            "-r",
            str(Path(__file__).with_name("requirements.lock.txt")),
        ],
        check=True,
    )
    if Path(sys.prefix).resolve() != env.resolve():
        subprocess.run([str(py), str(Path(__file__).resolve()), "--models"], check=True)
        return
    import yaml

    for name in ("person-detection-retail-0013", "person-reidentification-retail-0277"):
        spec = yaml.safe_load((repo / "models/intel" / name / "model.yml").read_text())
        for item in spec["files"]:
            if not item["name"].startswith("FP32/"):
                continue
            dest = BASE / "models" / name / item["name"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                urllib.request.urlretrieve(item["source"], dest)
            assert len(dest.read_bytes()) == item["size"]
            assert hashlib.sha384(dest.read_bytes()).hexdigest() == item["checksum"]
            print(dest)


if __name__ == "__main__":
    main()
