"""Bundle the repo's code for upload as the Kaggle Dataset ``asr-noisy-vi-code``.

The bundle holds the three packages and what ``pip install -e .`` needs, plus
``CODE_VERSION.json`` (commit, branch, uncommitted changes), so that a dataset built on
Kaggle records exactly which code produced it.

    python scripts/package_code.py            # -> dist/asr-noisy-vi-code/ and .zip

Upload the zip as a new version of the Kaggle Dataset ``asr-noisy-vi-code``.
"""

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ["core", "dataprep", "evaluation"]
FILES = ["pyproject.toml", "README.md", "LICENSE"]  # pyproject.toml refers to README and LICENSE
VERSION_FILE = "CODE_VERSION.json"


def _git(*args, cwd=REPO_ROOT):
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, cwd=cwd, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def code_version(repo_root=REPO_ROOT):
    """Commit, branch and whether the packaged files differ from that commit."""
    status = _git("status", "--porcelain", "--", *PACKAGES, *FILES, cwd=repo_root)
    return {
        "commit": _git("rev-parse", "--short", "HEAD", cwd=repo_root),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo_root),
        "uncommitted_changes": bool(status),
        "packaged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def package(out_dir, repo_root=REPO_ROOT):
    """Copy the code into ``out_dir`` (replaced if present) and zip it next to it."""
    repo_root, out_dir = Path(repo_root), Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info")
    for name in PACKAGES:
        shutil.copytree(repo_root / name, out_dir / name, ignore=ignore)
    for name in FILES:
        shutil.copy2(repo_root / name, out_dir / name)
    version = code_version(repo_root)
    (out_dir / VERSION_FILE).write_text(json.dumps(version, indent=2), encoding="utf-8")
    archive = shutil.make_archive(str(out_dir), "zip", root_dir=out_dir)
    return Path(archive), version


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(REPO_ROOT / "dist" / "asr-noisy-vi-code"),
                        help="output folder; the zip is written next to it")
    args = parser.parse_args()
    archive, version = package(args.out)
    print(f"wrote {archive}")
    print(json.dumps(version, indent=2))
    if version["uncommitted_changes"]:
        print("warning: the bundle contains uncommitted changes; commit first so the "
              "dataset can be traced to a commit")


if __name__ == "__main__":
    main()
