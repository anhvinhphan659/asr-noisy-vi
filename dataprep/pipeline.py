"""The shared dataset: find it, check it is complete, and describe it.

The dataset is prepared once (notebook ``01_prepare_dataset``), saved as a Kaggle Dataset and
attached by every model notebook. Model notebooks never build data: if something is
missing they stop with a message saying what to attach or run.
"""

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DATASET_FILE = "DATASET.json"
DATASET_NAME = "asr-noisy-vi"

# Manifests each part of the dataset needs.
STEP_MANIFESTS = {
    "speech": ["clean.csv"],
    "noise": ["noise_traffic.csv", "noise_voices.csv"],
    "plans": ["test_plan.csv", "dev_plan.csv", "train_schedule.csv"],
}
# Manifests whose ``path`` column points to audio files inside the dataset.
AUDIO_MANIFESTS = ["clean.csv", "noise_traffic.csv", "noise_voices.csv", "test_plan.csv",
                   "dev_plan.csv"]

LICENSES = """\
# Licenses and attribution

This dataset is derived from the sources below. It is shared privately for a course project
and must not be redistributed publicly. Cite the sources in any report.

| Part | Source | License |
|---|---|---|
| `clean/fleurs` | FLEURS `vi_vn` (Conneau et al., 2022), https://huggingface.co/datasets/google/fleurs | CC BY 4.0 |
| `clean/vivos` | VIVOS (Luong & Vu, 2016), https://huggingface.co/datasets/AILAB-VNUHCM/vivos | CC BY-NC-SA 4.0 (non-commercial, share alike) |
| `clean/vietmed` | VietMed (Le-Duc, 2024), https://huggingface.co/datasets/leduckhai/VietMed | MIT |
| `noise/traffic` | FSD50K (Fonseca et al., 2022), https://zenodo.org/records/4060432 | CC BY 4.0 for the dataset; each clip keeps its own Freesound license (CC0, CC BY, CC BY-NC, CC Sampling+) |
| `noise/voices` | MUSAN `speech/librivox` (Snyder et al., 2015), https://www.openslr.org/17/ | CC BY 4.0 |
| `noisy/` | Mixtures of `clean/` and `noise/` | Most restrictive of the mixed parts |

Because VIVOS and some FSD50K clips are non-commercial, the whole dataset is for
non-commercial research and teaching only.
"""


CODE_VERSION_FILE = "CODE_VERSION.json"


def code_version(code_root=None):
    """Which code is running: from ``CODE_VERSION.json`` when the code was installed from
    the packaged bundle (``scripts/package_code.py``), otherwise from git."""
    code_root = Path(code_root) if code_root else Path(__file__).resolve().parents[1]
    version_file = code_root / CODE_VERSION_FILE
    if version_file.is_file():
        return {"source": "bundle",
                **json.loads(version_file.read_text(encoding="utf-8"))}

    def git(*args):
        try:
            out = subprocess.run(["git", *args], capture_output=True, text=True, cwd=code_root,
                                 check=True)
            return out.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain", "--", "core", "dataprep", "evaluation")
    return {"source": "git", "commit": git("rev-parse", "--short", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "uncommitted_changes": None if status is None else bool(status)}


def _rows(path):
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f) - 1


def write_licenses(data_root):
    path = Path(data_root) / "LICENSES.md"
    path.write_text(LICENSES, encoding="utf-8")
    return path


def write_dataset_info(data_root, config, version="v1"):
    """Write ``DATASET.json``: version, code version, parameters and manifest row counts.

    Written last, once every step has finished: its presence marks a complete dataset.
    """
    data_root = Path(data_root)
    manifests = {}
    for names in STEP_MANIFESTS.values():
        for name in names:
            path = data_root / "manifests" / name
            if not path.exists():
                raise FileNotFoundError(f"cannot describe an incomplete dataset: {path} is missing")
            manifests[name] = {"rows": _rows(path)}
    info = {
        "name": DATASET_NAME,
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code": code_version(),
        "config": config,
        "manifests": manifests,
    }
    (data_root / DATASET_FILE).write_text(json.dumps(info, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    return info


def missing_audio(data_root):
    """Audio files referenced by the manifests but absent from ``data_root`` (should be empty)."""
    data_root = Path(data_root)
    rows = []
    for name in AUDIO_MANIFESTS:
        path = data_root / "manifests" / name
        if not path.exists():
            continue
        for rel in pd.read_csv(path, usecols=["path"])["path"].drop_duplicates():
            if not (data_root / rel).exists():
                rows.append({"manifest": name, "path": rel})
    return pd.DataFrame(rows, columns=["manifest", "path"])


def _candidates(search):
    if search is not None:
        return [Path(p) for p in search]
    out = []
    if os.environ.get("ASR_DATA_ROOT"):
        out.append(Path(os.environ["ASR_DATA_ROOT"]))
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for depth in range(1, 5):
            pattern = "/".join(["*"] * depth) + f"/{DATASET_FILE}"
            out += sorted(p.parent for p in kaggle_input.glob(pattern))
    out += [Path("/kaggle/working/asr-data"), Path("/content/asr-data"), Path("../asr-data"),
            Path("asr-data")]
    return out


def find_data_root(search=None):
    """The first folder holding a complete dataset (a ``DATASET.json``).

    Looks at ``$ASR_DATA_ROOT``, every attached Kaggle input, then the usual local folders
    (or only at ``search`` when given). Raises with instructions if none is found.
    """
    candidates = _candidates(search)
    for folder in candidates:
        if (folder / DATASET_FILE).is_file():
            return folder
    looked = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Dataset not found. Attach the Kaggle dataset '{DATASET_NAME}' to this notebook "
        f"(Add Input → Datasets → Shared with you), or set ASR_DATA_ROOT to its folder.\n"
        f"If it does not exist yet, run notebook 01_prepare_dataset first.\n"
        f"Looked in:\n  {looked}"
    )


def require_data(data_root, steps=("speech", "noise", "plans"), check_audio=False):
    """Check that the dataset has what ``steps`` need; return its ``DATASET.json`` content.

    Fails with the list of problems: missing manifests, row counts that differ from
    ``DATASET.json`` (an incomplete copy), and with ``check_audio`` missing audio files.
    Evaluation needs ``("speech", "plans")``; ``ft-noisy`` training also needs ``"noise"``.
    """
    data_root = Path(data_root)
    info_path = data_root / DATASET_FILE
    if not info_path.is_file():
        raise FileNotFoundError(f"{info_path} not found: {data_root} is not a complete dataset. "
                                f"Run notebook 01_prepare_dataset to the end.")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    problems = []
    for step in steps:
        if step not in STEP_MANIFESTS:
            raise ValueError(f"unknown step {step!r}; choose from {list(STEP_MANIFESTS)}")
        for name in STEP_MANIFESTS[step]:
            path = data_root / "manifests" / name
            if not path.exists():
                problems.append(f"missing manifests/{name}")
                continue
            expected = info["manifests"].get(name, {}).get("rows")
            actual = _rows(path)
            if expected is not None and actual != expected:
                problems.append(f"manifests/{name} has {actual} rows, DATASET.json says {expected}")
    if check_audio and not problems:
        lost = missing_audio(data_root)
        if len(lost):
            problems.append(f"{len(lost)} audio files are missing, e.g. "
                            f"{lost['path'].head(3).tolist()}")
    if problems:
        raise RuntimeError(f"Dataset at {data_root} is not usable:\n  - " + "\n  - ".join(problems))
    return info


def describe(info):
    """One line naming the dataset version, for notebook logs and reports."""
    config = info.get("config", {})
    code = info.get("code", {})
    commit = f"{code.get('commit')}{'+changes' if code.get('uncommitted_changes') else ''}"
    return (f"{info.get('name')} {info.get('version')} (code {commit}, "
            f"{info.get('created_at')}), seed {config.get('seed')}, "
            f"test SNRs {config.get('test_snrs')}, schedule epochs {config.get('schedule_epochs')}")
