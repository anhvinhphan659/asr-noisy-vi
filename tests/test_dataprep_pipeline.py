import json

import pandas as pd
import pytest

from dataprep import (
    DATASET_FILE,
    describe,
    find_data_root,
    missing_audio,
    require_data,
    write_dataset_info,
    write_licenses,
)

CONFIG = {"seed": 42, "test_snrs": [20, 10, 5, 0], "schedule_epochs": 3}


@pytest.fixture
def dataset(tmp_path):
    """A complete (tiny) dataset: every manifest, one audio file each."""
    root = tmp_path / "asr-data"
    (root / "manifests").mkdir(parents=True)
    for name in ["clean.csv", "noise_traffic.csv", "noise_voices.csv", "test_plan.csv",
                 "dev_plan.csv"]:
        audio = f"audio/{name}.wav"
        (root / "audio").mkdir(exist_ok=True)
        (root / audio).write_bytes(b"x")
        pd.DataFrame({"path": [audio, audio]}).to_csv(root / "manifests" / name, index=False)
    pd.DataFrame({"epoch": [1, 1, 2]}).to_csv(root / "manifests" / "train_schedule.csv",
                                              index=False)
    write_licenses(root)
    write_dataset_info(root, CONFIG, version="v1")
    return root


def test_dataset_info_records_config_and_rows(dataset):
    info = json.loads((dataset / DATASET_FILE).read_text(encoding="utf-8"))
    assert info["config"] == CONFIG and info["version"] == "v1"
    assert info["manifests"]["clean.csv"]["rows"] == 2
    assert info["manifests"]["train_schedule.csv"]["rows"] == 3
    assert "VIVOS" in (dataset / "LICENSES.md").read_text(encoding="utf-8")


def test_dataset_info_refuses_incomplete_dataset(tmp_path):
    (tmp_path / "manifests").mkdir()
    with pytest.raises(FileNotFoundError):
        write_dataset_info(tmp_path, CONFIG)


def test_require_data_accepts_complete_dataset(dataset):
    info = require_data(dataset, check_audio=True)
    assert "v1" in describe(info)


def test_require_data_reports_truncated_manifest(dataset):
    pd.DataFrame({"path": ["audio/clean.csv.wav"]}).to_csv(dataset / "manifests" / "clean.csv",
                                                         index=False)
    with pytest.raises(RuntimeError, match="clean.csv has 1 rows"):
        require_data(dataset)


def test_require_data_reports_missing_manifest_only_for_requested_steps(dataset):
    (dataset / "manifests" / "noise_voices.csv").unlink()
    require_data(dataset, steps=("speech", "plans"))  # evaluation does not need noise
    with pytest.raises(RuntimeError, match="missing manifests/noise_voices.csv"):
        require_data(dataset)


def test_require_data_reports_missing_audio(dataset):
    (dataset / "audio" / "test_plan.csv.wav").unlink()
    assert missing_audio(dataset)["manifest"].tolist() == ["test_plan.csv"]
    require_data(dataset)  # cheap check only
    with pytest.raises(RuntimeError, match="audio files are missing"):
        require_data(dataset, check_audio=True)


def test_require_data_without_dataset_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="01_prepare_dataset"):
        require_data(tmp_path)


def test_find_data_root(dataset, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert find_data_root(search=[empty, dataset]) == dataset
    with pytest.raises(FileNotFoundError, match="Attach the Kaggle dataset"):
        find_data_root(search=[empty])


def test_find_data_root_uses_environment_variable(dataset, monkeypatch):
    monkeypatch.setenv("ASR_DATA_ROOT", str(dataset))
    assert find_data_root() == dataset


def test_code_version_reads_bundle_file(tmp_path):
    from dataprep import code_version

    (tmp_path / "CODE_VERSION.json").write_text(
        json.dumps({"commit": "abc1234", "branch": "feature/x", "uncommitted_changes": False}),
        encoding="utf-8")
    version = code_version(tmp_path)
    assert version["source"] == "bundle" and version["commit"] == "abc1234"


def test_code_version_without_bundle_or_git(tmp_path):
    from dataprep import code_version

    assert code_version(tmp_path)["source"] == "git"


def test_dataset_info_records_code_version(dataset):
    info = json.loads((dataset / DATASET_FILE).read_text(encoding="utf-8"))
    assert {"source", "commit"} <= set(info["code"])
