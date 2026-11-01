import pandas as pd
import pytest

from evaluation import PredictionWriter, load_predictions


@pytest.fixture
def eval_set():
    return pd.DataFrame({"utt_id": ["u1", "u1", "u2", "u2"],
                         "condition": ["clean", "babble_0dB", "clean", "babble_0dB"]})


def test_writes_prediction_format(tmp_path, eval_set):
    path = tmp_path / "p.csv"
    with PredictionWriter(path, "m", "zero-shot") as writer:
        writer.add(eval_set, ["a", "b", "c, with comma", "multi\nline"])
    out = load_predictions([path])
    assert list(out.columns) == ["utt_id", "condition", "model", "variant", "hyp"]
    assert out["hyp"].tolist() == ["a", "b", "c, with comma", "multi line"]


def test_resumes_after_interruption(tmp_path, eval_set):
    path = tmp_path / "p.csv"
    with PredictionWriter(path, "m", "zero-shot") as writer:
        writer.add(eval_set.iloc[:2], ["a", "b"])
    # A new session: only the missing rows are pending.
    with PredictionWriter(path, "m", "zero-shot") as writer:
        pending = writer.pending(eval_set)
        assert pending["utt_id"].tolist() == ["u2", "u2"]
        writer.add(pending, ["c", "d"])
        assert writer.pending(eval_set).empty
    assert len(load_predictions([path])) == 4


def test_empty_hypothesis_is_kept(tmp_path, eval_set):
    path = tmp_path / "p.csv"
    with PredictionWriter(path, "m", "v") as writer:
        writer.add(eval_set.iloc[:1], [""])
    assert load_predictions([path])["hyp"].tolist() == [""]


def test_rejects_file_of_another_system(tmp_path, eval_set):
    path = tmp_path / "p.csv"
    with PredictionWriter(path, "m", "zero-shot") as writer:
        writer.add(eval_set.iloc[:1], ["a"])
    with pytest.raises(ValueError):
        PredictionWriter(path, "m", "ft-noisy")


def test_rejects_length_mismatch_and_use_outside_with(tmp_path, eval_set):
    writer = PredictionWriter(tmp_path / "p.csv", "m", "v")
    with pytest.raises(RuntimeError):
        writer.add(eval_set.iloc[:1], ["a"])
    with writer:
        with pytest.raises(ValueError):
            writer.add(eval_set, ["a"])


def test_progress_counts_pending_rows(tmp_path, eval_set, capsys):
    from dataprep import get_progress_interval, set_progress_interval

    before = get_progress_interval()
    set_progress_interval(0)
    try:
        with PredictionWriter(tmp_path / "p.csv", "m", "v") as writer:
            writer.add(writer.pending(eval_set).iloc[:3], ["a", "b", "c"])
    finally:
        set_progress_interval(before)
    out = capsys.readouterr().out
    assert "predictions m v: 3/4 rows (75.0%)" in out and "0 already written before" in out
