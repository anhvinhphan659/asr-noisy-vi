import numpy as np
import pandas as pd
import pytest

from evaluation import bootstrap_ci, compare, load_predictions, missing_predictions, score, summarize, wer_table


@pytest.fixture
def plan():
    rows = []
    for i in range(40):
        rows.append({"utt_id": f"u{i}", "condition": "clean", "dataset": "d",
                     "noise_type": "clean", "snr_db": np.nan})
        rows.append({"utt_id": f"u{i}", "condition": "babble_0dB", "dataset": "d",
                     "noise_type": "babble", "snr_db": 0.0})
    return pd.DataFrame(rows)


@pytest.fixture
def manifest():
    return pd.DataFrame({"utt_id": [f"u{i}" for i in range(40)], "text": "Hôm nay trời đẹp quá."})


def _predictions(plan, model, variant, noisy_hyp):
    hyp = np.where(plan["condition"] == "clean", "hôm nay trời đẹp quá", noisy_hyp)
    return plan[["utt_id", "condition"]].assign(model=model, variant=variant, hyp=hyp)


@pytest.fixture
def scored(plan, manifest):
    predictions = pd.concat([
        _predictions(plan, "m", "zero-shot", "hôm nay chời"),       # 3 errors / 5 words
        _predictions(plan, "m", "ft-noisy", "Hôm nay trời đẹp!"),   # 1 error / 5 words
    ], ignore_index=True)
    return score(predictions, plan, manifest)


def test_score_normalizes_both_sides(scored):
    clean = scored[scored["condition"] == "clean"]
    assert (clean["w_sub"] + clean["w_del"] + clean["w_ins"] == 0).all()


def test_summarize_gives_corpus_wer(scored):
    summary = summarize(scored).set_index(["variant", "noise_type"])
    assert summary.loc[("zero-shot", "babble"), "wer"] == 60.0
    assert summary.loc[("ft-noisy", "babble"), "wer"] == 20.0
    assert summary.loc[("ft-noisy", "clean"), "wer"] == 0.0
    assert summary.loc[("zero-shot", "babble"), "del_pct"] == 40.0


def test_wer_table_has_one_column_per_system(scored):
    table = wer_table(scored)
    assert table.shape == (2, 2)


def test_bootstrap_ci_contains_point_estimate(scored):
    ci = bootstrap_ci(scored, metric="wer", n_boot=200)
    assert (ci["ci_low"] <= ci["wer"]).all() and (ci["wer"] <= ci["ci_high"]).all()


def test_compare_detects_a_real_improvement(scored):
    out = compare(scored, ("m", "zero-shot"), ("m", "ft-noisy"), n_boot=200)
    babble = out[out["noise_type"] == "babble"].iloc[0]
    assert babble["diff"] == -40.0 and babble["significant"]
    clean = out[out["noise_type"] == "clean"].iloc[0]
    assert clean["diff"] == 0.0 and not clean["significant"]


def test_score_rejects_predictions_outside_plan(plan, manifest):
    bad = pd.DataFrame([{"utt_id": "zzz", "condition": "clean", "model": "m",
                         "variant": "v", "hyp": "x"}])
    with pytest.raises(ValueError):
        score(bad, plan, manifest)


def test_load_and_check_predictions(tmp_path, plan):
    preds = _predictions(plan, "m", "zero-shot", "x")
    preds.iloc[:-1].to_csv(tmp_path / "a.csv", index=False)
    loaded = load_predictions([tmp_path / "a.csv"])
    assert len(missing_predictions(loaded, plan)) == 1
    preds.iloc[:2].to_csv(tmp_path / "b.csv", index=False)
    with pytest.raises(ValueError):
        load_predictions([tmp_path / "a.csv", tmp_path / "b.csv"])
