import numpy as np
import pandas as pd
import pytest

from evaluation import failure_summary, flag_failures, has_repeated_phrase, score

REF = "hôm nay trời đẹp quá"  # 5 words
HYPS = {
    "ok": "hôm nay trời đẹp",
    "looping": "hôm nay " + "trời đẹp quá " * 4,          # 14 words, one phrase 4 times
    "long_no_loop": "một hai ba bốn năm sáu bảy tám chín mười mười một",
    "truncated": "hôm nay",
    "off_language": "And then the day turned into night.",
}


@pytest.fixture
def scored():
    plan = pd.DataFrame({"utt_id": list(HYPS), "condition": "babble_0dB", "dataset": "d",
                         "noise_type": "babble", "snr_db": 0.0})
    manifest = pd.DataFrame({"utt_id": list(HYPS), "text": REF})
    predictions = plan[["utt_id", "condition"]].assign(model="m", variant="v",
                                                       hyp=list(HYPS.values()))
    return score(predictions, plan, manifest)


def test_has_repeated_phrase():
    assert has_repeated_phrase("số số số".split())
    assert has_repeated_phrase("a b c x y x y x y".split())
    assert not has_repeated_phrase("x y x y".split())
    assert not has_repeated_phrase([])


def test_flag_failures_finds_each_kind(scored):
    flags = flag_failures(scored).set_index("utt_id")
    assert flags.loc["looping", "looping"] and flags.loc["looping", "failed"]
    assert not flags.loc["long_no_loop", "looping"]  # long but no repeated phrase
    assert flags.loc["truncated", "truncated"]
    assert flags.loc["off_language", "off_language"]
    assert not flags.loc["ok", "failed"]
    assert "looping" not in scored.columns  # input left unchanged


def test_off_language_needs_a_vietnamese_reference(scored):
    english_ref = scored.assign(ref="the day turned into night")
    assert not flag_failures(english_ref)["off_language"].any()


def test_digits_are_not_another_language():
    plan = pd.DataFrame({"utt_id": ["u"], "condition": "clean", "dataset": "d",
                         "noise_type": "clean", "snr_db": np.nan})
    manifest = pd.DataFrame({"utt_id": ["u"], "text": "sáu mươi sáu mươi mốt"})
    predictions = plan[["utt_id", "condition"]].assign(model="m", variant="v", hyp="60, 61")
    assert not flag_failures(score(predictions, plan, manifest))["failed"].any()


def test_failure_summary_splits_wer(scored):
    row = failure_summary(scored).iloc[0]
    assert row["utterances"] == 5 and row["failed"] == 3
    assert row["failed_pct"] == 60.0
    errors = scored[["w_sub", "w_del", "w_ins"]].sum(axis=1)
    words = scored["w_n_ref"].sum()
    assert row["wer"] == round(100 * errors.sum() / words, 2)
    kept = scored["utt_id"].isin(["ok", "long_no_loop"])
    assert row["wer_without_failed"] == round(100 * errors[kept].sum() / scored["w_n_ref"][kept].sum(), 2)
    assert np.isclose(row["failed_wer_pts"], round(100 * errors[~kept].sum() / words, 2))
