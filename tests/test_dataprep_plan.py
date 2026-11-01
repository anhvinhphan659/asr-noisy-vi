import numpy as np
import pandas as pd
import pytest

from dataprep import build_dev_plan, build_test_plan, build_train_schedule, parse_voices


def _pool(prefix, n, split_of):
    return pd.DataFrame({"noise_id": [f"{prefix}{i}" for i in range(n)],
                         "split": [split_of(i) for i in range(n)],
                         "path": "unused", "duration_s": 6.0})


@pytest.fixture
def pools():
    return {"traffic": _pool("t", 6, lambda i: "test" if i < 2 else "train"),
            "babble": _pool("v", 12, lambda i: "test" if i < 6 else "train")}


@pytest.fixture
def manifest():
    return pd.DataFrame({"utt_id": [f"u{i}" for i in range(20)], "dataset": "fleurs",
                         "split": "test", "path": [f"clean/u{i}.wav" for i in range(20)],
                         "duration_s": 3.0, "orig_sr": [16000] * 10 + [8000] * 10})


def test_test_plan_is_full_grid(manifest, pools):
    plan = build_test_plan(manifest, pools, snrs=(20, 10, 5, 0))
    assert len(plan) == len(manifest) * 9
    counts = plan.groupby("utt_id")["condition"].nunique()
    assert (counts == 9).all()
    assert set(plan["condition"]) >= {"clean", "traffic_0dB", "babble_20dB"}


def test_test_plan_uses_same_noise_segment_at_every_snr(manifest, pools):
    plan = build_test_plan(manifest, pools)
    noisy = plan[plan["noise_type"] != "clean"]
    specs = noisy.groupby(["utt_id", "noise_type"])[["noise_id", "offset_s", "voices"]].nunique(
        dropna=False)
    assert (specs <= 1).all().all()


def test_test_plan_uses_only_test_noise(manifest, pools):
    plan = build_test_plan(manifest, pools)
    assert set(plan["noise_id"].dropna()) <= {"t0", "t1"}
    voices = {v for spec in plan["voices"].dropna() for v, _ in parse_voices(spec)}
    assert voices <= {f"v{i}" for i in range(6)}


def test_babble_has_three_to_five_distinct_voices(manifest, pools):
    plan = build_test_plan(manifest, pools)
    for spec in plan["voices"].dropna():
        ids = [v for v, _ in parse_voices(spec)]
        assert 3 <= len(ids) <= 5 and len(set(ids)) == len(ids)


def test_plan_marks_narrow_band_speech(manifest, pools):
    plan = build_test_plan(manifest, pools)
    by_utt = plan.groupby("utt_id")["band_sr"].first()
    assert by_utt["u0"] is None or np.isnan(by_utt["u0"])
    assert by_utt["u15"] == 8000


def test_plans_are_reproducible(manifest, pools):
    pd.testing.assert_frame_equal(build_test_plan(manifest, pools, seed=3),
                                  build_test_plan(manifest, pools, seed=3))
    pd.testing.assert_frame_equal(build_train_schedule(manifest, pools, epochs=2, seed=3),
                                  build_train_schedule(manifest, pools, epochs=2, seed=3))


def test_dev_plan_has_one_condition_per_utterance_from_train_noise(manifest, pools):
    plan = build_dev_plan(manifest, pools, seed=0)
    assert len(plan) == len(manifest)
    assert set(plan["noise_id"].dropna()) <= {"t2", "t3", "t4", "t5"}


def test_train_schedule_varies_by_epoch(manifest, pools):
    big = pd.concat([manifest.assign(utt_id=manifest["utt_id"] + f"_{k}") for k in range(50)])
    schedule = build_train_schedule(big, pools, epochs=3, clean_prob=0.25, seed=0)
    assert len(schedule) == 3 * len(big)
    assert schedule["noise_type"].eq("clean").mean() == pytest.approx(0.25, abs=0.03)
    snr = schedule["snr_db"].dropna()
    assert snr.between(0, 20).all() and snr.nunique() > 100
    first = schedule[schedule["epoch"] == 1].set_index("utt_id")["snr_db"]
    second = schedule[schedule["epoch"] == 2].set_index("utt_id")["snr_db"]
    assert not first.equals(second)
    assert set(schedule["noise_id"].dropna()) <= {"t2", "t3", "t4", "t5"}
