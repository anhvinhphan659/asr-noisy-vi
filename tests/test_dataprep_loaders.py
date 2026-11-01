import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from core import SAMPLE_RATE, measure_snr
from dataprep import (
    NoiseBank,
    TrainSet,
    build_test_plan,
    build_train_schedule,
    collate_raw,
    iter_eval_batches,
    load_eval_set,
    load_train_set,
    write_noisy_set,
)


@pytest.fixture
def env(tmp_path, speech):
    """A small DATA_ROOT: 2 datasets × (2 train + 2 test) utterances, noise pools, test plan."""
    rng = np.random.default_rng(0)
    rows = []
    for d in ["a", "b"]:
        for split in ["train", "test"]:
            for i in range(2):
                n = len(speech) - 4000 * i  # different durations
                path = f"clean/{d}/{split}/{d}{split}{i}.wav"
                (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
                sf.write(tmp_path / path, speech[:n], SAMPLE_RATE, subtype="PCM_16")
                rows.append({"utt_id": f"{d}{split}{i}", "dataset": d, "split": split,
                             "path": path, "duration_s": n / SAMPLE_RATE, "orig_sr": 16000,
                             "text": f"câu {d} {i}"})
    manifest = pd.DataFrame(rows)

    pool_rows = []
    for prefix, kind, n in [("t", "traffic", 4), ("v", "voice", 10)]:
        for i in range(n):
            path = tmp_path / "noise" / f"{prefix}{i}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, 0.1 * rng.standard_normal(3 * SAMPLE_RATE), SAMPLE_RATE)
            pool_rows.append({"noise_id": f"{prefix}{i}", "kind": kind, "path": str(path),
                              "duration_s": 3.0, "split": "test" if i % 2 else "train"})
    pool = pd.DataFrame(pool_rows)
    traffic, voices = pool[pool["kind"] == "traffic"], pool[pool["kind"] == "voice"]
    pools = {"traffic": traffic, "babble": voices}

    test = manifest[manifest["split"] == "test"]
    plan = write_noisy_set(build_test_plan(test, pools), NoiseBank(traffic, voices), tmp_path)
    train = manifest[manifest["split"] == "train"]
    schedule = build_train_schedule(train, pools, epochs=2, seed=0, clean_prob=0.0)
    return tmp_path, manifest, plan, schedule, traffic, voices


def test_load_eval_set_joins_text_and_duration(env):
    _, manifest, plan, *_ = env
    eval_set = load_eval_set(plan, manifest)
    assert len(eval_set) == len(plan) == 4 * 9
    assert eval_set["text"].str.startswith("câu").all()
    assert eval_set["duration_s"].notna().all()


def test_load_eval_set_filters(env):
    _, manifest, plan, *_ = env
    assert set(load_eval_set(plan, manifest, datasets=["a"])["dataset"]) == {"a"}
    only_clean = load_eval_set(plan, manifest, conditions=["clean"])
    assert set(only_clean["condition"]) == {"clean"} and len(only_clean) == 4
    small = load_eval_set(plan, manifest, limit=1)
    assert small["utt_id"].nunique() == 2 and len(small) == 2 * 9


def test_load_eval_set_accepts_csv_paths(env, tmp_path):
    _, manifest, plan, *_ = env
    plan.to_csv(tmp_path / "plan.csv", index=False)
    manifest.to_csv(tmp_path / "clean.csv", index=False)
    assert len(load_eval_set(tmp_path / "plan.csv", tmp_path / "clean.csv")) == len(plan)


def test_iter_eval_batches_covers_everything_sorted_by_duration(env):
    root, manifest, plan, *_ = env
    eval_set = load_eval_set(plan, manifest)
    seen, durations = [], []
    for rows, audios in iter_eval_batches(eval_set, root, batch_size=5):
        assert len(rows) == len(audios) <= 5
        for (_, row), audio in zip(rows.iterrows(), audios):
            assert audio.dtype == np.float32
            assert len(audio) == round(row["duration_s"] * SAMPLE_RATE)
        seen += list(zip(rows["utt_id"], rows["condition"]))
        durations += list(rows["duration_s"])
    assert sorted(seen) == sorted(zip(eval_set["utt_id"], eval_set["condition"]))
    assert durations == sorted(durations)


def test_train_set_clean(env):
    root, manifest, *_ = env
    train = TrainSet(manifest, root)
    assert len(train) == 4
    item = train[0]
    assert set(item) == {"utt_id", "audio", "text", "noise_type", "snr_db"}
    assert item["noise_type"] == "clean" and item["audio"].dtype == np.float32


def test_train_set_noisy_follows_schedule_per_epoch(env):
    root, manifest, _, schedule, traffic, voices = env
    noisy = TrainSet(manifest, root, schedule=schedule, bank=NoiseBank(traffic, voices))
    clean = TrainSet(manifest, root)
    by_key = schedule.set_index(["epoch", "utt_id"])
    for epoch in (1, 2):
        noisy.set_epoch(epoch)
        for i in range(len(noisy)):
            item = noisy[i]
            spec = by_key.loc[(epoch, item["utt_id"])]
            assert item["noise_type"] == spec["noise_type"]
            assert abs(measure_snr(clean[i]["audio"], item["audio"]) - spec["snr_db"]) < 0.5
    noisy.set_epoch(1)
    first = noisy[0]["audio"]
    np.testing.assert_array_equal(first, noisy[0]["audio"])  # deterministic
    noisy.set_epoch(2)
    assert not np.array_equal(first, noisy[0]["audio"])  # new draw each epoch


def test_train_set_rejects_unknown_epoch_and_incomplete_schedule(env):
    root, manifest, _, schedule, traffic, voices = env
    bank = NoiseBank(traffic, voices)
    noisy = TrainSet(manifest, root, schedule=schedule, bank=bank)
    with pytest.raises(ValueError):
        noisy.set_epoch(3)
    with pytest.raises(ValueError):
        TrainSet(manifest, root, schedule=schedule.iloc[1:], bank=bank)
    with pytest.raises(ValueError):
        TrainSet(manifest, root, schedule=schedule)


def test_load_train_set_by_variant(env, tmp_path):
    root, manifest, _, schedule, traffic, voices = env
    traffic.to_csv(tmp_path / "traffic.csv", index=False)
    assert load_train_set(manifest, root, "ft-clean").specs is None
    noisy = load_train_set(manifest, root, "ft-noisy", schedule=schedule,
                           noise_pools=[tmp_path / "traffic.csv", voices])
    assert noisy.specs is not None
    with pytest.raises(ValueError):
        load_train_set(manifest, root, "ft-noisy")
    with pytest.raises(ValueError):
        load_train_set(manifest, root, "other")


def test_train_set_can_be_sent_to_dataloader_workers(env):
    # DataLoader workers receive a pickled copy of the dataset.
    import pickle

    root, manifest, _, schedule, traffic, voices = env
    noisy = TrainSet(manifest, root, schedule=schedule, bank=NoiseBank(traffic, voices))
    noisy.set_epoch(2)
    copy = pickle.loads(pickle.dumps(noisy))
    np.testing.assert_array_equal(copy[1]["audio"], noisy[1]["audio"])


def test_collate_raw_makes_lists(env):
    root, manifest, *_ = env
    train = TrainSet(manifest, root)
    batch = collate_raw([train[0], train[1]])
    assert len(batch["audio"]) == 2 and batch["text"] == [train[0]["text"], train[1]["text"]]
