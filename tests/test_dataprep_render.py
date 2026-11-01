from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from core import SAMPLE_RATE, measure_snr
from dataprep import NoiseBank, build_test_plan, render, write_noisy_set


@pytest.fixture
def setup(tmp_path, speech):
    rng = np.random.default_rng(0)
    rows = []
    for prefix, kind, n in [("t", "traffic", 4), ("v", "voice", 10)]:
        for i in range(n):
            path = tmp_path / "noise" / f"{prefix}{i}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, 0.1 * rng.standard_normal(int(1.5 * SAMPLE_RATE)), SAMPLE_RATE)
            rows.append({"noise_id": f"{prefix}{i}", "kind": kind, "path": str(path),
                         "duration_s": 1.5, "split": "test" if i % 2 else "train"})
    pool = pd.DataFrame(rows)
    pools = {"traffic": pool[pool["kind"] == "traffic"], "babble": pool[pool["kind"] == "voice"]}

    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    manifest_rows = []
    for i, sr in enumerate([16000, 8000]):
        sf.write(clean_dir / f"u{i}.wav", speech, SAMPLE_RATE, subtype="PCM_16")
        manifest_rows.append({"utt_id": f"u{i}", "dataset": "d", "split": "test",
                              "path": f"clean/u{i}.wav", "duration_s": len(speech) / SAMPLE_RATE,
                              "orig_sr": sr})
    manifest = pd.DataFrame(manifest_rows)
    return tmp_path, pools, manifest, NoiseBank(*pools.values())


def test_render_hits_target_snr_and_loops_short_noise(setup, speech):
    _, pools, manifest, bank = setup
    plan = build_test_plan(manifest, pools)
    # speech is 2 s, noise files are 1.5 s, so every noise is looped.
    for _, spec in plan[plan["noise_type"] != "clean"].iterrows():
        out = render(speech, spec, bank)
        assert len(out) == len(speech)
        assert abs(measure_snr(speech, out) - spec["snr_db"]) < 0.5


def test_render_is_deterministic_and_clean_is_identity(setup, speech):
    _, pools, manifest, bank = setup
    plan = build_test_plan(manifest, pools)
    spec = plan[plan["noise_type"] == "babble"].iloc[0]
    np.testing.assert_array_equal(render(speech, spec, bank), render(speech, spec, bank))
    clean = plan[plan["noise_type"] == "clean"].iloc[0]
    np.testing.assert_array_equal(render(speech, clean, bank), speech)


def test_babble_reads_voices_without_caching_them(setup, speech):
    _, pools, manifest, bank = setup
    plan = build_test_plan(manifest, pools)
    for _, spec in plan[plan["noise_type"] == "babble"].iterrows():
        render(speech, spec, bank)
    assert not any(noise_id.startswith("v") for noise_id in bank._cache)
    for _, spec in plan[plan["noise_type"] == "traffic"].iterrows():
        render(speech, spec, bank)
    assert any(noise_id.startswith("t") for noise_id in bank._cache)


def test_noise_bank_resolves_relative_paths(setup, speech):
    root, pools, manifest, _ = setup
    relative = {k: p.assign(path=[str(Path(x).relative_to(root)) for x in p["path"]])
                for k, p in pools.items()}
    bank = NoiseBank(*relative.values(), data_root=root)
    plan = build_test_plan(manifest, relative)
    for _, spec in plan[plan["noise_type"] != "clean"].iterrows():
        render(speech, spec, bank)


def test_render_accepts_rows_read_back_from_csv(setup, speech, tmp_path):
    _, pools, manifest, bank = setup
    build_test_plan(manifest, pools).to_csv(tmp_path / "plan.csv", index=False)
    plan = pd.read_csv(tmp_path / "plan.csv")
    for _, spec in plan.iterrows():
        render(speech, spec, bank)


def test_write_noisy_set_writes_files_and_measures_snr(setup):
    root, pools, manifest, bank = setup
    plan = build_test_plan(manifest, pools)
    out = write_noisy_set(plan, bank, root)
    noisy = out[out["noise_type"] != "clean"]
    assert all((root / p).exists() for p in noisy["path"])
    assert ((noisy["snr_measured"] - noisy["snr_db"]).abs() < 0.5).all()
    assert out.loc[out["noise_type"] == "clean", "snr_measured"].isna().all()
    # A second run reuses the files and gives the same measurements.
    again = write_noisy_set(plan, bank, root)
    pd.testing.assert_series_equal(out["snr_measured"], again["snr_measured"])
