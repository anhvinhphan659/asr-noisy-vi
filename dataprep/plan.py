"""Mixing plans: which noise, where in it, and at which SNR, decided once and saved as a table.

A plan row (a "spec") is enough to rebuild the noisy audio exactly, without storing it:

| Column | Meaning |
|---|---|
| ``noise_type`` | ``clean``, ``traffic`` or ``babble`` |
| ``snr_db`` | Target SNR (empty for clean) |
| ``noise_id`` | Traffic clip (empty otherwise) |
| ``offset_s`` | Where to start reading the traffic clip |
| ``voices`` | Babble voices as ``noise_id@offset_s;noise_id@offset_s;...`` |
| ``band_sr`` | Original sample rate of narrow-band speech (e.g. 8000); noise is limited to it |
"""

import numpy as np
import pandas as pd

from core import SAMPLE_RATE

from .noise import random_offset_s

TEST_SNRS = (20, 10, 5, 0)
TRAIN_SNR_RANGE = (0.0, 20.0)
CLEAN_PROB = 0.25
BABBLE_VOICES = (3, 5)

SPEC_COLUMNS = ["noise_type", "snr_db", "noise_id", "offset_s", "voices", "band_sr"]


def _split_pools(pools, noise_split):
    """``{"traffic": pool, "babble": pool}`` restricted to one noise split, sorted for determinism."""
    out = {}
    for noise_type, pool in pools.items():
        part = pool[pool["split"] == noise_split].sort_values("noise_id").reset_index(drop=True)
        if part.empty:
            raise ValueError(f"no {noise_type} noise in split {noise_split!r}")
        out[noise_type] = part
    return out


def _band_sr(orig_sr):
    return int(orig_sr) if orig_sr and int(orig_sr) < SAMPLE_RATE else None


def sample_spec(rng, noise_type, pool, speech_duration_s, babble_voices=BABBLE_VOICES):
    """Draw the noise part of a spec (no SNR) for one utterance."""
    if noise_type == "traffic":
        row = pool.iloc[int(rng.integers(len(pool)))]
        return {"noise_id": row["noise_id"],
                "offset_s": random_offset_s(rng, row["duration_s"], speech_duration_s),
                "voices": None}
    if noise_type == "babble":
        k = int(rng.integers(babble_voices[0], babble_voices[1] + 1))
        idx = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
        voices = [f"{pool.iloc[i]['noise_id']}@"
                  f"{random_offset_s(rng, pool.iloc[i]['duration_s'], speech_duration_s)}"
                  for i in sorted(idx)]
        return {"noise_id": None, "offset_s": None, "voices": ";".join(voices)}
    raise ValueError(f"unknown noise_type {noise_type!r}")


def condition_name(noise_type, snr_db):
    return "clean" if noise_type == "clean" else f"{noise_type}_{snr_db:g}dB"


def build_test_plan(manifest, pools, snrs=TEST_SNRS, seed=42, noise_split="test"):
    """Full grid for evaluation: every utterance × (clean + each noise type × each SNR).

    For one utterance and one noise type, the **same noise segment** is used at every SNR,
    so differences between SNR levels come from the SNR alone.
    """
    rng = np.random.default_rng(seed)
    pools = _split_pools(pools, noise_split)
    rows = []
    for utt in manifest.itertuples(index=False):
        base = {"utt_id": utt.utt_id, "dataset": utt.dataset, "split": utt.split,
                "clean_path": utt.path, "band_sr": _band_sr(utt.orig_sr)}
        rows.append({**base, "condition": "clean", "noise_type": "clean", "snr_db": None,
                     "noise_id": None, "offset_s": None, "voices": None,
                     "path": utt.path})
        for noise_type, pool in pools.items():
            spec = sample_spec(rng, noise_type, pool, utt.duration_s)
            for snr in snrs:
                cond = condition_name(noise_type, snr)
                rows.append({**base, "condition": cond, "noise_type": noise_type,
                             "snr_db": float(snr), **spec,
                             "path": f"noisy/{utt.dataset}/{utt.split}/{cond}/{utt.utt_id}.wav"})
    return pd.DataFrame(rows)


def build_dev_plan(manifest, pools, snrs=TEST_SNRS, clean_prob=CLEAN_PROB, seed=42,
                   noise_split="train"):
    """One fixed condition per utterance, for choosing checkpoints during fine-tuning.

    Noise comes from the **train** pool: test noise is never seen before the final test.
    """
    rng = np.random.default_rng(seed)
    pools = _split_pools(pools, noise_split)
    types = list(pools)
    rows = []
    for utt in manifest.itertuples(index=False):
        base = {"utt_id": utt.utt_id, "dataset": utt.dataset, "split": utt.split,
                "clean_path": utt.path, "band_sr": _band_sr(utt.orig_sr)}
        if rng.random() < clean_prob:
            rows.append({**base, "condition": "clean", "noise_type": "clean", "snr_db": None,
                         "noise_id": None, "offset_s": None, "voices": None, "path": utt.path})
            continue
        noise_type = types[int(rng.integers(len(types)))]
        snr = float(snrs[int(rng.integers(len(snrs)))])
        spec = sample_spec(rng, noise_type, pools[noise_type], utt.duration_s)
        cond = condition_name(noise_type, snr)
        rows.append({**base, "condition": cond, "noise_type": noise_type, "snr_db": snr, **spec,
                     "path": f"noisy/{utt.dataset}/{utt.split}/{cond}/{utt.utt_id}.wav"})
    return pd.DataFrame(rows)


def build_train_schedule(manifest, pools, epochs, clean_prob=CLEAN_PROB,
                         snr_range=TRAIN_SNR_RANGE, seed=42, noise_split="train"):
    """Online-mixing schedule: one spec per utterance per epoch, no audio written.

    Every model is trained from the same schedule, so all models see exactly the same
    noisy samples. SNRs are continuous in ``snr_range``, not the test levels.
    """
    rng = np.random.default_rng(seed)
    pools = _split_pools(pools, noise_split)
    types = list(pools)
    rows = []
    for epoch in range(1, epochs + 1):
        for utt in manifest.itertuples(index=False):
            base = {"epoch": epoch, "utt_id": utt.utt_id, "band_sr": _band_sr(utt.orig_sr)}
            if rng.random() < clean_prob:
                rows.append({**base, "noise_type": "clean", "snr_db": None, "noise_id": None,
                             "offset_s": None, "voices": None})
                continue
            noise_type = types[int(rng.integers(len(types)))]
            snr = round(float(rng.uniform(*snr_range)), 2)
            spec = sample_spec(rng, noise_type, pools[noise_type], utt.duration_s)
            rows.append({**base, "noise_type": noise_type, "snr_db": snr, **spec})
    return pd.DataFrame(rows)
