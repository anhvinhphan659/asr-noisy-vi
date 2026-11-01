"""Bootstrap confidence intervals: is a WER (or a WER difference) reliable or test-set luck?"""

import numpy as np
import pandas as pd

from .scoring import CONDITION_KEYS, SYSTEM_KEYS

N_BOOT = 1000
ALPHA = 0.05


def _corpus_rate(errors, n_ref, idx):
    """Corpus error rate (percent) for each bootstrap sample of utterance indices ``idx``."""
    return 100.0 * errors[idx].sum(axis=1) / n_ref[idx].sum(axis=1)


def _counts(group, metric, suffix=""):
    """Per-utterance error totals and reference lengths for ``metric`` ("wer" or "cer")."""
    p = "w" if metric == "wer" else "c"
    errors = sum(group[f"{p}_{kind}{suffix}"] for kind in ("sub", "del", "ins"))
    return errors.to_numpy(float), group[f"{p}_n_ref{suffix}"].to_numpy(float)


def bootstrap_ci(scored, by=SYSTEM_KEYS + CONDITION_KEYS, metric="wer", n_boot=N_BOOT,
                 alpha=ALPHA, seed=42):
    """WER (or CER) per group with a ``1 - alpha`` percentile interval.

    Utterances are resampled with replacement ``n_boot`` times; the metric is recomputed
    at corpus level on each resample.
    """
    rows = []
    for key, group in scored.groupby(list(by), dropna=False, sort=True):
        errors, n_ref = _counts(group, metric)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(group), size=(n_boot, len(group)))
        samples = _corpus_rate(errors, n_ref, idx)
        rows.append({**dict(zip(by, key)),
                     metric: round(100.0 * errors.sum() / n_ref.sum(), 2),
                     "ci_low": round(float(np.quantile(samples, alpha / 2)), 2),
                     "ci_high": round(float(np.quantile(samples, 1 - alpha / 2)), 2)})
    return pd.DataFrame(rows)


def compare(scored, system_a, system_b, by=CONDITION_KEYS, metric="wer", n_boot=N_BOOT,
            alpha=ALPHA, seed=42):
    """Paired bootstrap of ``metric(b) - metric(a)`` per group.

    ``system_a`` / ``system_b`` are ``(model, variant)`` tuples, e.g.
    ``("qwen3-asr-0.6b", "zero-shot")`` and ``("qwen3-asr-0.6b", "ft-noisy")``. Only
    utterances scored for both systems are used, and both are resampled with the same
    indices. ``significant`` is True when the interval excludes 0; a negative ``diff``
    means system b makes fewer errors.
    """
    def pick(system):
        model, variant = system
        part = scored[(scored["model"] == model) & (scored["variant"] == variant)]
        if part.empty:
            raise ValueError(f"no scored rows for {system}")
        return part

    keys = ["utt_id", "condition"]
    pair = pick(system_a).merge(pick(system_b), on=keys + list(by), suffixes=("_a", "_b"))
    rows = []
    for key, group in pair.groupby(list(by), dropna=False, sort=True):
        ea, na = _counts(group, metric, "_a")
        eb, nb = _counts(group, metric, "_b")
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(group), size=(n_boot, len(group)))
        diffs = _corpus_rate(eb, nb, idx) - _corpus_rate(ea, na, idx)
        low = float(np.quantile(diffs, alpha / 2))
        high = float(np.quantile(diffs, 1 - alpha / 2))
        rate_a = 100.0 * ea.sum() / na.sum()
        rate_b = 100.0 * eb.sum() / nb.sum()
        rows.append({**dict(zip(by, key)), "utterances": len(group),
                     f"{metric}_a": round(rate_a, 2), f"{metric}_b": round(rate_b, 2),
                     "diff": round(rate_b - rate_a, 2), "ci_low": round(low, 2),
                     "ci_high": round(high, 2), "significant": low > 0 or high < 0})
    return pd.DataFrame(rows)
