"""Join model outputs with references, count errors per utterance, and aggregate them."""

from pathlib import Path

import pandas as pd

from dataprep.text import normalize_text

from .metrics import char_counts, error_rate, word_counts

# One row per model output. ``condition`` is the plan's condition: "clean", "traffic_5dB"...
PREDICTION_COLUMNS = ["utt_id", "condition", "model", "variant", "hyp"]
CONDITION_KEYS = ["dataset", "noise_type", "snr_db"]
SYSTEM_KEYS = ["model", "variant"]
DEFAULT_BY = SYSTEM_KEYS + CONDITION_KEYS


def load_predictions(paths):
    """Read and stack prediction CSVs; reject missing columns and duplicated outputs."""
    frames = [pd.read_csv(p, dtype={"hyp": str}, keep_default_na=False) for p in map(Path, paths)]
    predictions = pd.concat(frames, ignore_index=True)
    missing = set(PREDICTION_COLUMNS) - set(predictions.columns)
    if missing:
        raise ValueError(f"prediction files lack columns {sorted(missing)}")
    key = ["utt_id", "condition", "model", "variant"]
    duplicated = predictions.duplicated(key)
    if duplicated.any():
        raise ValueError(f"{duplicated.sum()} duplicated predictions, e.g.\n"
                         f"{predictions.loc[duplicated, key].head()}")
    return predictions


def missing_predictions(predictions, plan):
    """Plan rows with no prediction, per system. Empty when every system covered the plan."""
    rows = []
    expected = plan[["utt_id", "condition"]].drop_duplicates()
    for (model, variant), got in predictions.groupby(SYSTEM_KEYS):
        merged = expected.merge(got[["utt_id", "condition"]], how="left", indicator=True)
        lost = merged[merged["_merge"] == "left_only"].drop(columns="_merge")
        rows.append(lost.assign(model=model, variant=variant))
    return pd.concat(rows, ignore_index=True) if rows else expected.iloc[0:0]


def score(predictions, plan, manifest):
    """Per-utterance error counts.

    ``plan`` gives each (utt_id, condition) its dataset, noise type and SNR; ``manifest``
    gives the reference text. Reference and hypothesis go through the same
    ``normalize_text``. Predictions outside the plan are an error.
    """
    conditions = plan[["utt_id", "condition", "dataset", "noise_type", "snr_db"]].drop_duplicates()
    scored = predictions.merge(conditions, on=["utt_id", "condition"], how="left",
                               validate="many_to_one", indicator=True)
    outside = scored["_merge"] == "left_only"
    if outside.any():
        raise ValueError(f"{outside.sum()} predictions are not in the plan")
    scored = scored.drop(columns="_merge").merge(manifest[["utt_id", "text"]], on="utt_id",
                                                 how="left", validate="many_to_one")
    scored["ref"] = scored["text"].map(normalize_text)
    scored["hyp_norm"] = scored["hyp"].fillna("").map(normalize_text)
    words = [word_counts(r, h) for r, h in zip(scored["ref"], scored["hyp_norm"])]
    chars = [char_counts(r, h) for r, h in zip(scored["ref"], scored["hyp_norm"])]
    for prefix, counts in [("w", words), ("c", chars)]:
        frame = pd.DataFrame(counts)
        for col in ["sub", "del", "ins", "n_ref"]:
            scored[f"{prefix}_{col}"] = frame[col].to_numpy()
    return scored.drop(columns="text")


def _rates(group):
    w = group[["w_sub", "w_del", "w_ins", "w_n_ref"]].sum()
    c = group[["c_sub", "c_del", "c_ins", "c_n_ref"]].sum()
    return pd.Series({
        "utterances": len(group),
        "words": int(w["w_n_ref"]),
        "wer": round(error_rate(w["w_sub"], w["w_del"], w["w_ins"], w["w_n_ref"]), 2),
        "cer": round(error_rate(c["c_sub"], c["c_del"], c["c_ins"], c["c_n_ref"]), 2),
        "sub_pct": round(100 * w["w_sub"] / w["w_n_ref"], 2),
        "del_pct": round(100 * w["w_del"] / w["w_n_ref"], 2),
        "ins_pct": round(100 * w["w_ins"] / w["w_n_ref"], 2),
    })


def summarize(scored, by=DEFAULT_BY):
    """Corpus-level WER / CER (percent) per group: errors are summed before dividing.

    ``sub_pct``, ``del_pct``, ``ins_pct`` split the WER into its three error kinds.
    """
    groups = scored.groupby(list(by), dropna=False, sort=True)
    return groups.apply(_rates, include_groups=False).reset_index()


def wer_table(scored, metric="wer"):
    """Report table: one row per dataset × noise type × SNR, one column per model / variant."""
    summary = summarize(scored)
    table = summary.pivot_table(index=CONDITION_KEYS, columns=SYSTEM_KEYS, values=metric,
                                dropna=False)
    return table.dropna(how="all")
