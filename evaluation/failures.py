"""Utterances where the model failed as a whole (looping, cut short, wrong language).

WER counts errors word by word. Under heavy babble, a few utterances fail as a whole and
can decide a condition's corpus WER; these flags separate them from ordinary misrecognition.
"""

import re
import unicodedata

import pandas as pd

from .scoring import DEFAULT_BY

FAILURE_KINDS = ["looping", "truncated", "off_language"]

LOOP_LENGTH_RATIO = 2.0   # a looping output is longer than this × the reference
LOOP_MAX_PHRASE = 20      # longest repeated phrase looked for, in words
LOOP_MIN_REPEATS = 3      # consecutive repeats of one phrase
TRUNCATED_RATIO = 0.5     # a truncated output is shorter than this × the reference
OFF_LANGUAGE_MIN_WORDS = 3

# Letters that only Vietnamese uses among the Latin-script languages of the noise.
_VI_LETTER = re.compile("[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợ"
                        "ùúủũụưừứửữựỳýỷỹỵđ]")


def has_repeated_phrase(words, max_phrase=LOOP_MAX_PHRASE, min_repeats=LOOP_MIN_REPEATS):
    """True if some phrase of 1 to ``max_phrase`` words occurs ``min_repeats`` times in a row."""
    words = list(words)
    for n in range(1, max_phrase + 1):
        for i in range(len(words) - min_repeats * n + 1):
            phrase = words[i:i + n]
            if all(words[i + k * n:i + (k + 1) * n] == phrase for k in range(1, min_repeats)):
                return True
    return False


def _has_vietnamese_letter(text):
    return bool(_VI_LETTER.search(unicodedata.normalize("NFC", text or "").lower()))


def flag_failures(scored):
    """Copy of ``scored`` (from ``score``) with one boolean column per failure kind.

    - ``looping``: the output is more than ``LOOP_LENGTH_RATIO`` × the reference length and
      repeats a phrase of up to ``LOOP_MAX_PHRASE`` words ``LOOP_MIN_REPEATS`` times in a row.
    - ``truncated``: the output has fewer than ``TRUNCATED_RATIO`` × the reference words.
    - ``off_language``: the reference has Vietnamese letters, the normalized output has at
      least ``OFF_LANGUAGE_MIN_WORDS`` words and none (e.g. English babble transcribed in
      English). The normalized output is used so that digits ("60, 61"), written as
      Vietnamese words by ``normalize_text``, do not count as another language.
    - ``failed``: any of the three.
    """
    out = scored.copy()
    hyp_words = out["hyp_norm"].str.split()
    n_hyp = hyp_words.str.len()
    n_ref = out["w_n_ref"]
    long_output = n_hyp > LOOP_LENGTH_RATIO * n_ref
    out["looping"] = False
    # The phrase search is slow; only long outputs can be looping.
    out.loc[long_output, "looping"] = hyp_words[long_output].map(has_repeated_phrase).astype(bool)
    out["truncated"] = n_hyp < TRUNCATED_RATIO * n_ref
    out["off_language"] = ((n_hyp >= OFF_LANGUAGE_MIN_WORDS)
                           & out["ref"].map(_has_vietnamese_letter)
                           & ~out["hyp_norm"].map(_has_vietnamese_letter))
    out["failed"] = out[FAILURE_KINDS].any(axis=1)
    return out


def _failure_rates(group):
    errors = group[["w_sub", "w_del", "w_ins"]].sum(axis=1)
    words = group["w_n_ref"].sum()
    kept = ~group["failed"]
    row = {"utterances": len(group)}
    for kind in FAILURE_KINDS + ["failed"]:
        row[kind] = int(group[kind].sum())
        row[f"{kind}_pct"] = round(100 * group[kind].mean(), 2)
    row["wer"] = round(100 * errors.sum() / words, 2)
    row["failed_wer_pts"] = round(100 * errors[group["failed"]].sum() / words, 2)
    kept_words = group.loc[kept, "w_n_ref"].sum()
    row["wer_without_failed"] = (round(100 * errors[kept].sum() / kept_words, 2)
                                 if kept_words else float("nan"))
    return pd.Series(row)


def failure_summary(scored, by=DEFAULT_BY):
    """Failed utterances per group, and how much of the corpus WER they account for.

    Gives the count and percentage of utterances of each kind, ``wer`` (as in
    ``summarize``), ``failed_wer_pts`` (WER points from failed utterances: their errors
    divided by all reference words of the group, so ``wer - failed_wer_pts`` is what the
    other utterances contribute) and ``wer_without_failed`` (corpus WER of the others).
    ``scored`` comes from ``score``; flags are added if missing.
    """
    if "failed" not in scored.columns:
        scored = flag_failures(scored)
    groups = scored.groupby(list(by), dropna=False, sort=True)
    summary = groups.apply(_failure_rates, include_groups=False).reset_index()
    counts = ["utterances"] + FAILURE_KINDS + ["failed"]
    return summary.astype(dict.fromkeys(counts, int))
