"""Error counts between a reference and a hypothesis, at word and character level."""

from rapidfuzz.distance import Levenshtein


def edit_counts(ref, hyp):
    """Minimum substitutions, deletions and insertions turning sequence ``ref`` into ``hyp``.

    Returns ``{"sub", "del", "ins", "n_ref"}``. A deletion is a reference token the
    hypothesis misses; an insertion is an extra hypothesis token.
    """
    counts = {"sub": 0, "del": 0, "ins": 0}
    for op in Levenshtein.editops(ref, hyp):
        if op.tag == "replace":
            counts["sub"] += 1
        elif op.tag == "delete":
            counts["del"] += 1
        else:
            counts["ins"] += 1
    counts["n_ref"] = len(ref)
    return counts


def word_counts(ref_text, hyp_text):
    """Edit counts over words (whitespace tokens) of two normalized texts."""
    return edit_counts(ref_text.split(), hyp_text.split())


def char_counts(ref_text, hyp_text):
    """Edit counts over characters of two normalized texts, spaces included (as in jiwer)."""
    return edit_counts(list(ref_text), list(hyp_text))


def error_rate(sub, dele, ins, n_ref):
    """(S + D + I) / N in percent. Inputs may be totals over many utterances."""
    if n_ref == 0:
        raise ValueError("reference is empty; error rate is undefined")
    return 100.0 * (sub + dele + ins) / n_ref
