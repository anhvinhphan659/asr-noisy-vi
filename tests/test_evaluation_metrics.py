import pytest

from evaluation import char_counts, edit_counts, error_rate, word_counts


def test_word_counts_example():
    # "trời" -> "chời" is a substitution, "quá" is deleted.
    counts = word_counts("hôm nay trời đẹp quá", "hôm nay chời đẹp")
    assert counts == {"sub": 1, "del": 1, "ins": 0, "n_ref": 5}
    assert error_rate(counts["sub"], counts["del"], counts["ins"], counts["n_ref"]) == 40.0


def test_insertions_can_push_wer_above_100():
    counts = word_counts("xin chào", "a b c d e")
    assert error_rate(counts["sub"], counts["del"], counts["ins"], counts["n_ref"]) > 100


def test_perfect_and_empty_hypothesis():
    assert word_counts("một hai", "một hai") == {"sub": 0, "del": 0, "ins": 0, "n_ref": 2}
    assert word_counts("một hai", "") == {"sub": 0, "del": 2, "ins": 0, "n_ref": 2}


def test_char_counts_penalize_a_wrong_tone_lightly():
    assert char_counts("trời", "trơi") == {"sub": 1, "del": 0, "ins": 0, "n_ref": 4}


def test_edit_counts_on_token_lists():
    assert edit_counts(["a", "b"], ["a", "x", "b"])["ins"] == 1


def test_error_rate_rejects_empty_reference():
    with pytest.raises(ValueError):
        error_rate(0, 0, 1, 0)
