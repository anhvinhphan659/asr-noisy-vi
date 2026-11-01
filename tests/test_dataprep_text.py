import unicodedata

import pytest

from dataprep import has_digit, normalize_text, verbalize_number, verbalize_numbers


def test_normalize_lowercases_and_strips_punctuation():
    assert normalize_text("  KHÁCH SẠN, ĐẸP!  ") == "khách sạn đẹp"


def test_normalize_composes_unicode():
    decomposed = unicodedata.normalize("NFD", "Tiếng Việt")
    assert normalize_text(decomposed) == "tiếng việt"


def test_normalize_handles_empty_and_none():
    assert normalize_text("") == ""
    assert normalize_text(None) == ""


@pytest.mark.parametrize("new, old", [
    ("hoà bình", "hòa bình"),
    ("khoẻ", "khỏe"),
    ("thuỷ", "thủy"),
    ("loá", "lóa"),
])
def test_normalize_unifies_tone_placement(new, old):
    assert normalize_text(new) == normalize_text(old) == old


@pytest.mark.parametrize("word", ["quý", "hoàng", "khuyến", "hoài", "quà"])
def test_normalize_keeps_tone_where_spelling_is_fixed(word):
    assert normalize_text(word) == word


# --- Numbers ------------------------------------------------------------------------------

@pytest.mark.parametrize("digits, words", [
    ("0", "không"),
    ("5", "năm"),
    ("10", "mười"),
    ("11", "mười một"),
    ("15", "mười lăm"),
    ("20", "hai mươi"),
    ("21", "hai mươi mốt"),
    ("24", "hai mươi bốn"),
    ("25", "hai mươi lăm"),
    ("100", "một trăm"),
    ("105", "một trăm linh năm"),
    ("115", "một trăm mười lăm"),
    ("991", "chín trăm chín mươi mốt"),
    ("1000", "một nghìn"),
    ("1005", "một nghìn không trăm linh năm"),
    ("1967", "một nghìn chín trăm sáu mươi bảy"),
    ("2011", "hai nghìn không trăm mười một"),
    ("4892", "bốn nghìn tám trăm chín mươi hai"),
    ("19500", "mười chín nghìn năm trăm"),
    ("1000000", "một triệu"),
    ("1000001", "một triệu không trăm linh một"),
    ("2500000000", "hai tỷ năm trăm triệu"),
    ("0912", "không chín một hai"),
])
def test_verbalize_number(digits, words):
    assert verbalize_number(digits) == words


def test_verbalize_number_rejects_non_digits():
    with pytest.raises(ValueError):
        verbalize_number("12a")


@pytest.mark.parametrize("raw, expected", [
    # FLEURS
    ("trong năm 2015 hoặc 2016", "trong năm hai nghìn không trăm mười lăm hoặc hai nghìn "
                                 "không trăm mười sáu"),
    ("có 9.000 học viên", "có chín nghìn học viên"),
    ("diện tích trên 19.500 km²", "diện tích trên mười chín nghìn năm trăm ki lô mét vuông"),
    ("cao 4892 m.", "cao bốn nghìn tám trăm chín mươi hai mét"),
    ("cách biệt chỉ một điểm, 21 - 20,", "cách biệt chỉ một điểm hai mươi mốt hai mươi"),
    # VIVOS
    ("CHỈ CÒN CÓ 4 NGÀY", "chỉ còn có bốn ngày"),
    # VietMed
    ("8 milimet", "tám milimet"),
    ("sau khi 35 tuổi", "sau khi ba mươi lăm tuổi"),
    ("giảm 20% cân nặng", "giảm hai mươi phần trăm cân nặng"),
    ("ba phẩy năm", "ba phẩy năm"),
    ("3,5 kg", "ba phẩy năm ki lô gam"),
])
def test_normalize_writes_numbers_as_words(raw, expected):
    assert normalize_text(raw) == expected


@pytest.mark.parametrize("code", ["covid19", "h1n1", "hba1c", "glp1"])
def test_codes_with_digits_are_kept(code):
    assert normalize_text(f"bệnh {code} rồi") == f"bệnh {code} rồi"


def test_unit_letters_inside_words_are_not_read_as_units():
    assert normalize_text("5 mét") == "năm mét"
    assert normalize_text("5 mẹ") == "năm mẹ"


def test_digits_and_words_give_the_same_text():
    assert normalize_text("Năm 2011.") == normalize_text("năm hai nghìn không trăm mười một")


@pytest.mark.parametrize("variant, canonical", [
    ("hai ngàn", "hai nghìn"),
    ("một trăm lẻ một", "một trăm linh một"),
    ("hai nghìn lẻ năm", "hai nghìn linh năm"),
    ("hai mươi tư", "hai mươi bốn"),
    ("hai mươi nhăm", "hai mươi lăm"),
    ("ba mươi một ngày", "ba mươi mốt ngày"),
])
def test_normalize_unifies_number_word_variants(variant, canonical):
    assert normalize_text(variant) == canonical


@pytest.mark.parametrize("text", ["số lẻ", "thứ tư", "hai mươi năm qua", "hàng ngàn người"])
def test_variants_are_only_changed_in_number_context(text):
    expected = {"hàng ngàn người": "hàng nghìn người"}.get(text, text)
    assert normalize_text(text) == expected


def test_verbalize_numbers_keeps_other_text():
    assert verbalize_numbers("không có số") == "không có số"


def test_has_digit():
    assert has_digit("năm 2011")
    assert not has_digit("hai nghìn")
    assert not has_digit(normalize_text("năm 2011"))
