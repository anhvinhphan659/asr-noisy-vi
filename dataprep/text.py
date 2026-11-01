import re
import unicodedata

_NON_WORD = re.compile(r"[^\w\s]|_")
_DIGIT = re.compile(r"\d")

# --- Tone placement -----------------------------------------------------------------------

# Tone placement in the open syllables "oa", "oe", "uy" has two accepted spellings
# ("hoà" / "hòa", "thuỷ" / "thủy"). Both are mapped to the tone on the first vowel so
# that a model is not penalized for choosing the other convention. "quý" is left alone:
# after "q", the "u" is part of the consonant and the tone always sits on "y".
_TONES = {
    "a": "àáảãạ",
    "e": "èéẻẽẹ",
    "y": "ỳýỷỹỵ",
    "o": "òóỏõọ",
    "u": "ùúủũụ",
}
_TONE_MOVES = {}
for _first, _second in [("o", "a"), ("o", "e"), ("u", "y")]:
    for _i, _toned in enumerate(_TONES[_second]):
        _TONE_MOVES[_first + _toned] = _TONES[_first][_i] + _second
_TONE_PLACEMENT = re.compile(
    r"(?<!q)(" + "|".join(sorted(_TONE_MOVES)) + r")(?!\w)"
)

# --- Numbers ------------------------------------------------------------------------------

_DIGIT_WORDS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
_SCALES = ["", "nghìn", "triệu", "tỷ", "nghìn tỷ", "triệu tỷ"]
_MAX_DIGITS = 3 * len(_SCALES)  # longer digit strings are read digit by digit

# Units that commonly follow a number, written as they are read.
_UNITS = {
    "km²": "ki lô mét vuông",
    "m²": "mét vuông",
    "km": "ki lô mét",
    "cm": "xen ti mét",
    "mm": "mi li mét",
    "kg": "ki lô gam",
    "m": "mét",
    "%": "phần trăm",
}
_ALNUM = r"[^\W_]"  # a letter or a digit
# A whole number not glued to letters (codes such as "h1n1", "covid19" are left as they are).
# Vietnamese writing: "." separates thousands ("9.000"), "," marks decimals ("3,5").
_NUMBER = re.compile(
    rf"(?<!{_ALNUM})"
    r"(?P<int>\d{1,3}(?:\.\d{3})+(?!\d)|\d+)"
    r"(?:[.,](?P<frac>\d+))?"
    rf"(?:\s?(?P<unit>{'|'.join(map(re.escape, _UNITS))})(?!{_ALNUM})|(?!{_ALNUM}))"
)

# Regional and spoken variants of number words, mapped to the form verbalize_number uses.
_VARIANTS = [
    (re.compile(r"\bngàn\b"), "nghìn"),
    (re.compile(r"\b(trăm|nghìn) lẻ\b"), r"\1 linh"),
    (re.compile(r"\bnhăm\b"), "lăm"),
    (re.compile(r"\bmươi tư\b"), "mươi bốn"),
    (re.compile(r"\bmươi một\b"), "mươi mốt"),
]


def _read_digits(digits):
    return " ".join(_DIGIT_WORDS[int(d)] for d in digits)


def _read_below_100(n, after_hundreds):
    """1–99. ``after_hundreds``: a tens digit of 0 is read "linh" (105 = một trăm linh năm)."""
    tens, units = divmod(n, 10)
    if tens == 0:
        return f"linh {_DIGIT_WORDS[units]}" if after_hundreds else _DIGIT_WORDS[units]
    head = "mười" if tens == 1 else f"{_DIGIT_WORDS[tens]} mươi"
    if units == 0:
        return head
    if units == 5:
        tail = "lăm"
    elif units == 1 and tens >= 2:
        tail = "mốt"
    else:
        tail = _DIGIT_WORDS[units]
    return f"{head} {tail}"


def _read_group(n, full):
    """0–999. ``full``: a non-leading group always says its hundreds (2011 = ... không trăm mười một)."""
    hundreds, rest = divmod(n, 100)
    words = []
    if full or hundreds:
        words.append(f"{_DIGIT_WORDS[hundreds]} trăm")
    if rest:
        words.append(_read_below_100(rest, after_hundreds=bool(words)))
    return " ".join(words)


def verbalize_number(digits):
    """Read a string of digits as Vietnamese words, the way it is spoken.

    ``"2011"`` → ``"hai nghìn không trăm mười một"``, ``"105"`` → ``"một trăm linh năm"``,
    ``"21"`` → ``"hai mươi mốt"``. A string with a leading zero (``"0912"``) or longer
    than 18 digits is read digit by digit.
    """
    if not digits.isdigit():
        raise ValueError(f"not a digit string: {digits!r}")
    if (len(digits) > 1 and digits[0] == "0") or len(digits) > _MAX_DIGITS:
        return _read_digits(digits)
    n = int(digits)
    if n == 0:
        return "không"
    groups = []
    while n:
        n, group = divmod(n, 1000)
        groups.append(group)
    words = []
    for i in range(len(groups) - 1, -1, -1):
        if groups[i] == 0:
            continue
        words.append(_read_group(groups[i], full=i < len(groups) - 1))
        if _SCALES[i]:
            words.append(_SCALES[i])
    return " ".join(words)


def _verbalize_match(match):
    words = [verbalize_number(match["int"].replace(".", ""))]
    frac = match["frac"]
    if frac is not None:
        words += ["phẩy", verbalize_number(frac)]
    if match["unit"]:
        words.append(_UNITS[match["unit"]])
    return f" {' '.join(words)} "


def verbalize_numbers(text):
    """Write the numbers of a (lowercased) text as words, with their unit if any.

    ``"9.000 học viên"`` → ``"chín nghìn học viên"``, ``"19.500 km²"`` →
    ``"mười chín nghìn năm trăm ki lô mét vuông"``, ``"20%"`` → ``"hai mươi phần trăm"``.
    Digits glued to letters (``"h1n1"``, ``"covid19"``) are left as they are.
    """
    return _NUMBER.sub(_verbalize_match, text)


# --- Normalization ------------------------------------------------------------------------

def normalize_text(text):
    """Normalize a transcript for training and scoring.

    Unicode NFC, lowercase, numbers written as words, punctuation replaced by spaces,
    whitespace collapsed, tone placement unified in word-final "oa", "oe", "uy", and
    regional variants of number words unified ("ngàn" → "nghìn", "lẻ" → "linh"...).
    """
    text = unicodedata.normalize("NFC", text or "").lower()
    text = verbalize_numbers(text)
    text = _NON_WORD.sub(" ", text)
    text = _TONE_PLACEMENT.sub(lambda m: _TONE_MOVES[m.group(1)], text)
    text = " ".join(text.split())
    for pattern, replacement in _VARIANTS:
        text = pattern.sub(replacement, text)
    return text


def has_digit(text):
    """True if ``text`` contains a digit. Apply it to the raw transcript: normalized text
    has its numbers written as words."""
    return bool(_DIGIT.search(text or ""))
