"""Variant generators for guard fuzzing.

Two kinds:
- invariance ("must stay the same"): the text means the same thing to a human, so the guard's
  decision must not change (Unicode forms, invisible characters, case, spacing, look-alike letters).
- sensitivity ("must differ"): the text means something different, so a comparator must not call
  it equal (code suffixes, one changed digit).
Synonyms are reported separately: whether "Send" should be treated like "Submit" is a policy
question the guard's owner answers, not a bug by definition.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

from ajantik import legacy

# Turkish case and ASCII-folding rules (data). The case variants use them because a guard that
# changes case with the default rules mishandles the dotted and dotless I.
TR_UPPER = str.maketrans({"i": "İ", "ı": "I"})
TR_LOWER = str.maketrans({"I": "ı", "İ": "i"})
TR_ASCII = str.maketrans("ıİşŞğĞçÇöÖüÜ", "iIsSgGcCoOuU")
HOMOGLYPH = {"a": "а", "e": "е", "o": "о", "c": "с", "p": "р", "x": "х", "y": "у", "A": "А",
             "E": "Е", "O": "О", "C": "С", "P": "Р", "K": "К", "M": "М", "T": "Т", "H": "Н"}

SYNONYMS_EN: dict[str, list[str]] = {
    "submit": ["send", "confirm", "complete", "finalize"],
    "confirm": ["approve", "accept", "agree"],
    "delete": ["remove", "erase", "discard"],
    "pay": ["checkout", "purchase", "buy"],
    "publish": ["post", "share"],
    "transfer": ["move"],
}

# Verbs of Turkish user interfaces (data), for guards that protect Turkish buttons.
SYNONYMS_TR: dict[str, list[str]] = {
    "gönder": ["ilet", "yolla", "tamamla", "kesinleştir", "submit", "send"],
    "onayla": ["onay ver", "kabul et", "confirm", "approve"],
    "sil": ["kaldır", "delete", "remove"],
    "öde": ["ödeme yap", "satın al", "pay", "checkout"],
    "yayınla": ["paylaş", "publish", "post"],
    "aktar": ["transfer et", "taşı", "transfer"],
}

# Named synonym sets a spec can ask for with `synonyms: <name>`. "default" is both languages.
SYNONYM_SETS: dict[str, dict[str, list[str]]] = {
    "default": {**SYNONYMS_EN, **SYNONYMS_TR},
    "en": SYNONYMS_EN,
    "tr": SYNONYMS_TR,
}
DEFAULT_SYNONYMS = SYNONYM_SETS["default"]


@dataclass(frozen=True)
class Mutation:
    name: str
    description: str
    fn: Callable[[str], str | None]  # None: not applicable to this text


def _longest_word_middle(text: str, insert: str) -> str | None:
    words = re.findall(r"\w{3,}", text)
    if not words:
        return None
    w = max(words, key=len)
    i = text.index(w) + len(w) // 2
    return text[:i] + insert + text[i:]


def _fullwidth(text: str) -> str | None:
    out = "".join(chr(ord(c) + 0xFEE0) if "!" <= c <= "~" else c for c in text)
    return out if out != text else None


def _homoglyph(text: str) -> str | None:
    for i, c in enumerate(text):
        if c in HOMOGLYPH:
            return text[:i] + HOMOGLYPH[c] + text[i + 1 :]
    return None


def _changed(fn: Callable[[str], str]) -> Callable[[str], str | None]:
    def wrapped(text: str) -> str | None:
        out = fn(text)
        return out if out != text else None

    return wrapped


INVARIANCE: list[Mutation] = [
    Mutation("decomposed_unicode",
             "Letters in decomposed Unicode (é = e + ´), common in Mac/Word text",
             _changed(lambda t: unicodedata.normalize("NFD", t))),
    Mutation("fullwidth", "Full-width letters (Ｓubmit)", _fullwidth),
    Mutation("zero_width", "Invisible zero-width space in the middle of a word",
             lambda t: _longest_word_middle(t, "\u200b")),
    Mutation("soft_hyphen", "Invisible soft hyphen in the middle of a word",
             lambda t: _longest_word_middle(t, "\u00ad")),
    Mutation("no_break_space", "Spaces replaced by no-break spaces (NBSP)",
             lambda t: t.replace(" ", "\u00a0") if " " in t else None),
    Mutation("upper_case", "Upper case with Turkish rules (i becomes dotted capital I)",
             _changed(lambda t: t.translate(TR_UPPER).upper())),
    Mutation("lower_case", "Lower case with Turkish rules (I becomes dotless i)",
             _changed(lambda t: t.translate(TR_LOWER).lower())),
    Mutation("ascii_letters", "Turkish letters folded to their ASCII base letters",
             _changed(lambda t: t.translate(TR_ASCII))),
    Mutation("lookalike_letter", "A Latin letter replaced by an identical-looking Cyrillic one",
             _homoglyph),
    Mutation("extra_whitespace", "Leading/trailing spaces, double spaces, line break",
             lambda t: "  " + t.replace(" ", "  ") + "\n"),
    Mutation("punctuation", "Wrapped in quotes and followed by a colon", lambda t: f"«{t}»:"),
]

EMBED = Mutation("embedded_text", "Joined with other text (Save | …)", lambda t: f"Save | {t}")

FAMILY_ALIASES: dict[str, str] = legacy.GUARD_FAMILIES  # older lab specs


def synonym_variants(text: str, synonyms: dict[str, list[str]]) -> list[tuple[str, str]]:
    """(synonym, variant) pairs: every word matching a key replaced by each synonym."""
    out = []
    folded = text.translate(TR_LOWER).lower()
    for key, alts in synonyms.items():
        k = key.translate(TR_LOWER).lower()
        idx = folded.find(k)
        if idx < 0:
            continue
        for alt in alts:
            out.append((alt, text[:idx] + alt + text[idx + len(k) :]))
    return out


CODE = re.compile(r"^(?P<pre>[A-Za-z]*)(?P<num>\d+)(?P<suf>[A-Za-z]*)$")


def near_misses(code: str) -> list[tuple[str, str]]:
    """Codes that look alike but mean something else: (why, variant)."""
    m = CODE.match(code.strip())
    if not m:
        return []
    pre, num, suf = m["pre"], m["num"], m["suf"]
    out: list[tuple[str, str]] = []
    if suf:
        out.append(("suffix removed", pre + num))
        out.append(("suffix letter changed",
                    pre + num + ("D" if suf[0].upper() != "D" else "F") + suf[1:]))
        flipped = suf.swapcase()
        if flipped != suf:
            out.append(("suffix case changed", pre + num + flipped))
    else:
        out.append(("suffix added", pre + num + "i"))
    last = str((int(num[-1]) + 1) % 10)
    out.append(("last digit changed", pre + num[:-1] + last + suf))
    return [(why, v) for why, v in out if v != code]
