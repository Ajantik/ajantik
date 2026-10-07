import shutil
from pathlib import Path

import pytest

from ajantik.guards.mutations import INVARIANCE, near_misses, synonym_variants
from ajantik.guards.run import report, run_spec, visible

K = Path(__file__).parent.parent / "examples" / "kapilar"
needs_examples = pytest.mark.skipif(not K.exists(),
                                    reason="guard examples kept out of the public release")

GUARD_PY = """
import unicodedata

def blocked(text):
    t = unicodedata.normalize("NFKC", text).casefold()
    return "submit" in "".join(c for c in t if unicodedata.category(c) != "Cf")
"""


def test_mutations_change_text_but_not_meaning():
    by = {m.name: m for m in INVARIANCE}
    assert by["decomposed_unicode"].fn("Café") == "Cafe\u0301"
    assert "\u200b" in by["zero_width"].fn("Confirm")
    assert by["no_break_space"].fn("Pay now") == "Pay\u00a0now"
    assert by["decomposed_unicode"].fn("Save") is None  # nothing to decompose: not applicable


def test_near_misses_cover_suffixes():
    variants = {v for _, v in near_misses("H360F")}
    assert {"H360", "H360D", "H360f", "H361F"} <= variants
    assert ("suffix added", "H350i") in near_misses("H350")


def test_synonyms_replace_the_matching_word():
    pairs = synonym_variants("Submit the form", {"submit": ["send"]})
    assert pairs == [("send", "send the form")]


def test_visible_marks_invisible_characters():
    assert visible("a\u200bb\u00a0c") == "a⟨U+200B⟩b⟨U+00A0⟩c"


def test_spec_with_english_keys_runs(tmp_path):
    (tmp_path / "guard.py").write_text(GUARD_PY, encoding="utf-8")
    spec = tmp_path / "spec.yaml"
    spec.write_text(
        "guard: py:guard.py#blocked\n"
        "kind: decision\n"
        "seeds:\n"
        "  - {text: 'Submit the form', expected: true}\n"
        "  - {text: 'Save draft', expected: false}\n"
        "synonyms: en\n"
        "exclude: [ascii_letters]\n"
        "family_expected: {lookalike_letter: true}\n",
        encoding="utf-8",
    )
    cases, s = run_spec(spec)
    families = {c.family for c in cases}
    assert {"seed", "zero_width", "fullwidth", "synonym"} <= families
    assert "ascii_letters" not in families
    assert all(c.passed for c in cases if c.family in ("seed", "zero_width", "fullwidth"))
    text = report(spec, s, cases)
    assert "# Guard fuzzing report" in text and "Synonyms" in text


def test_unknown_kind_is_refused(tmp_path):
    (tmp_path / "guard.py").write_text(GUARD_PY, encoding="utf-8")
    spec = tmp_path / "spec.yaml"
    spec.write_text("guard: py:guard.py#blocked\nkind: ranking\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown guard kind"):
        run_spec(spec)


def _failures(spec: str) -> dict[str, int]:
    cases, _ = run_spec(K / spec)
    out: dict[str, int] = {}
    for c in cases:
        if c.family != "synonym" and not c.passed:
            out[c.family] = out.get(c.family, 0) + 1
    return out


@needs_examples
def test_naive_code_comparator_accepts_different_hazard_codes():
    fails = _failures("tehlike_kodu.yaml")
    assert fails["near_code"] >= 8  # H360F == H360D etc.
    assert _failures("tehlike_kodu_saglam.yaml") == {}


@needs_examples
def test_naive_character_gate_misses_decomposed_unicode():
    assert _failures("karakter_kapisi.yaml").get("decomposed_unicode") == 2
    assert _failures("karakter_kapisi_saglam.yaml") == {}


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@needs_examples
def test_js_send_lock_is_bypassed_by_invisible_characters():
    fails = _failures("gonder_kilidi.yaml")
    assert fails.get("zero_width") and fails.get("decomposed_unicode")
    assert _failures("gonder_kilidi_saglam.yaml") == {}
