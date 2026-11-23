import pytest

from hinglish_asr.metrics import score, t_wer, wer
from hinglish_asr.utils import Normalizer


@pytest.fixture
def norm(tmp_path):
    loanwords = tmp_path / "loanwords.csv"
    loanwords.write_text("devanagari,english\nक्लिक,click\n", encoding="utf-8")
    forms = tmp_path / "transliterations.csv"
    forms.write_text("english,devanagari\ncustom,कस्टम\nclick,क्लिक\ncell,सेल\n", encoding="utf-8")
    return Normalizer(loanwords, forms)


def test_wer_identical():
    refs = ["अब हम terminal में click करेंगे", "तीन"]
    assert wer(refs, refs) == 0.0


def test_t_wer_ignores_loanword_script():
    refs, hyps = ["click करें"], ["क्लिक करें"]
    assert wer(refs, hyps) > 0
    assert t_wer(refs, hyps, {"click": {"क्लिक"}}) == 0.0


def test_devanagari_form(norm):
    # 1 error in 4 words for WER, none for T-WER.
    result = score(["custom background बना सकते"], ["कस्टम background बना सकते"], norm)
    assert result == {"wer": 0.25, "t_wer": 0.0}


def test_misspelled_form(norm):
    assert score(["custom बना"], ["कसटम बना"], norm) == {"wer": 0.5, "t_wer": 0.5}


def test_hindi_word(norm):
    assert score(["आप बना सकते हैं"], ["आप बना सकती हैं"], norm) == {"wer": 0.25, "t_wer": 0.25}


def test_refs_against_refs(norm):
    refs = ["तीन बार click करें", "आप custom backgrounds बना सकते हैं"]
    assert score(refs, refs, norm) == {"wer": 0.0, "t_wer": 0.0}


def test_t_wer_not_above_wer(norm):
    # सेल is both a reference word and a form of cell, so it must not be changed.
    refs = ["सेल में cell चुनें", "custom बना", "click करें"]
    hyps = ["सेल में सेल चुनें", "कस्टम कस्टम", "बटन करें"]
    result = score(refs, hyps, norm)
    assert result["t_wer"] <= result["wer"]
    assert result["t_wer"] < result["wer"]


def test_score(norm):
    result = score(["तीन बार Click करें।"], ["3 बार क्लिक करें"], norm)
    assert result == {"wer": 0.25, "t_wer": 0.0}
