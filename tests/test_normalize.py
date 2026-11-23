import pytest

from hinglish_asr.utils import Normalizer


@pytest.fixture
def norm(tmp_path):
    path = tmp_path / "loanwords.csv"
    path.write_text("devanagari,english\nक्लिक,click\nथ्री,three\n", encoding="utf-8")
    return Normalizer(path)


def test_normalize(norm):
    assert norm.normalize("अब  हम Terminal में क्लिक करेंगे।") == "अब हम terminal में click करेंगे"


@pytest.mark.parametrize("words, digit", [
    (["three", "तीन", "3", "३", "थ्री"], "3"),
    (["zero", "शून्य", "0"], "0"),
    (["fifty", "पचास", "50"], "50"),
    (["पाँच", "पांच", "5"], "5"),
    (["अड़तीस", "38"], "38"),        # ड़ is decomposed by NFC
    (["उनहत्तर", "69"], "69"),       # spot-checks that list positions match values
    (["निन्यानवे", "99"], "99"),
    (["hundred", "सौ", "100"], "100"),
])
def test_for_scoring_numbers(norm, words, digit):
    assert {norm.for_scoring(w) for w in words} == {digit}
