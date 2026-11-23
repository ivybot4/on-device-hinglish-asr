from hinglish_asr.metrics import score, t_wer, wer
from hinglish_asr.utils import Normalizer


def test_wer_identical():
    refs = ["अब हम terminal में click करेंगे", "तीन"]
    assert wer(refs, refs) == 0.0


def test_t_wer_ignores_loanword_script():
    refs, hyps = ["click करें"], ["क्लिक करें"]
    assert wer(refs, hyps) > 0
    assert t_wer(refs, hyps, {"क्लिक": "click"}) == 0.0


def test_score(tmp_path):
    path = tmp_path / "loanwords.csv"
    path.write_text("devanagari,english\nक्लिक,click\n", encoding="utf-8")
    result = score(["तीन बार Click करें।"], ["3 बार क्लिक करें"], Normalizer(path))
    assert result == {"wer": 0.0, "t_wer": 0.0}
