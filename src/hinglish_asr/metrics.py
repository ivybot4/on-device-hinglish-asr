import jiwer


def wer(refs, hyps):
    # Corpus-level: total errors over total reference words.
    return jiwer.wer(refs, hyps)


def t_wer(refs, hyps, loanwords):
    # A loanword counts as correct in either script (क्लिक == click).
    def to_english(t):
        return " ".join(loanwords.get(w, w) for w in t.split())
    return wer([to_english(r) for r in refs], [to_english(h) for h in hyps])


def score(refs, hyps, normalizer):
    refs = [normalizer.for_scoring(t) for t in refs]
    hyps = [normalizer.for_scoring(t) for t in hyps]
    return {"wer": wer(refs, hyps), "t_wer": t_wer(refs, hyps, normalizer.loanwords)}
