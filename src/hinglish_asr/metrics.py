import jiwer


def wer(refs, hyps):
    # Corpus-level: total errors over total reference words.
    return jiwer.wer(refs, hyps)


def t_wer(refs, hyps, transliterations):
    # MUCS 2021: an English reference word is correct in Latin script or in an
    # accepted Devanagari form (क्लिक == click). Only words in that reference are
    # mapped, and never a word the reference itself has, so T-WER <= WER.
    def to_ref(ref, hyp):
        ref_words = set(ref.split())
        forms = {f: e for e in ref_words for f in transliterations.get(e, ())}
        return " ".join(w if w in ref_words else forms.get(w, w) for w in hyp.split())
    return wer(refs, [to_ref(r, h) for r, h in zip(refs, hyps)])


def score(refs, hyps, normalizer):
    refs = [normalizer.for_scoring(t) for t in refs]
    hyps = [normalizer.for_scoring(t) for t in hyps]
    return {"wer": wer(refs, hyps), "t_wer": t_wer(refs, hyps, normalizer.transliterations)}
