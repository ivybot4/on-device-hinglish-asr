import csv
import re
import unicodedata

# List position = value. English 21-99 are two words, so only single words are mapped.
_EN = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_EN_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
_HI = """शून्य एक दो तीन चार पांच छह सात आठ नौ दस ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह उन्नीस
बीस इक्कीस बाईस तेईस चौबीस पच्चीस छब्बीस सत्ताईस अट्ठाईस उनतीस तीस इकतीस बत्तीस तैंतीस चौंतीस पैंतीस छत्तीस सैंतीस अड़तीस उनतालीस
चालीस इकतालीस बयालीस तैंतालीस चवालीस पैंतालीस छियालीस सैंतालीस अड़तालीस उनचास पचास इक्यावन बावन तिरेपन चौवन पचपन छप्पन सत्तावन अट्ठावन उनसठ
साठ इकसठ बासठ तिरसठ चौंसठ पैंसठ छियासठ सड़सठ अड़सठ उनहत्तर सत्तर इकहत्तर बहत्तर तिहत्तर चौहत्तर पचहत्तर छिहत्तर सतहत्तर अठहत्तर उनासी
अस्सी इक्यासी बयासी तिरासी चौरासी पचासी छियासी सत्तासी अट्ठासी नवासी नब्बे इक्यानवे बानवे तिरानवे चौरानवे पचानवे छियानवे सत्तानवे अट्ठानवे निन्यानवे सौ""".split()
# Spelling variants seen in the transcripts.
_HI_VARIANTS = {"पाँच": "5", "छः": "6", "छे": "6", "पन्द्रह": "15", "अट्ठारह": "18"}

_NUMBERS = (
    {w: str(i) for i, w in enumerate(_EN)}
    | {w: str(20 + 10 * i) for i, w in enumerate(_EN_TENS)}
    | {"hundred": "100"}
    | {w: str(i) for i, w in enumerate(_HI)}
    | _HI_VARIANTS
)
# NFC to match normalize() output: it decomposes ड़ (a composition exclusion).
_NUMBERS = {unicodedata.normalize("NFC", w): d for w, d in _NUMBERS.items()}
_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


class Normalizer:
    def __init__(self, loanwords_path, transliterations_path=None):
        with open(loanwords_path, encoding="utf-8") as f:
            self.loanwords = {unicodedata.normalize("NFC", r["devanagari"]): r["english"] for r in csv.DictReader(f)}
        # English word -> accepted Devanagari forms, for T-WER. Without the file, T-WER equals WER.
        self.transliterations = {}
        if transliterations_path:
            with open(transliterations_path, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    forms = self.transliterations.setdefault(self.for_scoring(r["english"]), set())
                    forms.add(self.for_scoring(r["devanagari"]))

    def clean(self, text):
        t = unicodedata.normalize("NFC", text)
        t = re.sub("[​‌‍﻿]", "", t)
        t = t.lower()
        t = re.sub(r"[^ऀ-ॣ॰-ॿa-z0-9०-९ ]", " ", t)
        return " ".join(t.split())

    def normalize(self, text):
        # Must stay identical to the code that made the training data.
        return " ".join(self.loanwords.get(w, w) for w in self.clean(text).split())

    def for_scoring(self, text):
        # No loanword mapping, so a Devanagari loanword is an error in WER (T-WER allows it).
        # Models write numbers as digits or words; the transcripts only use words.
        t = self.clean(text).translate(_DEV_DIGITS)
        return " ".join(_NUMBERS.get(w, w) for w in t.split())
