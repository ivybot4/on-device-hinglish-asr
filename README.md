# On-Phone Accessibility: Distress Sound Alerts and Code-Switched Bilingual Captions

# Problem Statement

1. During in-person conversations display code-switched text on phone with real-time streaming for bilingual accessibility. 
2. An alert notification on phone about specific sounds like a glass breaking, a loud thud, or someone calling out your name or an alias you enter.


# Datasets

## Training datasets

| Dataset    | Language | Link                         |
|------------|----------|------------------------------|
| OpenSLR    | Hinglish | https://www.openslr.org/104/ |

## Test datasets

Main test sets are 2 standard benchmarking sets for Hinglish. Hindi, English and English with an Indian setting are more for model performance comparison and signals to reduce/increase the data.

| Dataset | Language | Link |
|---|---|---|
| OpenSLR | Hinglish | https://www.openslr.org/104/ |
| HiACC | Hinglish | https://zenodo.org/records/15551669 |
| Lahaja | Hindi | https://huggingface.co/datasets/ai4bharat/Lahaja |
| Svarah | Indian English | https://huggingface.co/datasets/ai4bharat/Svarah |
| LibriSpeech test-clean | English | https://huggingface.co/datasets/fixie-ai/librispeech_asr |

# Architecture

| # | Model | Role | Method |
|---|---|---|---|
| 1 | Whisper small | Baseline | Zero-shot |
| 2 | Whisper small (CarelessWhisper) | Phone candidate (experimental) | Fine-tune on Hinglish to make Whisper causal, so it can stream |
| 3 | Qwen3-ASR-0.6B | Baseline | Zero-shot |
| 4 | Srota (Qwen3-ASR-0.6B, Hinglish fine-tune) | Reference | Zero-shot scored with the project normalizer |
| 5 | Nemotron 3.5 ASR Streaming 0.6B | Baseline | Zero-shot |
| 6 | Nemotron 3.5 ASR Streaming 0.6B | Phone candidate (main) | Full fine-tune on Hinglish |
| 7 | Whisper Hinglish large-v3 (Trelis) | Distillation teacher | Zero-shot; labels untranscribed audio for distillation |
| 8 | Streaming Zipformer (25–65M) | Phone candidate (small) | Distillation from the teacher (model 7) |

# Methodology

## Data Cleaning and Analysis

[openslr-analysis.ipynb](notebooks/analysis/openslr-analysis.ipynb) prepares the OpenSLR 104 training data:

1. Joins the Kaldi files (`segments`, `text`, `utt2spk`, `wav.scp`) into one utterance table and checks IDs and audio format (16 kHz, mono, 16-bit).
2. Splits train/validation by recording (~5% of hours), so no recording is in both.
3. Removes test leaks: recordings shared with the test set and identical transcripts with 3+ words.
4. Drops clips shorter than 0.5 s or longer than 20 s.
5. Finds English words written in Devanagari (e.g. फ़ाइल → file) for conversion to Latin script.
6. Normalizes the transcript text

## Normalization and Scoring

Every model is scored with the same two modules:

- [utils.py](src/hinglish_asr/utils.py): `Normalizer.normalize` applies the training-data cleaning (NFC, lowercase, punctuation removed, Devanagari loanwords in Latin script). `for_scoring` also maps number words 0–100 (English and Hindi) and Devanagari digits to ASCII digits, so "three", "तीन" and "3" match.
- [metrics.py](src/hinglish_asr/metrics.py): corpus-level WER (jiwer), and T-WER, which counts a loanword as correct in either script (क्लिक = click). `score` returns both on normalized text.

### Metrics

#### 1. WER (Word Error Rate)
Minimum number of word changes that make the reference and output equal divided by the number of words in ref

**WER = (S + D + I) / N**

- **S (substitutions):** a wrong word in place of the correct word
- **D (deletions):** a reference word that is missing in the output
- **I (insertions):** an extra word in the output
- **N:** the number of words in the reference

#### 2. T-WER
Transliterated WER has the same formula as WER.

**T-WER = (S + D + I) / N**

The change is that an English word in the reference is correct if the output has it in Latin script or in a correct Devanagari form from the mapping file.

Basically, WER is exact, T-WER is exact but okay without the proper script.


# Results

## Zero-shot

1. Each model transcribes the full OpenSLR 104 test set (2,942 clips, 4.76 h) with no training, one notebook per model in [notebooks/zero-shot/](notebooks/zero-shot/), on a Colab L4 GPU.
2. Outputs are scored with the normalizer and metrics above.

| Model | WER | Real-time factor |
|---|---|---|
| Whisper Hinglish large-v3 (Trelis) | 0.295 | 0.121 |
| Srota* | 0.306 | 0.053 |
| Qwen3-ASR-0.6B | 0.436 | 0.050 |
| Nemotron 3.5 ASR Streaming 0.6B, 1120 ms latency | 0.455 | 0.009 |
| Nemotron 3.5 ASR Streaming 0.6B, 560 ms latency | 0.458 | 0.009 |
| Whisper small | 0.915 | 0.234 |

Real-time factor is processing time divided by audio length on the L4, so lower is faster.

- T-WER equals WER for every model, because normalization already puts Devanagari loanwords in Latin script before scoring.
- Trelis has the lowest WER, which supports using it as the distillation teacher.
- Nemotron is the fastest by about 5x, and halving its latency to 560 ms barely changes WER. It often writes English words in Devanagari (बैकग्राउंड्स for backgrounds), which fine-tuning should fix.
- Whisper small writes English words in Devanagari and loops on about 6% of clips.
- The 3 test clips longer than 30 s have transcripts that do not match their audio, so every model loses some words there.


# Future Scope

1. Deploy the model on laptop for live Zoom or Google Meet calls.
2. When Android permits, deploy the model for live calls on Zoom, Meet or WhatsApp video calling on phone.