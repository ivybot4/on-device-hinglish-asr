# On-Phone Accessibility: Distress Sound ALerts and Code-Switched Bilingual Captions

# Objectives

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

[openssl-analysis.ipynb](notebooks/analysis/openssl-analysis.ipynb) prepares the OpenSLR 104 training data:

1. Joins the Kaldi files (`segments`, `text`, `utt2spk`, `wav.scp`) into one utterance table and checks IDs and audio format (16 kHz, mono, 16-bit).
2. Splits train/validation by recording (~5% of hours), so no recording is in both.
3. Removes test leaks: recordings shared with the test set and identical transcripts with 3+ words.
4. Drops clips shorter than 0.5 s or longer than 20 s.
5. Finds English words written in Devanagari (e.g. फ़ाइल → file) for conversion to Latin script.
6. Normalizes the transcript text

# Future Scope

1. Deploy the model on laptop for live Zoom or Google Meet calls.
2. When Andriod permits, deploy the model for live calls on Zoom, Meet or WhatsApp video calling on phone.