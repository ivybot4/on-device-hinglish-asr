# On-Device AI for Real-Time Audio to Code-Switched Text

# Objective

Fine-tune and quantize a cache-aware streaming Conformer that runs on an Android phone and converts live speech into mixed-script Hinglish text (Hindi in Devanagari, English in Latin script) with less than 1 second of latency.

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

# Methodology

## Data Cleaning and Analysis

[openssl-analysis.ipynb](notebooks/analysis/openssl-analysis.ipynb) prepares the OpenSLR 104 training data:

1. Joins the Kaldi files (`segments`, `text`, `utt2spk`, `wav.scp`) into one utterance table and checks IDs and audio format (16 kHz, mono, 16-bit).
2. Splits train/validation by recording (~5% of hours), so no recording is in both.
3. Removes test leaks: recordings shared with the test set and identical transcripts with 3+ words.
4. Drops clips shorter than 0.5 s or longer than 20 s.
5. Finds English words written in Devanagari (e.g. फ़ाइल → file) for conversion to Latin script.
