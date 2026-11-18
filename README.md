# Whisper for Hinglish Speech Recognition

On-device speech-to-text for people who are hard of hearing, for **English, Hindi, and mixed Hindi-English (Hinglish)** speech.

This repository contains a **zero-shot Whisper baseline** and a **fine-tuned Whisper small model** for Hinglish.

**Fine-tuning is done, and the fine-tuned model performs better than zero-shot Whisper:**
- Hinglish WER: **365.03% → 83.62%** (4.4 times fewer errors)
- Hallucinations: **86 → 7**
- English WER stays stable: **3.26% → 3.61%**

---

## Why this project

- Many people in India speak **Hinglish**: they switch between Hindi and English in the same sentence.
- Most speech recognition models are trained for one language at a time. They make many errors at the language-switch points.
- People who are hard of hearing need live captions that work **on the phone, without the internet**, for in-person conversations and video calls.

The project goal: a small, quantized Whisper model that writes Hinglish captions in real time on a mid-range phone.

**Script rule:** Hindi words in **Devanagari**, English words in **Latin** script. Example: `background टैब पर click करें`.

---

## What the notebook does

The notebook `whisper_baseline_openslr104.ipynb` runs in Google Colab on one T4 GPU. It has two parts.

### Part A: Zero-shot baseline (sections 1–12)

1. Downloads the OpenSLR 104 Hindi-English **test** set (443 MB).
2. Cuts the long recordings into utterances with the Kaldi `segments` file.
3. Runs Whisper tiny, base, and small **without fine-tuning**, with forced Hindi (`hi`) and with automatic language detection (`auto`).
4. Normalizes the references and the outputs with the same rules.
5. Calculates WER, CER, error types, script mix, and possible hallucinations.
6. Shows the worst examples.

### Part B: Fine-tuning and evaluation (cells F1–F10)

1. Streams 50 hours of Hinglish speech from Hugging Face and saves it to the Colab disk.
2. Removes every training clip whose normalized transcript matches an OpenSLR 104 test transcript.
3. Keeps validation speakers separate from training speakers.
4. Adds English speech (10% of the clips) so that the model does not forget English.
5. Fine-tunes Whisper small and keeps the checkpoint with the lowest validation WER.
6. Compares the zero-shot and fine-tuned models on the same test clips, in Hinglish and in English.

---

## Data

| Dataset | Use | Content | License |
|---|---|---|---|
| [OpenSLR 104](https://www.openslr.org/104/) (MUCS 2021, Hindi-English) | **Test only** | Code-switched speech from IIT Bombay Spoken Tutorials. Test set: 3,136 utterances, 5.18 h | CC BY-SA 4.0 |
| [agarwalayushi/hinglish](https://huggingface.co/datasets/agarwalayushi/hinglish) | Train + validation | Code-switched Spoken Tutorial speech, about 2,264 h (estimate). We use 50 h. | CC-BY-4.0 (stated on the dataset page) |
| [fixie-ai/librispeech_asr](https://huggingface.co/datasets/fixie-ai/librispeech_asr) (LibriSpeech) | English train mix + English check | Read English audiobooks | CC BY 4.0 |

### Data used in this run

| Split | Clips | Hours | Notes |
|---|---|---|---|
| Train (Hinglish) | 21,698 | 50.00 | From agarwalayushi/hinglish |
| Train (English) | 2,169 | ~7.6 | LibriSpeech `train.100` |
| **Train total** | **23,867** | **57.60** | |
| Validation | 300 | — | 5 held-out speakers |
| Test (Hinglish) | 200 | 0.31 | Fixed random subset (seed 0) of the OpenSLR 104 test set |
| Test (English) | 200 | — | LibriSpeech `test` (clean) |

**Clips skipped while streaming the Hinglish training data:**

| Reason | Clips |
|---|---|
| Duration outside 0.5–30 s | 1,065 |
| Transcript longer than 440 tokens | 237 |
| Transcript matches an OpenSLR 104 test transcript | 85 |
| Validation speaker, validation already full | 109 |
| Empty transcript | 7 |

### Overlap control

Both the training data and the test set come from the IIT Bombay Spoken Tutorial project. To decrease leakage:
- The notebook removes training clips whose **normalized transcript** matches any of the 2,877 unique OpenSLR 104 test transcripts (85 clips removed).
- It cannot check for the **same speakers** in training and test, because the two datasets use different speaker IDs. See [Limitations](#limitations).

---

## Method

### Models
- Zero-shot: `openai/whisper-tiny` (39M), `openai/whisper-base` (74M), `openai/whisper-small` (244M), multilingual checkpoints.
- Fine-tuned: `openai/whisper-small`.
- Library: Hugging Face `transformers` (not the `openai-whisper` package).

### Decoding
- Greedy (beam size 1), temperature 0, maximum 400 new tokens.
- Clips are 0.5–30 s, because Whisper reads a 30-second window.

### Text normalization
The same rules are applied to the references and to the outputs:
1. Unicode NFC.
2. Remove tags such as `<noise>`, `[music]`, `{laugh}`.
3. Devanagari digits → ASCII digits.
4. Lowercase (Latin only).
5. Remove punctuation, symbols, and format characters (for example, ZWJ). Devanagari vowel signs and virama are kept.
6. Collapse whitespace.

Numbers are **not** converted between digits and words. English words written in Devanagari are **not** converted to Latin.

### Metrics
| Metric | Meaning |
|---|---|
| WER / CER | Word / character error rate (same definition as `jiwer`). Can be above 100% when the output has many extra words. |
| Sub / Del / Ins | Substitutions, deletions, insertions, as a percentage of reference words |
| Latin % / Devanagari % / Other script % | Script of the output words. "Other" is a wrong script, for example Urdu. |
| Possible hallucinations | Outputs with more than 2 × the reference words + 5 |
| RTF | Processing time ÷ audio duration, **on the Colab T4 GPU** (not on a phone) |

### Fine-tuning (full fine-tuning, all parameters)
| Setting | Value |
|---|---|
| Objective | Cross-entropy on the next token (teacher forcing) |
| Optimizer | AdamW (Trainer default) |
| Learning rate | 1e-5, 300 warmup steps, then linear decrease |
| Steps / batch | 3,000 steps, batch size 16 (about 2 epochs) |
| Precision | fp16 mixed precision |
| Memory | Gradient checkpointing |
| Augmentation | SpecAugment, time-mask probability 0.05 |
| Language tokens | `<\|hi\|>` for Hinglish clips, `<\|en\|>` for English clips |
| Model selection | Lowest validation WER (evaluated every 500 steps) |

---

## Results

All Hinglish results use the same **200-utterance subset** (0.31 h) of the OpenSLR 104 test set. Lower is better.

### Zero-shot baseline (OpenSLR 104 test, 200 utterances)

Reference: **22.8%** of reference words are in Latin script.

| Model | Lang | WER | CER | Sub | Del | Ins | Latin % | Other script % | Halluc. | RTF (T4) |
|---|---|---|---|---|---|---|---|---|---|---|
| whisper-tiny | hi | 239.71 | 199.13 | 65.21 | 26.43 | 148.07 | 50.19 | 0.06 | 26 | 0.03 |
| whisper-tiny | auto | 234.79 | 196.72 | 65.43 | 26.25 | 143.12 | 51.62 | 0.27 | 26 | 0.03 |
| whisper-base | hi | 246.35 | 233.01 | 64.45 | 26.43 | 155.47 | 57.33 | 14.84 | 32 | 0.04 |
| whisper-base | auto | 224.44 | 202.11 | 63.79 | 27.18 | 133.47 | 54.96 | 16.80 | 29 | 0.04 |
| whisper-small | hi | 365.03 | 300.14 | 70.87 | 10.00 | 284.15 | 0.14 | 0.00 | 86 | 0.08 |
| whisper-small | auto | 343.07 | 282.11 | 70.39 | 10.71 | 261.97 | 0.76 | 0.68 | 79 | 0.08 |

### Zero-shot vs fine-tuned (Whisper small, forced Hindi, greedy)

| Model | WER | CER | Sub | Del | Ins | Latin % | Other script % | Halluc. |
|---|---|---|---|---|---|---|---|---|
| whisper-small (zero-shot) | 365.03 | 300.14 | 70.87 | 10.00 | 284.15 | 0.14 | 0.00 | 86 |
| **whisper-small-hinglish-ft** | **83.62** | **73.00** | 40.19 | 10.09 | 33.33 | 5.42 | 0.00 | **7** |

Validation (held-out speakers) before training: WER 253.60, CER 196.75.
Validation WER at each evaluation step is in `log_history.json`.

### English check (LibriSpeech test-clean, 200 clips)

| Model | WER | CER |
|---|---|---|
| whisper-small (zero-shot) | 3.26 | 1.11 |
| whisper-small-hinglish-ft | 3.61 | 1.15 |

---

## Observations

1. **Zero-shot Whisper fails on Hinglish.** All models have WER above 200%. Most errors are **insertions**: the models often go into repetition loops (for example, the same phrase many times for a 5-second clip).
2. **Script errors.** Zero-shot Whisper small writes almost all English words in Devanagari (Latin 0.14% vs 22.8% in the references). Whisper base sometimes writes in a third script (Other script ≈ 15–17%).
3. **Fine-tuning removes most hallucinations.** Possible hallucinations drop from 86 to 7, and insertions from 284% to 33%. This is most of the WER improvement.
4. **Fine-tuning gives a large improvement.** WER decreases 4.4 times (365.03% → 83.62%) and CER 4.1 times (300.14% → 73.00%).
5. **English stays stable.** English WER increases by only 0.35 points (3.26 → 3.61) after fine-tuning, so the English mix works.
6. **The fine-tuned model still writes few Latin words** (5.42% vs 22.8% in the references). Script mismatch is probably a large part of the remaining error. This needs to be confirmed.

---

## Limitations

- **Small test subset.** The results use 200 utterances (0.31 h), not the full test set (3,132 utterances, 5.14 h). Small differences are not reliable.
- **Same source for training and test.** Both come from IIT Bombay Spoken Tutorials. Transcript matches are removed, but the same speakers can be in both. The OpenSLR 104 result is an **in-domain** result.
- **Small validation set.** 300 clips from only 5 speakers.
- **Script variation in the references.** OpenSLR 104 sometimes writes English words in Devanagari (for example, `वीडियो`, `सेटिंग्स`) and sometimes in Latin. The normalization does not join the two scripts, so a correct word in the other script counts as an error.
- **Possible label errors.** Other users report misaligned segments in the MUCS test set. The worst examples need to be checked by listening.
- **No number normalization.** "4" and "four" count as different words.
- **Long clips removed.** Clips longer than 30 s are not used (4 test clips, 1,065 training clips).
- **Detected-language column.** In `transformers` 5.22.1, the notebook shows `unknown` for most outputs. The language-token parsing does not work with this version.
- **Not yet tested:** conversational speech (HiACC), noisy speech, video-call audio, quantized models, phone speed.

---

## How to run

1. Open `whisper_baseline_openslr104.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Select **Runtime → Change runtime type → T4 GPU**.
3. **Part A:** run sections 1–12. Set `MAX_UTTS = 200` for a quick run, or `0` for the full test set.
4. **Part B:** run cells F1–F10.
   - First do a short test: `TRAIN_HOURS = 1`, `MAX_STEPS = 50`, `EVAL_STEPS = SAVE_STEPS = 25`, `VAL_UTTS = 50`.
   - With `USE_DRIVE = True`, checkpoints go to Google Drive (about 7–8 GB free space is necessary). After a disconnect, run sections 1–9 and F1–F9 again. Training continues from the last checkpoint.
5. No package installation is necessary. Colab already has all the libraries.

### Environment used for the results

| Item | Version |
|---|---|
| GPU | Tesla T4 |
| Python | 3.13.16 |
| PyTorch | 2.14.1+cu130 |
| Transformers | 5.22.1 |

---

## Outputs

| Path | Content |
|---|---|
| `results/baseline_openslr104/summary.csv`, `summary.md` | Zero-shot results table |
| `results/baseline_openslr104/<model>__lang-<x>__beam-<n>.jsonl` | Every reference and output, original and normalized, with utterance WER |
| `results/baseline_openslr104/run_info.json` | Settings, versions, GPU, normalization rules |
| `<RUN_DIR>/final/` | Fine-tuned model and processor |
| `<RUN_DIR>/log_history.json` | Training loss and validation metrics |
| `<RUN_DIR>/comparison_openslr104.csv` | Zero-shot vs fine-tuned (Hinglish) |
| `<RUN_DIR>/comparison_english.csv` | Zero-shot vs fine-tuned (English) |

`RUN_DIR` is `/content/drive/MyDrive/whisper_hinglish/whisper-small` when `USE_DRIVE = True`.

---

## Planned work

- [ ] Evaluate on the full OpenSLR 104 test set.
- [ ] Find the cause of the remaining error (script, labels, or training length).
- [ ] Evaluate on HiACC (conversational Hinglish, adults and children).
- [ ] Quantize to INT8 and INT4, and measure the accuracy loss.
- [ ] Run on a mid-range Android phone: real-time factor, latency, memory.
- [ ] Real-time captions: voice activity detection and streaming chunks.
- [ ] Audio capture for in-person speech and video calls.
- [ ] Accessible interface and testing with hard-of-hearing users.

---

## Licenses and acknowledgements

- **Whisper** models by OpenAI (MIT License).
- **OpenSLR 104 / MUCS 2021** Hindi-English data (CC BY-SA 4.0), from the IIT Bombay Spoken Tutorial project. Used for testing only.
- **agarwalayushi/hinglish** (CC-BY-4.0, as stated on the dataset page).
- **LibriSpeech** (CC BY 4.0), via `fixie-ai/librispeech_asr`.
- **Hugging Face** `transformers` and `datasets`.

The code license for this repository: _add your license here (for example, MIT)._

Audio data is not included in this repository. The notebook downloads it from the original sources.
