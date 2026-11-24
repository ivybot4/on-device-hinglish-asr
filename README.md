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

# Models

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
4. Drops train and validation clips shorter than 0.5 s or longer than 20 s. The test set keeps all clips.
5. Finds English words written in Devanagari (e.g. फ़ाइल → file) for conversion to Latin script.
6. Normalizes the transcript text, and drops utterances with digits (11%) or empty text.

Final sizes: train 41,531 clips (67.5 h), validation 2,058 clips (3.4 h), test 2,942 clips (4.76 h).

[validation-leak-check.ipynb](notebooks/analysis/validation-leak-check.ipynb) checks validation against train with the same rule: 9.0% of validation clips (7.3% of words) have a transcript that is also in train.

## Normalization and Scoring

Every model is scored with the same two modules:

- [utils.py](src/hinglish_asr/utils.py): `Normalizer.normalize` applies the training-data cleaning (NFC, lowercase, punctuation removed, Devanagari loanwords in Latin script). `for_scoring` uses the same cleaning without the loanword mapping, and maps number words 0–100 (English and Hindi) and Devanagari digits to ASCII digits, so "three", "तीन" and "3" match.
- [metrics.py](src/hinglish_asr/metrics.py): corpus-level WER (jiwer), and T-WER, which counts an English word as correct in either script (क्लिक = click) using `data/lists/transliterations.csv`. `score` returns both on normalized text.

### Scoring Fix

The first scoring (v1) ran the loanword mapping on the model output before WER, so कस्टम already matched "custom" and WER always equalled T-WER. The fix (v2):

1. WER is strict: `for_scoring` no longer maps loanwords, so an English word in Devanagari is an error.
2. T-WER uses a new mapping, [transliterations.csv](data/lists/transliterations.csv): every English word in the test references with one or more Devanagari forms, plus all of `loanwords.csv`. Forms that are also normal Hindi words (पास, मेल) are left out. It is built from a review file sorted by word frequency, for a hand check.
3. A Devanagari output word counts as an English word only if that English word is in the same reference, so T-WER is never above WER.
4. Tests cover custom/कस्टम (an error in WER only), a misspelled form, a Hindi word, references against themselves, and T-WER ≤ WER.

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

## Fine-Tuning

[nemotron-3.5-asr-streaming-0.6b.ipynb](notebooks/fine-tune/nemotron-3.5-asr-streaming-0.6b.ipynb) fine-tunes all 638M weights of Nemotron 3.5 ASR Streaming 0.6B on the OpenSLR 104 training set (41,531 clips, 67.5 h) in NeMo, following NVIDIA's fine-tuning recipe:

1. Every clip gets the `hi-IN` language prompt.
2. The model's own config is kept: features, SpecAugment, and streaming attention that trains on 4 latencies (80, 320, 560 and 1,120 ms) picked at random. Only the data, optimizer and trainer settings change.
3. AdamW with Noam annealing (peak learning rate 1e-4 after 100 warmup steps), bf16, and batches of 100 s of audio grouped by clip length.
4. Validation WER and T-WER at 560 ms every 1,000 steps. The checkpoint with the lowest WER is kept, and training stops after 3 validations without a lower WER.
5. Checks before training: streaming settings unchanged, `hi-IN` on every clip, the tokenizer covers the labels (8 of 2.1M characters cannot be encoded, all ऩ), and the model can overfit 50 clips (loss 123 → 10).
6. The best checkpoint transcribes the test set at 560 ms and 1,120 ms.

## Export and Quantization

[nemo-fine-quan.ipynb](notebooks/quantization/nemo-fine-quan.ipynb) exports the fine-tuned model with the official sherpa-onnx script (`scripts/nemo/nemotron-3.5-asr-streaming-0.6b/export_onnx.py`, changed to load the local checkpoint):

1. Three ONNX graphs at 560 ms: encoder (with cache inputs and the `prompt_index` input), decoder and joiner.
2. INT8 is the script default: ONNX Runtime dynamic quantization, per-tensor, for all three graphs.
3. Size: FP32 2.47 GB, INT8 650 MB (encoder 627 MB).

[nemo-ft-v1-int8-test.ipynb](notebooks/quantization/nemo-ft-v1-int8-test.ipynb) runs the INT8 model in sherpa-onnx on CPU with real chunk-by-chunk streaming at 560 ms, and scores the full test set.

## Android App

[android/](android/) is the sherpa-onnx v2.0.3 Android demo (Apache-2.0, see [LICENSE-sherpa-onnx](android/LICENSE-sherpa-onnx)) with these changes:

1. Model case `1001` in `OnlineRecognizer.kt` for `assets/nemotron-v1-560ms/`.
2. `stream.setOption("language", "hi-IN")` in `MainActivity.kt`, after each new stream and each reset.
3. `noCompress 'onnx'` in `app/build.gradle` and `-Xmx4096m` in `gradle.properties`, so the 650 MB model builds.

Files not in git:

| Files | Source | Copy to |
|---|---|---|
| `libsherpa-onnx-jni.so`, `libonnxruntime.so` | `sherpa-onnx-v2.0.3-android.tar.bz2` (GitHub releases), `arm64-v8a` | `android/app/src/main/jniLibs/arm64-v8a/` |
| `encoder.int8.onnx`, `decoder.int8.onnx`, `joiner.int8.onnx`, `tokens.txt` | The INT8 export | `android/app/src/main/assets/nemotron-v1-560ms/` |

The app runs live microphone captions in an arm64 emulator.


# Results

## Zero-shot

1. Each model transcribes the full OpenSLR 104 test set (2,942 clips, 4.76 h) with no training, one notebook per model in [notebooks/zero-shot/](notebooks/zero-shot/), on a Colab L4 GPU.
2. Outputs are scored with the normalizer and metrics above.
3. [rescore-zeroshot.ipynb](notebooks/analysis/rescore-zeroshot.ipynb) scores the saved outputs again after a scoring fix, with no model run.

| Model | WER | T-WER | Real-time factor |
|---|---|---|---|
| Whisper Hinglish large-v3 (Trelis) | 0.297 | 0.295 | 0.121 |
| Srota | 0.328 | 0.304 | 0.053 |
| Nemotron 3.5 ASR Streaming 0.6B, 1120 ms latency | 0.604 | 0.399 | 0.009 |
| Nemotron 3.5 ASR Streaming 0.6B, 560 ms latency | 0.605 | 0.403 | 0.009 |
| Qwen3-ASR-0.6B | 0.502 | 0.421 | 0.050 |
| Whisper small | 0.934 | 0.908 | 0.234 |

Real-time factor is processing time divided by audio length on the L4, so lower is faster.

- The first scores mapped Devanagari loanwords to English before WER, so WER equalled T-WER. The table uses the fixed scoring: strict WER, and T-WER as in MUCS 2021. Both versions are in [summary.csv](results/summary.csv) (`scoring` v1 and v2).
- Trelis has the lowest WER and T-WER, which supports using it as the distillation teacher.
- Nemotron has the second-highest WER but the third-lowest T-WER: most of its errors are English words in Devanagari (बैकग्राउंड्स for backgrounds), which fine-tuning should fix.
- Nemotron is the fastest by about 5x, and halving its latency to 560 ms barely changes WER or T-WER.
- Nemotron rows are whole clips with the attention limited to the latency, not chunk-by-chunk streaming.
- Whisper small writes English words in Devanagari and loops on about 6% of clips.
- The 3 test clips longer than 30 s have transcripts that do not match their audio, so every model loses some words there.

## Fine-tuned (v1)

Training ran for 0.91 h on a Colab A100 (40 GB). Validation WER was lowest at step 3,000 and training stopped at step 6,000.

| Validation, 560 ms | WER | T-WER | Empty outputs |
|---|---|---|---|
| Step 0 (no training) | 0.480 | 0.297 | 2 |
| Step 3,000 (best) | 0.224 | 0.223 | 78 |

| Test | WER | T-WER | Empty outputs |
|---|---|---|---|
| Nemotron zero-shot, NeMo, 560 ms | 0.611 | 0.399 | 15 |
| Nemotron v1, NeMo, 560 ms | 0.392 | 0.390 | 220 |
| Nemotron v1, NeMo, 1120 ms | 0.377 | 0.375 | 308 |
| **Nemotron v1, sherpa-onnx INT8, real streaming 560 ms** | **0.4045** | **0.4020** | **236** |

NeMo rows are whole clips with the attention limited to the latency. The sherpa-onnx row is the deployed pipeline.

- Test WER fell by about a third (0.611 → 0.392). The model now writes English words in Latin script, so WER and T-WER are almost equal.
- Test T-WER barely moved (0.399 → 0.390), while validation T-WER fell from 0.297 to 0.223. Validation was easier before training too (0.480 against 0.611), so the test scores are the ones to compare.
- The deployed pipeline (export, INT8 and real streaming together) costs 1.3 points of WER.
- NeMo with `auto` reproduces the `transformers` zero-shot score (0.603 against 0.605 WER), so both pipelines agree.
- The validation leak does not explain the gap: v1 scores 0.217 on leaked clips and 0.224 on the others.
- Fine-tuned Nemotron is still behind Trelis (0.297) and Srota (0.328) on WER.

[nemotron-v1-diagnostics.ipynb](notebooks/analysis/nemotron-v1-diagnostics.ipynb) checks the empty outputs:

- They rise on validation too (2 → 78), so the fine-tune causes them, not only the test data.
- Empty clips are a little shorter (median 4 s against 5 s).
- Without them (2,722 clips), WER is 0.348 and T-WER 0.346 (zero-shot on the same clips: 0.600 and 0.394).
- Most segment boundaries are whole seconds, so clips can cut words or hold extra speech at the edges. The likely cause is training on these edges: the model learns to output nothing for speech without a label.

## Known Limitations (v1)

1. Empty outputs on 8% of test clips.
2. Misses the WER target (≤ 1.3 × Trelis = 0.386).
3. Trained on tutorials only: full English sentences come out in Devanagari.
4. Not evaluated on HiACC, Lahaja or Svarah yet.
5. No phone measurements yet (latency, real-time factor, memory).

v2: re-cut training clips with a forced aligner, and add English and conversational data.


# Future Scope

1. Deploy the model on laptop for live Zoom or Google Meet calls.
2. When Android permits, deploy the model for live calls on Zoom, Meet or WhatsApp video calling on phone.
3. Voxtral Mini 4B Realtime for the laptop.