This is NOT the final plan. DO NOT USE IT AS A ROADMAP.

# Success criteria

| Metric | Target | Measurement |
|---|---|---|
| Hinglish WER | ≤ 1.3 × the WER of the Whisper Hinglish large-v3 ceiling (Architecture, model 3) | OpenSLR 104 test, HiACC |
| WER loss from quantization | ≤ 1.0 absolute point, INT8 against FP32 | All test sets |
| Word latency | P50 ≤ 700 ms, P90 ≤ 1,000 ms | On-device timestamps |
| Real-time factor (RTF) | ≤ 0.2 on CPU, ≤ 0.1 on NPU | Compute time ÷ audio time |
| Model size | ≤ 130 MB (INT8) | Asset size |
| Peak RAM | ≤ 300 MB | Android Studio memory profiler |
| Script accuracy | ≥ 95% of English words in Latin script | Word-level script check |

Assumptions:
- Primary test phone: Snapdragon 8 Gen 2 or later (CPU and NPU path).
- Second test phone: mid-range ARM64 phone (CPU path only).
- Compute: approximately 150 Colab compute units.

# Data

## Training mix

| Dataset | Hours used (start value) | Share | Reason |
|---|---|---|---|
| OpenSLR 104 (train split) | All (~90 h) | ~55% | Only code-switched training data |
| Kathbath | ~50 h | ~30% | Keeps Hindi accuracy. Adds speaker variety. |
| TIE_shorts | ~25 h | ~15% | Keeps Indian English accuracy |

Train on everything to see if performance improves, but for version 1, if the error metric is ok with 1 dataset, it's ok.

Change the mix only after the first dev results. Record each change in the report.

## Data rules

1. **No test leaks.** OpenSLR 104 is in train and test. Train only on the train split. Do not use the 3.14k test rows of the Parquet copy for training. Compare these rows with the official OpenSLR 104 test list. Remove duplicate utterances by normalized transcript and audio duration.
2. **Tutorial overlap.** OpenSLR 104 comes from spoken tutorials. If the metadata has tutorial or speaker IDs, make sure that the dev split has no overlap with the train split.
3. **Kathbath cleaning.** Kathbath writes English words in Devanagari. Convert these words to Latin script with the script rules below.
4. **OpenSLR 104 cleaning.** OpenSLR 104 writes some English words in Latin script and some in Devanagari. Apply the same script rules.
5. **TIE_shorts cleaning.** Remove or write out math symbols and code text. Remove utterances longer than 20 s.
6. **Licenses.** Record the license of each dataset in the repository. OpenSLR 104 is CC BY-SA 4.0. Check the ShareAlike terms before you publish model weights. Check the Kathbath and Lahaja licenses.

## Script rules (text normalization)

Use one normalizer for training labels, model output and test references.

1. Apply Unicode NFC normalization. Keep Devanagari matras and nuktas.
2. Remove punctuation. Convert all text to lowercase for Latin script.
3. Write numbers as spoken words.
4. Tag each word as Devanagari or Latin.
5. For each Devanagari word, transliterate it to Latin script with IndicXlit. Compare the result with an English word list.
6. If the word is of English origin, write it in Latin script. If not, keep the Devanagari form.
7. Make one fixed loanword list for borderline words (for example, words that are common in Hindi text in Devanagari). Use this list for all data.
8. Audit 300 random utterances by hand. Continue only when ≥ 95% of the words follow these rules.

## Augmentation

- Speed perturbation: 0.9, 1.0, 1.1.
- Background noise and room impulse responses.
- SpecAugment.
- Phone-channel simulation: band-limit, encode with a low-bitrate codec, decode.

## Domain gap

| Set | Domain | Expected result |
|---|---|---|
| OpenSLR 104 test | Same domain as training (tutorials) | Best WER |
| HiACC | Read and spontaneous, adults and children | Higher WER. Child speech is the most difficult. |
| Lahaja | Hindi, many accents | Shows accent robustness |
| Svarah | Indian English, many speakers | Shows English robustness |

Report in-domain and out-of-domain results separately. If out-of-domain WER is too high, add more data from this list:

| Dataset | Language | Reason |
|---|---|---|
| Vaani (Hindi subset) | Hinglish, spontaneous | English words labeled in both Latin and Devanagari script. Large untranscribed part for pseudo-labels. |
| CS-FLEURS (Hindi-English) | Hinglish | Real and synthetic code-switched speech |
| IndicVoices (Hindi subset) | Hindi, spontaneous | More conversational Hindi |

# Architecture

## Candidates

The following architectures are tried to select the best one:

| # | Model | Link | Method |
|---|---|---|---|
| 1 | Whisper small | https://huggingface.co/openai/whisper-small | Zero-shot for baseline. LocalAgreement streaming on the phone. |
| 2 | Whisper small | https://huggingface.co/openai/whisper-small | LoRA fine-tune on the training mix. LocalAgreement streaming on the phone. Fine-tuned baseline. |
| 3 | Whisper Hinglish large-v3 (Trelis) | https://huggingface.co/Trelis/whisper-hinglish-preview | Zero-shot, offline, on Colab GPU. Accuracy ceiling. Not on the phone. |
| 4 | IndicConformer Hindi (120M, hybrid CTC-RNNT) | https://huggingface.co/ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large | Convert to cache-aware streaming. New mixed-script tokenizer. Full fine-tune. |
| 5 | FastConformer Hybrid Large Streaming Multi (114M, English) | https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_streaming_multi | Already cache-aware with multi-look-ahead. New mixed-script tokenizer. Full fine-tune. |
| 6 | Streaming Zipformer transducer (optional) | https://huggingface.co/Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17 | New mixed-script vocabulary. Full fine-tune. Only if compute remains. |

## Comparison

| | Whisper (1, 2) | IndicConformer (4) | FastConformer streaming (5) | Zipformer (6) |
|---|---|---|---|---|
| Streaming method | Re-decode a growing buffer | Cache-aware, after conversion | Cache-aware, native | Chunked, native |
| Expected latency | 2–4 s | 160–1,120 ms | 160–1,120 ms | 300–600 ms |
| Start knowledge | Multilingual, some code-switching | Hindi acoustics | English acoustics, streaming | English acoustics |
| Tokenizer for Devanagari | Too many tokens per word | New, small | New, small | New, small |
| Static INT8 / NPU | Poor | Good | Good | Good |
| Android runtime | whisper.cpp | sherpa-onnx | sherpa-onnx | sherpa-onnx |

## Why Whisper is the baseline only

- Whisper-Streaming with LocalAgreement gave 3.3 s average latency for English, on an A40 GPU.
- The Whisper tokenizer makes too many tokens for Indian languages. This makes decoding slow.
- Static INT8 quantization causes a large WER loss on Whisper. NPUs need static quantization.

## Selection method

1. Train candidates 4 and 5 (and 6, if compute remains) for 2 hours each on a 20 h subset of the training mix.
2. Measure each candidate with the 560 ms look-ahead.
3. Export each candidate to dynamic INT8. Measure RTF on the mid-range phone.
4. Reject a candidate if its P90 latency is more than 1,000 ms.
5. Score the remaining candidates:

| Criterion | Weight |
|---|---|
| Hinglish dev WER | 40% |
| P90 latency on device | 25% |
| WER loss from INT8 | 15% |
| Model size | 10% |
| Training cost on Colab | 10% |

6. Train the candidate with the best score on the full training mix.

Expected result: model 4 has better accuracy (Hindi acoustics). Model 5 has a lower risk (native streaming). The pilot decides.

# Fine-tuning

## Tokenizer

1. Train a SentencePiece model on all normalized training text. Vocabulary: 1,024 to 2,048 tokens.
2. Make sure that both scripts have good coverage. Measure the tokens per word for each script.
3. Compare tokens per word with the Whisper tokenizer. Put the result in the report.
4. Replace the decoder and joint vocabulary. Keep the encoder weights.

## Streaming configuration

| `att_context_size` [left, right] | Chunk (algorithmic latency) | Use |
|---|---|---|
| [70, 0] | 80 ms | Ablation |
| [70, 1] | 160 ms | Ablation |
| [70, 3] | 320 ms | Fallback |
| [70, 6] | 560 ms | Primary |
| [70, 13] | 1,120 ms | Ablation (too slow for P90 target) |

Train with all five look-ahead values. Then select the latency on the device without a second model.

## Fine-tuning technique comparison

| Technique | Use for | Advantages | Disadvantages |
|---|---|---|---|
| Full fine-tune | Models 4, 5, 6 (selected) | Best accuracy. Required after a vocabulary change. | More memory. Risk of forgetting. |
| LoRA | Model 2 (Whisper) | Low memory | Less gain than full fine-tune on a small model |
| Freeze lower encoder layers | Pilot only | Faster | Lower accuracy on new accents |

## Training stages

| Stage | Data | Task | Exit condition |
|---|---|---|---|
| 1. Vocabulary change and streaming fine-tune | Training mix | CTC + RNNT losses (start with 0.3 CTC, 0.7 RNNT) | Dev WER stops decreasing for 3 evaluations |
| 2. Pseudo-labels (optional) | Stage 1 data + untranscribed Vaani audio labeled by model 3 | Same losses | Out-of-domain WER decreases |
| 3. Quantization-aware training (if necessary) | 10–20% of the training mix | Fake quantization in the encoder | INT8 WER loss ≤ 1.0 point |

Pseudo-label filter for stage 2: keep an utterance only if it has no repeated n-grams and a normal ratio of characters to audio seconds.

## Start values

- Optimizer: AdamW. Peak learning rate: 1e-4 to 3e-4. Warmup: 5–10%. Cosine decay.
- Precision: bf16 on A100 and L4. fp16 on T4 (T4 does not support bf16).
- Batch by total audio duration. Use gradient accumulation to reach approximately 10 minutes of audio per update.
- Maximum utterance length: 20 s.

## Colab practice

- Convert the data to tar shards before training. Store the shards on the Hugging Face Hub or Google Drive.
- Save a checkpoint every 30 minutes. Make the training script resume from the last checkpoint.
- Do data preparation and evaluation on CPU, T4 or L4. Use A100 only for training.

# Quantization

## Methods

| Method | Target | Calibration data | Expected effect | Decision |
|---|---|---|---|---|
| FP16 | GPU, NPU | No | Almost no loss. 2× smaller. | Reference |
| Dynamic INT8 | CPU (sherpa-onnx) | No | Small loss. 4× smaller. | Step 1: CPU path |
| Static INT8, A8W8 | NPU (QNN) | Yes | Larger loss. Fastest. | Step 2: NPU path |
| A16W8 | NPU | Yes | Less loss than A8W8 | Fallback |
| A16W4 | NPU | Yes | Smallest. Highest risk. | Ablation |
| QAT, then static INT8 | NPU | Training data | Recovers loss | Only if A8W8 loss > 1.0 point |

## Mixed-precision rules

1. Quantize the encoder linear and convolution layers with per-channel weights.
2. Keep the RNNT prediction network (LSTM) and the joint network at FP16 or FP32. LSTMs lose much accuracy at low precision.
3. Keep the subsampling layers and softmax inputs at 16 bits if the error analysis shows loss there.
4. Calibrate with 300–500 dev utterances. Include code-switched, Hindi, English, noisy and quiet audio.
5. Do a layer-sensitivity analysis. Quantize one layer group at a time. Record the WER change for each group.

## Acceptance

Accept the smallest precision with a WER loss of ≤ 1.0 absolute point.

# Streaming on device

## Pipeline

1. Capture audio with `AudioRecord`: 16 kHz, mono, 16-bit PCM, 20 ms buffers.
2. Write the audio into a lock-free ring buffer.
3. Calculate 80-dimension log-mel features as the audio arrives.
4. When 560 ms of new features are available, run one encoder step. Send the cache tensors in. Receive the new cache tensors.
5. Run greedy RNNT decoding on the new encoder frames.
6. Show the new tokens as a partial result (grey text).
7. Detect the end of an utterance with RNNT blank frames or Silero VAD. Start value: 600–800 ms of silence.
8. At the end of an utterance, show the final text (black text). Reset the decoder state.

## Threads

| Thread | Work |
|---|---|
| Audio | Capture only. No allocation. No locks. |
| Inference | Features, encoder, decoder. Pinned to big CPU cores. |
| UI | Text updates only |

## Runtimes

| Runtime | Hardware | Effort | Phase |
|---|---|---|---|
| sherpa-onnx | CPU, all ARM64 phones | Low | 1 |
| sherpa-onnx with QNN | Qualcomm NPU | Medium | 2 (option) |
| ExecuTorch with QNN backend | Qualcomm NPU | High | 2 (option) |
| LiteRT with NPU delegate | Vendor NPU | Medium–high | 2 (option) |

Build the full app on the CPU path first. Then move only the encoder to the NPU. Keep the decoder and joint on the CPU.

## Export test

Some ONNX exports remove the cache inputs and outputs. Then streaming breaks at chunk boundaries. Make a test that compares the streaming output with the offline output on 100 dev utterances. The two outputs must be almost the same.

## Latency budget (560 ms chunks)

| Part | Budget |
|---|---|
| Audio buffer | 20 ms |
| Chunk fill (worst case) | 560 ms |
| Encoder + decoder compute | ≤ 60 ms (CPU), ≤ 20 ms (NPU) |
| UI update | 16 ms |
| **Total, worst case** | **≈ 660 ms (CPU)** |

If CPU compute on the mid-range phone is more than 150 ms per chunk, use the 320 ms chunk or a smaller encoder.

# Evaluation

## Test sets

| Test set | Measures |
|---|---|
| OpenSLR 104 test | In-domain Hinglish |
| HiACC | Out-of-domain Hinglish, spontaneous speech, child speech |
| Lahaja | Hindi accents, forgetting of Hindi |
| Svarah | Indian English, forgetting of English |
| Own recordings (optional, 1–2 h) | Real phone microphone, real conversation |

## Accuracy metrics

| Metric | Description |
|---|---|
| WER, CER | With the project normalizer |
| T-WER | An English word is correct in Latin script or in Devanagari transliteration. WER − T-WER = script errors. |
| Span WER | WER on Hindi spans and English spans, separately |
| Switch-point accuracy | Correct language at each word where the language changes |
| Script accuracy | % of English words in Latin script |

## Latency and efficiency metrics

| Metric | Description |
|---|---|
| First-partial latency | Speech start to first displayed token |
| Word latency | Word end (forced alignment) to display of that word |
| Final latency | End of utterance to final text |
| Partial stability | Displayed words that change later, per 100 words |
| RTF | Compute time ÷ audio time |
| Peak RAM, load time | Android Studio profiler |
| Energy | mAh for 30 min of continuous use. CPU temperature. Thermal throttling. |

## Ablations

| Variable | Values |
|---|---|
| Architecture | Models 1–6 |
| Look-ahead | 80, 160, 320, 560, 1,120 ms |
| Precision | FP32, FP16, dynamic INT8, A8W8, A16W8, A16W4 |
| Runtime | sherpa-onnx CPU, NPU path |
| Device | Flagship phone, mid-range phone |
| Training mix | With and without Kathbath, with and without TIE_shorts |

# Compute budget

Colab rates (March 2026, can change): T4 1.19 units/h, L4 1.71 units/h, A100 (40 GB) 5.40 units/h.

| Task | GPU | Hours (estimate) | Units (estimate) |
|---|---|---|---|
| Data preparation, normalization, shards | CPU / T4 | 10 | 12 |
| Zero-shot runs (models 1 and 3) | L4 | 4 | 7 |
| LoRA fine-tune (model 2) | L4 | 6 | 10 |
| Pilot runs (models 4, 5, 6) | A100 | 6 | 32 |
| Full training (selected model) | A100 | 10 | 54 |
| Quantization-aware training (if necessary) | A100 | 2 | 11 |
| Evaluation and ablations | L4 | 6 | 10 |
| **Total** | | **44** | **≈ 136** |

Measure the real throughput in the pilot. Then calculate the budget again. If A100 is not available, use L4. Time increases approximately 2–3 times.

# Timeline

| Week | Work | Deliverable |
|---|---|---|
| 1 | Normalizer, evaluation harness, zero-shot baselines (models 1, 3) | Baseline results table |
| 2 | Data download, leak check, script cleaning, audit, shards | Clean training mix, audit report |
| 3 | Tokenizer, pilot runs (models 4, 5, 6), LoRA baseline (model 2) | Architecture selection table |
| 4 | Full training of the selected model | FP32 streaming model |
| 5 | Quantization, sensitivity analysis, QAT if necessary | INT8 models, quantization table |
| 6 | Android app on the CPU path, export test | Working real-time app |
| 7 | NPU path for the encoder, on-device measurements | Latency and energy results |
| 8 | Ablations, error analysis, report, demo video | Final repository and report |

# Risks

| Risk | Effect | Mitigation |
|---|---|---|
| Script labels are not consistent (OpenSLR 104, Kathbath) | Script errors in output | Script rules, hand audit, script-accuracy metric |
| Test leak from the OpenSLR 104 Parquet copy | Results are too good | Leak check (Data rules 1, 2) |
| Training data is mostly tutorials, lectures and read speech | High WER on spontaneous speech | Out-of-domain test sets, optional extra data |
| Few code-switched training hours (~90 h) | Weak switch-point accuracy | Optional Vaani and CS-FLEURS data, pseudo-labels |
| Cache ports removed in ONNX export | Streaming breaks at chunk boundaries | Export test |
| NPU does not support an operator or dynamic shape | NPU path fails | Fixed chunk shapes, CPU fallback |
| Model forgets Hindi or English | Worse Lahaja and Svarah WER | Mixed training set, monolingual test sets |
| Colab disconnects or A100 not available | Lost time and units | Checkpoints, resume logic, L4 fallback |
| CC BY-SA 4.0 terms on OpenSLR 104 | Limits on weight release | Read terms in week 1, state the license in the model card |

# Deliverables

| Deliverable | Content |
|---|---|
| Repository | Data scripts, normalizer, training configs, export scripts, Android app |
| Technical report | Architecture selection, quantization tables, latency results, ablations, error analysis |
| Model card | Data, licenses, metrics, limits |
| Demo video | Whisper baseline and streaming Conformer side by side on the phone, with a latency overlay |

# References

| Topic | Link |
|---|---|
| Cache-aware streaming FastConformer | https://arxiv.org/abs/2312.17279 |
| Whisper-Streaming (LocalAgreement) | https://arxiv.org/abs/2307.14743 |
| Whisper tokenizer for Indian languages | https://arxiv.org/abs/2412.19785 |
| Whisper-small quantization study | https://arxiv.org/pdf/2511.08093 |
| Quantized Whisper and Conformer (NF4) | https://arxiv.org/pdf/2309.09136 |
| 4-bit RNN-T quantization | https://arxiv.org/pdf/2108.12074 |
| Code-switching ASR: mixed vs sequential training | https://arxiv.org/pdf/2509.24310 |
| T-WER for Hindi-English | https://arxiv.org/pdf/2104.00235 |
| sherpa-onnx | https://github.com/k2-fsa/sherpa-onnx |
| ExecuTorch Qualcomm backend | https://docs.pytorch.org/executorch/1.1/android-qualcomm.html |
| Colab GPU rates | https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/ |
| Vaani dataset | https://huggingface.co/datasets/ARTPARK-IISc/VAANI |
| CS-FLEURS | https://huggingface.co/datasets/byan/cs-fleurs |
