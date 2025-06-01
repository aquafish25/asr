
# 🗣️ WAVLM ASR Pipeline

**An end-to-end Automatic Speech Recognition (ASR) system using WavLM + GRU + CTC + BART tokenizer, implemented in PyTorch.**

---

## 📌 Overview

This repository implements a lightweight yet powerful ASR pipeline that transcribes spoken English into text using the LibriSpeech dataset. The pipeline leverages:

- **WavLM** for robust feature extraction
- **Bidirectional GRU** for temporal modeling
- **Connectionist Temporal Classification (CTC)** for alignment-free training
- **BART tokenizer** for token-level control and decoding
- **WER/CER metrics** for evaluation

---

## 🎯 Pipeline Stages

### 1. **Audio Processing**
- Automatically downloads LibriSpeech subsets
- Converts all audio to mono 16kHz
- Dynamically pads audio for batching

### 2. **Feature Extraction**
- Uses pre-trained [`WAVLM_BASE`](https://huggingface.co/microsoft/wavlm-base) model from `torchaudio`
- Outputs 768-dimensional embeddings per timestep

### 3. **Sequence Modeling**
- 2-layer Bidirectional GRU (512 hidden units each direction)
- Outputs passed through a linear projection to vocab size

### 4. **CTC Decoding**
- Greedy decoding with blank removal and repetition collapse
- BART tokenizer used for token-level encoding/decoding

---

## 🏗️ Architecture

```text
Audio → WavLM → Bi-GRU ×2 → Linear + LogSoftmax → CTC Loss → Text
```

---


### ⚙️ Customization

| Component         | How to Customize                                  |
|------------------|----------------------------------------------------|
| Dataset           | `subset_size`, `url` parameters in DataLoader     |
| Model Architecture| Modify GRU layers or hidden dimensions            |
| Training          | Tweak `batch_size`, `lr`, `epochs`                |
| Audio Handling    | Pad/resample logic in `LibriSpeechDataset`        |

---



## 🛠️ Dependencies

- [`torch`](https://pytorch.org/)
- [`torchaudio`](https://pytorch.org/audio/stable/)
- [`transformers`](https://huggingface.co/transformers/)
- [`jiwer`](https://github.com/jitsi/jiwer) (for WER/CER computation)


---