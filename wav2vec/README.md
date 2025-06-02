
# 🗣️ wav2vec2 Speech Recognition Pipeline

Welcome to the **wav2vec2 ASR pipeline** — a PyTorch-powered, Hugging Face-free, fully customizable training and evaluation loop for automatic speech recognition. Built on top of `torchaudio`’s pretrained Wav2Vec 2.0 model, this repo gives you full control over data handling, training, validation, and decoding — no black-box abstractions here.

## 🚀 What’s Inside?

- 🧠 **Model**: `torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_100H`
- 🧾 **Loss Function**: Custom CTC loss (blank token = 0)
- 📊 **Metric**: Word Error Rate (via `jiwer`)
- 📁 **Data**: Custom `Dataset` for audio-transcript pairs from a CSV
- 🔁 **Training & Validation**: Custom loops with audio padding, gradient clipping, and logging

---

## 📂 Folder Structure

```
.
├── wav2vec2.py        # Main training + evaluation script
├── README.md          # You’re here!
├── your_dataset.csv   # Your audio file paths and transcripts
```

CSV format expected:
```csv
path,transcript
/path/to/audio1.wav,hello world
/path/to/audio2.wav,open the pod bay doors
...
```

---

## ⚙️ Setup

First, clone this repo and install the dependencies:

```bash
git clone https://github.com/yourusername/wav2vec2-asr-pipeline.git
cd wav2vec2-asr-pipeline
pip install torch torchaudio pandas jiwer
```

---

## 🎓 Train Your Model

Update the dataset path in `main()` of `wav2vec2.py`:

```python
df = pd.read_csv("/path/to/your_dataset.csv")
```

Then simply run:

```bash
python wav2vec2.py
```

This will:
1. Split your dataset (80% train, 10% val, 10% test)
2. Train the model (100 epochs by default)
3. Validate each epoch
4. Evaluate on the test set (prints sample predictions and final WER)
5. Save the model to `wav2vec2.pt`

---


## 🛠️ Customization

Feel free to modify:
- 🔡 **Tokenizer** in `tokenize()` if you're using a custom vocabulary
- 📉 **Loss or optimizer** to experiment with training
- 🗂️ **DataLoader batch size or augmentations**
- 📈 **Epoch count / learning rate / scheduler**

---

## ⚠️ Known Issues / Gotchas

- Make sure your audio sample rate matches `16kHz` or it will be resampled.
- Empty or too-short transcripts may be skipped or default to blank token `0`.
- Model’s feature extractor is frozen — adjust that if you want full fine-tuning.

---

## 📬 Questions or Suggestions?

Open an [issue](https://github.com/yourusername/wav2vec2-asr-pipeline/issues) or feel free to fork and customize!

---
