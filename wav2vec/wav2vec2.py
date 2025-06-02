import pandas as pd
import torch.nn as nn
import torch.optim as optim
import os, tqdm, torch, torchaudio
from torch.utils.data import Dataset, DataLoader
import jiwer  # For Word Error Rate calculation

# Check for GPU availability and set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load pre-trained wav2vec 2.0 model and vocabulary
wav2vec_bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_100H
asr_model = wav2vec_bundle.get_model().to(device)
vocab = wav2vec_bundle.get_labels()  # Character-level vocabulary

class YourDataset(Dataset):
    """Custom Dataset for loading audio files and transcripts.
    
    Args:
        csv_path (str): Path to CSV file containing 'path' and 'transcript' columns
        train_mode (bool): Flag indicating training mode (unused in this implementation)
    """
    
    def __init__(self, csv_path, train_mode=True):
        self.df = pd.read_csv(csv_path)
        self.train_mode = train_mode
        # Create character-to-index mapping using wav2vec's vocabulary
        self.char2idx = {c: i for i, c in enumerate(vocab)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Load and process a single audio sample and its transcript."""
        audio_path = self.df.iloc[idx]['path']
        transcript = self.df.iloc[idx]['transcript']

        # Load and preprocess audio
        wav, sr = torchaudio.load(audio_path)
        # Resample if necessary to match wav2vec's expected sample rate
        if sr != wav2vec_bundle.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, wav2vec_bundle.sample_rate)
        # Convert stereo to mono by averaging channels
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        # Convert transcript to token indices
        tokens = self.tokenize(transcript)
        return wav, tokens, transcript

    def tokenize(self, text):
        """Convert text to sequence of token indices using vocabulary."""
        # Convert to uppercase and replace spaces with '|' (wav2vec's word separator)
        text = text.upper().replace(' ', '|')
        # Map characters to indices, ignoring out-of-vocabulary characters
        tokens = [self.char2idx[char] for char in text if char in self.char2idx]
        return torch.tensor(tokens, dtype=torch.long) if tokens else torch.tensor([0], dtype=torch.long)

    def collate_fn(self, batch):
        """Custom collate function to handle variable-length audio and transcripts."""
        wavs, tokens, transcripts = zip(*batch)
        # Pad audio sequences to longest in batch
        max_len = max(w.shape[1] for w in wavs)
        padded_wavs = [nn.functional.pad(w, (0, max_len - w.shape[1])) for w in wavs]
        wav_batch = torch.stack(padded_wavs).squeeze(1)
        return wav_batch, tokens, transcripts

class CTCLoss(nn.Module):
    """Connectionist Temporal Classification (CTC) loss wrapper."""
    def __init__(self, blank=0):
        super().__init__()
        # blank=0 is the CTC blank token index
        self.ctc = nn.CTCLoss(blank=blank, zero_infinity=True)

    def forward(self, log_probs, targets, in_lens, tgt_lens):
        return self.ctc(log_probs, targets, in_lens, tgt_lens)

def decode_predictions(emissions, vocab):
    """Convert model emissions to text predictions.
    
    Args:
        emissions (torch.Tensor): Raw model output [batch, seq_len, vocab_size]
        vocab (list): List of vocabulary characters
        
    Returns:
        list: Decoded text predictions for each batch item
    """
    # Apply log softmax for probability distribution
    emissions = torch.log_softmax(emissions, dim=-1)
    # Get most probable character indices
    indices = torch.argmax(emissions, dim=-1)
    results = []
    for seq in indices:
        decoded = []
        prev = -1  # Track previous character to remove duplicates
        for idx in seq.cpu().numpy():
            if idx and idx != prev:  # Skip blank (0) and repeated characters
                char = vocab[idx]
                # Convert word separator back to space
                if char == '|':
                    char = ' '
                decoded.append(char)
            prev = idx
        results.append(''.join(decoded))
    return results

def train_model(model, train_loader, val_loader, epochs=5, lr=5e-5):
    """Main training loop with validation.
    
    Args:
        model: wav2vec 2.0 model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        Trained model
    """
    # Freeze feature extractor layers
    for param_name, param in model.named_parameters():
        if "feature_extractor" in param_name:
            param.requires_grad = False

    # Optimizer only for trainable parameters
    optim = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = CTCLoss()

    for epoch in range(epochs):
        model.train()
        total_loss, samples = 0.0, 0

        # Training loop
        for batch_idx, (wavs, tokens, _) in enumerate(tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            # Skip empty batches
            if not tokens or all(len(t) == 0 for t in tokens):
                continue
            wavs = wavs.to(device)

            # Forward pass
            emissions, _ = model(wavs)
            # Prepare for CTC: [seq_len, batch, vocab_size]
            log_probs = nn.functional.log_softmax(emissions, dim=2).permute(1, 0, 2)
            in_lens = torch.full((wavs.size(0), emissions.size(1), dtype=torch.long, device=device)

            # Prepare target sequences
            targets, tgt_lens = [], []
            for token_seq in tokens:
                if token_seq.numel() > 0:
                    targets.extend(token_seq.tolist())
                    tgt_lens.append(len(token_seq))
                else:  # Handle empty sequences
                    targets.append(0)
                    tgt_lens.append(1)

            targets = torch.tensor(targets, dtype=torch.long, device=device)
            tgt_lens = torch.tensor(tgt_lens, dtype=torch.long, device=device)

            # Compute loss
            loss = criterion(log_probs, targets, in_lens, tgt_lens)
            
            # Backpropagation
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optim.step()

            # Update metrics
            total_loss += loss.item() * wavs.size(0)
            samples += wavs.size(0)

        # Calculate epoch loss
        avg_loss = total_loss / samples if samples else float('inf')
        print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f}")

        # Validation
        val_loss = validate(model, val_loader, criterion, epoch)
        print(f"Epoch {epoch+1} | Val Loss: {val_loss:.4f}")

    return model

def validate(model, loader, criterion, epoch):
    """Validation loop."""
    model.eval()
    total_loss, samples = 0.0, 0

    with torch.no_grad():
        for wavs, tokens, _ in tqdm.tqdm(loader, desc=f"Epoch {epoch+1} Validation"):
            if not tokens or all(len(t) == 0 for t in tokens):
                continue
            wavs = wavs.to(device)

            # Forward pass
            emissions, _ = model(wavs)
            log_probs = nn.functional.log_softmax(emissions, dim=2).permute(1, 0, 2)
            in_lens = torch.full((wavs.size(0), emissions.size(1), dtype=torch.long, device=device)

            # Prepare targets
            targets, tgt_lens = [], []
            for token_seq in tokens:
                if token_seq.numel() > 0:
                    targets.extend(token_seq.tolist())
                    tgt_lens.append(len(token_seq))
                else:
                    targets.append(0)
                    tgt_lens.append(1)

            targets = torch.tensor(targets, dtype=torch.long, device=device)
            tgt_lens = torch.tensor(tgt_lens, dtype=torch.long, device=device)

            # Calculate loss
            loss = criterion(log_probs, targets, in_lens, tgt_lens)
            total_loss += loss.item() * wavs.size(0)
            samples += wavs.size(0)

    return total_loss / samples if samples else float('inf')

def evaluate_model(model, loader):
    """Evaluate model performance using Word Error Rate (WER)."""
    model.eval()
    total_wer, samples = 0.0, 0

    # Text normalization pipeline
    transformation = jiwer.Compose([
        jiwer.RemovePunctuation(),  # Remove punctuation
        jiwer.ToLowerCase(),         # Convert to lowercase
        jiwer.RemoveMultipleSpaces(),  # Collapse multiple spaces
        jiwer.Strip()                # Remove leading/trailing whitespace
    ])

    with torch.no_grad():
        for wavs, _, transcripts in tqdm.tqdm(loader, desc="Testing"):
            wavs = wavs.to(device)
            emissions, _ = model(wavs)
            predictions = decode_predictions(emissions, vocab)

            # Calculate WER for each sample
            for i in range(len(transcripts)):
                ref_clean = transformation(transcripts[i])
                hyp_clean = transformation(predictions[i])

                wer = jiwer.wer(ref_clean, hyp_clean)  # Word Error Rate
                total_wer += wer
                samples += 1
                
                # Print first 5 samples for inspection
                if samples <= 5:
                    print(f"Reference: {transcripts[i]}")
                    print(f"Predicted: {predictions[i]}")
                    print(f"Cleaned Reference: {ref_clean}")
                    print(f"Cleaned Predicted: {hyp_clean}")
                    print(f"WER: {wer:.4f}\n")

    avg_wer = total_wer / samples
    print(f"Test WER: {avg_wer:.4f}")
    return avg_wer

def main():
    """Main execution pipeline:
    1. Load and split dataset
    2. Create data loaders
    3. Train and validate model
    4. Evaluate on test set
    5. Save model
    """
    # Load dataset (replace with actual path)
    df = pd.read_csv("/path/to/your/dataset.csv")
    # Shuffle and split into train/val/test (80/10/10)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    n = len(df)
    n_test = int(0.1 * n)
    n_val = int(0.1 * n)
    n_train = n - n_val - n_test

    # Create splits
    df_test = df.iloc[:n_test]
    df_val = df.iloc[n_test:n_test+n_val]
    df_train = df.iloc[n_test+n_val:]
    
    print(f"Samples - Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    # Save temporary CSV files
    train_csv, val_csv, test_csv = "train.csv", "val.csv", "test.csv"
    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv, index=False)
    df_test.to_csv(test_csv, index=False)

    # Create datasets and data loaders
    train_ds = YourDataset(train_csv)
    val_ds = YourDataset(val_csv, train_mode=False)
    test_ds = YourDataset(test_csv, train_mode=False)

    train_dl = DataLoader(
        train_ds, 
        batch_size=4, 
        shuffle=True, 
        collate_fn=train_ds.collate_fn, 
        num_workers=2
    )
    val_dl = DataLoader(
        val_ds, 
        batch_size=4, 
        shuffle=False, 
        collate_fn=val_ds.collate_fn, 
        num_workers=2
    )
    test_dl = DataLoader(
        test_ds, 
        batch_size=4, 
        shuffle=False, 
        collate_fn=test_ds.collate_fn, 
        num_workers=2
    )

    # Initialize model
    model = wav2vec_bundle.get_model().to(device)
    # Train model
    model = train_model(model, train_dl, val_dl, epochs=100)
    # Final evaluation
    test_wer = evaluate_model(model, test_dl)

    # Save trained model
    torch.save(model.state_dict(), "wav2vec2.pt")
    print("Model saved to wav2vec2.pt")

    # Cleanup temporary files
    for f in [train_csv, val_csv, test_csv]:
        os.remove(f)

    print(f"Final WER: {test_wer:.4f}")

if __name__ == "__main__":
    main()