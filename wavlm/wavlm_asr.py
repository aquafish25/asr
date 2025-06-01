import jiwer  # For Word Error Rate (WER) and Character Error Rate (CER) calculations
import torch
import torchaudio  # Audio processing library
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartTokenizer  # Tokenizer for text processing
import torchaudio.pipelines as pipelines  # Pre-trained audio models
from torch.utils.data import Dataset, DataLoader  # For dataset handling

# Set random seed for reproducibility
torch.manual_seed(42)

# Initialize BART tokenizer for text processing
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
VOCAB_SIZE = tokenizer.vocab_size  # Vocabulary size for output layer

class LibriSpeechDataset(Dataset):
    """Custom dataset for LibriSpeech audio data with padding and resampling.
    
    Args:
        root (str): Root directory for dataset storage
        url (str): Dataset subset identifier (e.g., 'train-clean-100')
        subset_size (int): Maximum number of samples to include
    """
    def __init__(self, root="data", url="train-clean-100", subset_size=100):
        self.data = torchaudio.datasets.LIBRISPEECH(root=root, url=url, download=True)
        self.size = min(subset_size, len(self.data))  # Actual dataset size
        self.max_len = self.calc_max_duration()  # Calculate max audio duration

    def calc_max_duration(self):
        """Calculate maximum audio duration in the dataset for padding."""
        max_dur = 0
        for i in range(self.size):
            wav, sr, _, _, _, _ = self.data[i]
            dur = wav.shape[1] / sr  # Duration in seconds
            max_dur = max(max_dur, dur)
        return max_dur

    def pad_audio(self, wav, sr):
        """Pad audio to uniform length based on max duration.
        
        Args:
            wav (Tensor): Audio waveform
            sr (int): Sample rate
            
        Returns:
            Tensor: Padded waveform
        """
        target_len = int(self.max_len * sr)  # Target length in samples
        pad_needed = target_len - wav.shape[1]
        if pad_needed > 0:
            wav = F.pad(wav, (0, pad_needed))  # Apply right padding
        return wav

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """Retrieve and process a single dataset item.
        
        Returns dictionary with:
            audio: Padded waveform tensor
            sr: Sample rate
            text: Transcript string
            spkr: Speaker ID
            chap: Chapter ID
            utt: Utterance ID
        """
        wav, sr, text, spkr, chap, utt = self.data[idx]
        wav = self.pad_audio(wav, sr)
        return {
            "audio": wav,
            "sr": sr,
            "text": text,
            "spkr": spkr,
            "chap": chap,
            "utt": utt,
        }

def get_dataloaders(batch_size=5, subset_size=100):
    """Create training and test dataloaders.
    
    Args:
        batch_size (int): Number of samples per batch
        subset_size (int): Maximum samples per dataset
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    train_set = LibriSpeechDataset(subset_size=subset_size)
    test_set = LibriSpeechDataset(url="test-clean", subset_size=subset_size)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# Initialize WavLM model for feature extraction
wavlm = pipelines.WAVLM_BASE.get_model()
wavlm.eval()  # Set to evaluation mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wavlm = wavlm.to(device)
print(f"WavLM loaded on {device}")

def extract_features(waveform, sample_rate):
    """Extract audio features using WavLM model.
    
    Args:
        waveform (Tensor): Input audio tensor
        sample_rate (int): Original sampling rate
        
    Returns:
        Tensor: Extracted features (batch_size, seq_len, feature_dim)
    """
    # Resample if necessary (WavLM requires 16kHz)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    
    # Convert multi-channel to mono by averaging
    if waveform.dim() == 3 and waveform.shape[1] > 1:
        waveform = waveform.mean(dim=1, keepdim=True)
    
    # Remove channel dimension if exists
    waveform = waveform.squeeze(1) if waveform.dim() == 3 else waveform

    # Extract features without gradient computation
    with torch.no_grad():
        features, _ = wavlm(waveform.to(device))
    return features

class ASRModel(nn.Module):
    """Automatic Speech Recognition model using GRU and CTC loss.
    
    Args:
        feat_dim (int): Input feature dimension
        hid_dim (int): GRU hidden dimension
        n_layers (int): Number of GRU layers
        vocab_size (int): Output vocabulary size
    """
    def __init__(self, feat_dim=768, hid_dim=512, n_layers=2, vocab_size=VOCAB_SIZE):
        super().__init__()
        # Bidirectional GRU for sequence modeling
        self.rnn = nn.GRU(
            input_size=feat_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
        )
        # Output layer to vocabulary size
        self.out = nn.Linear(hid_dim * 2, vocab_size)
        # LogSoftmax for CTC loss compatibility
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x, _ = self.rnn(x)  # Process sequence
        x = self.out(x)      # Convert to vocabulary space
        return self.softmax(x)  # Apply log-softmax

def ctc_decode(indices):
    """Convert CTC output indices to text strings.
    
    Args:
        indices (Tensor): Model output indices (batch_size, seq_len)
        
    Returns:
        list: Decoded text strings
    """
    texts = []
    blank = tokenizer.pad_token_id  # CTC blank token (usually pad token)
    for seq in indices:
        tokens = []
        prev = None
        # Collapse repeated tokens and remove blanks
        for token in seq:
            token_id = token.item()
            if token_id == blank:  # Skip blank tokens
                prev = token_id
                continue

            # Only add token if different from previous
            if token_id != prev:
                tokens.append(token_id)
            prev = token_id

        # Convert token IDs to string
        decoded = tokenizer.decode(tokens, skip_special_tokens=True)
        texts.append(decoded.strip())
    return texts

def compute_ctc_loss(log_probs, targets, in_lens, tgt_lens):
    """Compute Connectionist Temporal Classification (CTC) loss.
    
    Args:
        log_probs (Tensor): Model outputs (log probabilities)
        targets (list): Target token sequences
        in_lens (Tensor): Input sequence lengths
        tgt_lens (Tensor): Target sequence lengths
        
    Returns:
        Tensor: Computed CTC loss
    """
    ctc_loss = nn.CTCLoss(blank=tokenizer.pad_token_id, 
                          reduction="mean", 
                          zero_infinity=True)
    log_probs = log_probs.permute(1, 0, 2)  # CTC requires (seq_len, batch, vocab)
    
    # Convert targets to tensor
    targets = [torch.tensor(t, dtype=torch.long) for t in targets]
    targets = torch.cat(targets)
    
    return ctc_loss(log_probs, targets, in_lens, tgt_lens)

def train(model, loader, optim, epochs=200):
    """Training loop for ASR model.
    
    Args:
        model: ASRModel instance
        loader: Training dataloader
        optim: Optimizer
        epochs: Number of training epochs
    """
    model = model.to(device)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_wer = 0.0
        total_cer = 0.0
        n_batches = 0

        for batch in loader:
            # Extract batch data
            audio, sr, text = batch["audio"], batch["sr"], batch["text"]
            
            # Feature extraction
            feats = extract_features(audio, sr[0])
            
            # Prepare targets
            targets = [tokenizer.encode(t, add_special_tokens=False) for t in text]
            in_lens = torch.full(
                size=(feats.shape[0],),
                fill_value=feats.shape[1],
                dtype=torch.long,
            )
            tgt_lens = torch.tensor([len(t) for t in targets])
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            # Forward pass
            logits = model(feats.to(device))
            
            # Compute loss
            loss = compute_ctc_loss(logits, targets, in_lens, tgt_lens)
            
            # Backpropagation
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            # Update metrics
            total_loss += loss.item()
            n_batches += 1
            
            # Decode predictions
            pred_ids = torch.argmax(logits, dim=-1)
            pred_texts = ctc_decode(pred_ids)
            
            # Calculate WER and CER
            total_wer += jiwer.wer(text, pred_texts)
            total_cer += jiwer.cer(text, pred_texts)
            
        # Calculate epoch averages
        avg_loss = total_loss / n_batches
        avg_wer = total_wer / n_batches
        avg_cer = total_cer / n_batches
        
        # Print progress
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Word WER: {avg_wer:.4f} | "
            f"Char WER: {avg_cer:.4f}"
        )

def evaluate(model, loader):
    """Evaluate model on a single test batch.
    
    Args:
        model: Trained ASRModel
        loader: Test dataloader
    """
    model.eval()
    batch = next(iter(loader))
    audio, sr, text = batch["audio"], batch["sr"], batch["text"]

    with torch.no_grad():
        # Feature extraction
        feats = extract_features(audio, sr[0])
        
        # Model prediction
        logits = model(feats.to(device))
        pred_ids = torch.argmax(logits, dim=-1)
        pred_texts = ctc_decode(pred_ids)

    # Get first sample for demonstration
    ref = text[0]
    pred = pred_texts[0]
    
    # Calculate error rates
    wer = jiwer.wer(ref, pred)
    cer = jiwer.cer(ref, pred)

    # Print results
    print("\nEvaluation Sample:")
    print(f"Reference:  {ref}")
    print(f"Prediction: {pred}")
    print(f"Word WER: {wer:.4f} | Char WER: {cer:.4f}")

if __name__ == "__main__":
    # Main execution pipeline
    train_loader, test_loader = get_dataloaders(batch_size=5, subset_size=100)
    model = ASRModel(feat_dim=768, vocab_size=VOCAB_SIZE)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Train and evaluate
    train(model, train_loader, optim, epochs=50)
    evaluate(model, test_loader)