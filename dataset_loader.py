import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class NoiseSuppressionDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, segment_length=16000*2):
        """
        Args:
            clean_dir: Path to directory with clean audio files
            noisy_dir: Path to directory with mixed noisy audio files
            segment_length: Fixed length of audio chunks to return (in samples). 16000*2 = 2 seconds.
        """
        self.clean_dir = Path(clean_dir)
        self.noisy_dir = Path(noisy_dir)
        self.segment_length = segment_length
        
        # Get matching files by assuming they have the same name in both dirs
        self.noisy_files = sorted(list(self.noisy_dir.glob("*.wav")))
        
    def __len__(self):
        return len(self.noisy_files)
        
    def __getitem__(self, idx):
        noisy_path = self.noisy_files[idx]
        clean_path = self.clean_dir / noisy_path.name
        
        import soundfile as sf
        import numpy as np
        # Load audio safely
        try:
            noisy_data, sr = sf.read(str(noisy_path))
            clean_data, _ = sf.read(str(clean_path))
        except Exception as e:
            # If the file is corrupted on disk, return silent audio to prevent crashing the epoch
            noisy_data = np.zeros(self.segment_length, dtype=np.float32)
            clean_data = np.zeros(self.segment_length, dtype=np.float32)
        
        # Convert to tensor and add channel dimension (channels, time)
        noisy_audio = torch.tensor(noisy_data, dtype=torch.float32).unsqueeze(0) if noisy_data.ndim == 1 else torch.tensor(noisy_data, dtype=torch.float32).T
        clean_audio = torch.tensor(clean_data, dtype=torch.float32).unsqueeze(0) if clean_data.ndim == 1 else torch.tensor(clean_data, dtype=torch.float32).T
        
        # Convert to mono if necessary
        if noisy_audio.shape[0] > 1:
            noisy_audio = torch.mean(noisy_audio, dim=0, keepdim=True)
            clean_audio = torch.mean(clean_audio, dim=0, keepdim=True)
            
        # Ensure fixed segment length (pad or trim)
        noisy_audio = self._pad_or_trim(noisy_audio, self.segment_length)
        clean_audio = self._pad_or_trim(clean_audio, self.segment_length)
        
        return noisy_audio.squeeze(0), clean_audio.squeeze(0)
        
    def _pad_or_trim(self, audio, length):
        if audio.shape[-1] < length:
            # Pad with zeros
            pad_size = length - audio.shape[-1]
            audio = torch.nn.functional.pad(audio, (0, pad_size))
        elif audio.shape[-1] > length:
            # Random crop
            start = torch.randint(0, audio.shape[-1] - length + 1, (1,)).item()
            audio = audio[:, start:start+length]
        return audio

def get_dataloader(clean_dir, noisy_dir, batch_size=4, segment_length=32000):
    dataset = NoiseSuppressionDataset(clean_dir, noisy_dir, segment_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return dataloader

if __name__ == "__main__":
    # Simple test
    print("Testing DataLoader initialization...")
    clean_path = "data/processed/train/clean"
    noisy_path = "data/processed/train/noisy"
    
    if os.path.exists(clean_path) and os.path.exists(noisy_path):
        loader = get_dataloader(clean_path, noisy_path, batch_size=2)
        for noisy, clean in loader:
            print(f"Batch shape: Noisy={noisy.shape}, Clean={clean.shape}")
            break
        print("DataLoader verification PASSED.")
    else:
        print("Dataset directories not found. Run dataset_mixer.py first.")
