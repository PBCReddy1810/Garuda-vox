import torch
import os
import soundfile as sf
import numpy as np
from pathlib import Path
from pystoi import stoi

import warnings
warnings.filterwarnings("ignore")

try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False

from model_architecture import SimpleCRN
from audio_utils import AudioPreprocessor

def calculate_metrics(clean_audio, noisy_audio, enhanced_audio, sr=16000):
    """
    Calculate STOI and PESQ for noisy and enhanced audio compared to clean.
    Audio inputs should be 1D numpy arrays.
    """
    metrics = {}
    
    # STOI (Speech Transmission Index) - higher is better (0 to 1)
    metrics['stoi_noisy'] = stoi(clean_audio, noisy_audio, sr, extended=False)
    metrics['stoi_enhanced'] = stoi(clean_audio, enhanced_audio, sr, extended=False)
    
    # PESQ (Perceptual Evaluation of Speech Quality) - higher is better (-0.5 to 4.5)
    if PESQ_AVAILABLE:
        try:
            # 'wb' is wideband (16kHz)
            metrics['pesq_noisy'] = pesq(sr, clean_audio, noisy_audio, 'wb')
            metrics['pesq_enhanced'] = pesq(sr, clean_audio, enhanced_audio, 'wb')
        except Exception as e:
            print(f"PESQ Error: {e}")
            metrics['pesq_noisy'] = 0.0
            metrics['pesq_enhanced'] = 0.0
    else:
        metrics['pesq_noisy'] = 0.0
        metrics['pesq_enhanced'] = 0.0
        
    return metrics

def evaluate_model(model_path="best_model.pth"):
    print("Initializing Phase 4: Evaluation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    preprocessor = AudioPreprocessor()
    model = SimpleCRN().to(device)
    
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model weights from {model_path}")
    else:
        print("No trained weights provided. Evaluating with random initialization (untrained model).")
        print("Note: Since the model is untrained, the enhanced metrics will likely be WORSE than the noisy metrics.")
        
    model.eval()
    
    clean_dir = Path("data/processed/train/clean")
    noisy_dir = Path("data/processed/train/noisy")
    
    if not clean_dir.exists() or not noisy_dir.exists():
        print("Data directories not found. Run Phase 1 first.")
        return
        
    test_files = list(noisy_dir.glob("*.wav"))[:5] # Test on first 5 files
    if not test_files:
        print("No test files found.")
        return
        
    sr = 16000
    all_metrics = []
    
    with torch.no_grad():
        for noisy_path in test_files:
            clean_path = clean_dir / noisy_path.name
            
            # Load audio
            noisy_audio, _ = sf.read(str(noisy_path))
            clean_audio, _ = sf.read(str(clean_path))
            
            # Ensure 1D and same length
            if noisy_audio.ndim > 1: noisy_audio = noisy_audio[:, 0]
            if clean_audio.ndim > 1: clean_audio = clean_audio[:, 0]
            
            min_len = min(len(noisy_audio), len(clean_audio))
            noisy_audio = noisy_audio[:min_len]
            clean_audio = clean_audio[:min_len]
            
            # Convert to tensor
            noisy_tensor = torch.tensor(noisy_audio, dtype=torch.float32).unsqueeze(0).to(device)
            
            # STFT
            mag, phase = preprocessor.transform(noisy_tensor)
            
            # Forward pass
            mag = mag.unsqueeze(1) # Add channel
            mask = model(mag)
            
            # Apply mask
            estimated_mag = (mag * mask).squeeze(1)
            
            # iSTFT
            enhanced_tensor = preprocessor.inverse_transform(estimated_mag, phase, length=min_len)
            enhanced_audio = enhanced_tensor.squeeze(0).cpu().numpy()
            
            # Calculate metrics
            m = calculate_metrics(clean_audio, noisy_audio, enhanced_audio, sr)
            all_metrics.append(m)
            
            print(f"File: {noisy_path.name}")
            print(f"  Noisy STOI: {m['stoi_noisy']:.4f} -> Enhanced STOI: {m['stoi_enhanced']:.4f}")
            if PESQ_AVAILABLE:
                print(f"  Noisy PESQ: {m['pesq_noisy']:.4f} -> Enhanced PESQ: {m['pesq_enhanced']:.4f}")
                
    # Averages
    avg_stoi_n = np.mean([m['stoi_noisy'] for m in all_metrics])
    avg_stoi_e = np.mean([m['stoi_enhanced'] for m in all_metrics])
    print("\n--- Average Metrics ---")
    print(f"Average STOI: {avg_stoi_n:.4f} -> {avg_stoi_e:.4f}")
    if PESQ_AVAILABLE:
        avg_pesq_n = np.mean([m['pesq_noisy'] for m in all_metrics])
        avg_pesq_e = np.mean([m['pesq_enhanced'] for m in all_metrics])
        print(f"Average PESQ: {avg_pesq_n:.4f} -> {avg_pesq_e:.4f}")
        
    print("Phase 4 verified!")

if __name__ == "__main__":
    evaluate_model()
