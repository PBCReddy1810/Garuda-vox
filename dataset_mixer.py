import os
import random
import yaml
import numpy as np
import librosa
import soundfile as sf
import scipy.signal as signal
from pathlib import Path

def calculate_rms(audio):
    """Calculate the Root Mean Square (RMS) energy of an audio signal."""
    return np.sqrt(np.mean(audio**2))

def apply_reverb(audio, sr):
    """Simulate room reverberation using a simple exponentially decaying white noise impulse response."""
    # Generate a synthetic impulse response
    reverb_time = random.uniform(0.1, 0.5)  # RT60 between 100ms and 500ms
    t = np.arange(0, reverb_time, 1.0 / sr)
    ir = np.random.randn(len(t)) * np.exp(-t * (6.91 / reverb_time)) # 6.91 is ~ln(1000)
    
    # Normalize IR
    ir = ir / np.max(np.abs(ir))
    
    # Convolve with audio
    reverberant = signal.convolve(audio, ir, mode='full')[:len(audio)]
    return reverberant

def apply_clipping(audio):
    """Simulate microphone clipping (hard clipping)."""
    threshold = random.uniform(0.5, 0.9) # Clip values above 50-90% of max
    max_val = np.max(np.abs(audio))
    if max_val == 0:
        return audio
    
    clip_val = threshold * max_val
    clipped = np.clip(audio, -clip_val, clip_val)
    
    # Re-normalize
    return clipped / np.max(np.abs(clipped)) * max_val

def mix_audio(clean, noise, snr_db):
    """Mix clean audio and noise at a specific SNR."""
    # Ensure audio length matches
    if len(clean) > len(noise):
        # Repeat noise if it's shorter to match clean speech length
        repeats = (len(clean) // len(noise)) + 1
        noise = np.tile(noise, repeats)[:len(clean)]
    else:
        # Trim noise if it's longer
        noise = noise[:len(clean)]

    rms_clean = calculate_rms(clean)
    rms_noise = calculate_rms(noise)

    # Avoid division by zero if noise is completely silent
    if rms_noise == 0:
        return clean, noise

    # Calculate required noise RMS based on desired SNR
    snr_linear = 10 ** (snr_db / 20)
    desired_rms_noise = rms_clean / snr_linear

    # Scale noise to reach desired SNR
    noise_scaled = noise * (desired_rms_noise / rms_noise)

    # Mix the audio
    mixed = clean + noise_scaled

    # Prevent digital clipping by normalizing if values exceed 1.0
    max_val = max(np.max(np.abs(mixed)), 1.0)
    mixed = mixed / max_val
    clean_scaled = clean / max_val # Keep clean aligned with mixed amplitude

    return mixed, clean_scaled

def main():
    print("Loading config.yaml...")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    clean_dir = Path(config['paths']['clean_speech_dir'])
    noise_dir = Path(config['paths']['noise_dir'])
    output_dir = Path(config['paths']['output_dir'])
    
    noisy_out_dir = Path("data/processed/train/noisy")
    clean_out_dir = Path("data/processed/train/clean")

    # Clear old generated files to prevent endless dataset growth
    import shutil
    if noisy_out_dir.exists():
        shutil.rmtree(noisy_out_dir)
    if clean_out_dir.exists():
        shutil.rmtree(clean_out_dir)

    # Ensure output directories exist
    noisy_out_dir.mkdir(parents=True, exist_ok=True)
    clean_out_dir.mkdir(parents=True, exist_ok=True)

    snr_levels = config['snr_levels']
    sr = config['sample_rate']

    clean_files = list(clean_dir.glob("*.wav"))
    noise_files = list(noise_dir.glob("*.wav"))

    if not clean_files or not noise_files:
        print("---------------------------------------------------------")
        print("Missing audio files! Please add .wav files to:")
        print(f"- {clean_dir.absolute()}")
        print(f"- {noise_dir.absolute()}")
        print("Then run this script again.")
        print("---------------------------------------------------------")
        # Create directories to help the user place files
        clean_dir.mkdir(parents=True, exist_ok=True)
        noise_dir.mkdir(parents=True, exist_ok=True)
        return

    print(f"Found {len(clean_files)} clean files and {len(noise_files)} noise files.")

    num_augmentations = 40 # Create 40 different noisy versions of each clean file
    for i, clean_path in enumerate(clean_files):
        for aug in range(num_augmentations):
            # Randomly select a noise file and SNR
            noise_path = random.choice(noise_files)
            snr = random.choice(snr_levels)

            # Load audio (mono)
            try:
                clean_audio, _ = librosa.load(clean_path, sr=sr)
                noise_audio, _ = librosa.load(noise_path, sr=sr)
            except Exception as e:
                print(f"Warning: Skipping corrupted or unsupported audio file -> {e}")
                continue
                
            # Random Reverberation (30% chance)
            if random.random() < 0.30:
                clean_audio = apply_reverb(clean_audio, sr)

            # Mix audio
            mixed_audio, clean_scaled = mix_audio(clean_audio, noise_audio, snr)
            
            # Random Clipping (20% chance)
            if random.random() < 0.20:
                mixed_audio = apply_clipping(mixed_audio)

            # Save the pair
            base_name = f"mix_{i:04d}_aug_{aug:02d}_snr{snr}dB"
            sf.write(noisy_out_dir / f"{base_name}.wav", mixed_audio, sr)
            sf.write(clean_out_dir / f"{base_name}.wav", clean_scaled, sr)
            
            print(f"Created {base_name}.wav (Clean: {clean_path.name}, Noise: {noise_path.name})")

if __name__ == "__main__":
    main()
