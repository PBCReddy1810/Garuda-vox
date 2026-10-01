import torch
import torchaudio

class AudioPreprocessor:
    def __init__(self, n_fft=512, hop_length=256, win_length=512, window_fn=torch.hann_window):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window_fn = window_fn
        
    def transform(self, audio):
        """
        Converts 1D audio waveform to STFT (magnitude and phase).
        Audio expected shape: (batch, time) or (time)
        Returns: 
            magnitude (batch, freq, time)
            phase (batch, freq, time)
        """
        window = self.window_fn(self.win_length).to(audio.device)
        stft_result = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True
        )
        
        magnitude = torch.abs(stft_result)
        phase = torch.angle(stft_result)
        
        return magnitude, phase

    def inverse_transform(self, magnitude, phase, length=None):
        """
        Converts STFT (magnitude and phase) back to 1D audio waveform.
        """
        window = self.window_fn(self.win_length).to(magnitude.device)
        
        # Combine magnitude and phase back to complex tensor
        stft_result = magnitude * torch.exp(1j * phase)
        
        # Apply inverse STFT (Reverting to original PyTorch windowing)
        audio_out = torch.istft(
            stft_result,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            length=length
        )
        return audio_out

def verify_stft():
    """Test function to ensure perfect reconstruction."""
    print("Testing STFT/iSTFT reconstruction...")
    # Generate a dummy audio signal (e.g., sine wave)
    sample_rate = 16000
    t = torch.linspace(0, 1, sample_rate)
    dummy_audio = torch.sin(2 * torch.pi * 440 * t)  # 440 Hz tone
    
    preprocessor = AudioPreprocessor()
    
    # Forward transform
    mag, phase = preprocessor.transform(dummy_audio)
    
    # Inverse transform
    reconstructed = preprocessor.inverse_transform(mag, phase, length=dummy_audio.shape[-1])
    
    # Calculate error
    error = torch.max(torch.abs(dummy_audio - reconstructed)).item()
    print(f"Maximum reconstruction error: {error:.8f}")
    if error < 1e-5:
        print("Verification PASSED: STFT is mathematically lossless.")
    else:
        print("Verification FAILED: Error is too high.")

if __name__ == "__main__":
    verify_stft()
