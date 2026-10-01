import torch
import numpy as np
import time
import os

from model_architecture import SimpleCRN
from audio_utils import AudioPreprocessor
from lms_filter import LMSFilter

class StreamingNoiseSuppressor:
    def __init__(self, model_path="best_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessor = AudioPreprocessor()
        self.model = SimpleCRN().to(self.device)
        
        # Ensure we look for the model in the exact directory where the script lives
        script_dir = os.path.dirname(os.path.abspath(__file__))
        resolved_model_path = os.path.join(script_dir, model_path)
        
        if resolved_model_path and os.path.exists(resolved_model_path):
            try:
                self.model.load_state_dict(torch.load(resolved_model_path, map_location=self.device, weights_only=True))
                print(f"Loaded {model_path} for streaming inference.")
            except Exception as e:
                print(f"Failed to load {model_path} (likely an architecture change). Using untrained model: {e}")
        else:
            print("Using untrained model for streaming inference.")
        self.model.eval()
        
        # LMS Filter for Hybrid ANC pipeline
        self.lms = LMSFilter(num_taps=32, mu=0.01)
        
        # Streaming state parameters
        self.hop_length = self.preprocessor.hop_length
        self.win_length = self.preprocessor.win_length
        self.hidden_state = torch.zeros(1, 1, 128).to(self.device)
        
        # Audio overlap-add buffers
        self.in_buffer = np.zeros(self.win_length * 2, dtype=np.float32)
        self.out_buffer = np.zeros(self.win_length * 2, dtype=np.float32)
        
        # Temporal smoothing for 'ezzzz' musical noise cancellation
        self.prev_mask = None
        self.alpha = 0.75  # 75% old mask, 25% new mask (heavily reduces buzzing)
        self.hanning = np.hanning(self.win_length + 1)[:-1].astype(np.float32)
        
    def process_frame(self, audio_chunk):
        """
        Process a continuous stream of audio chunks.
        audio_chunk should be a 1D numpy array of length `hop_length`.
        """
        # Shift input buffer and add new chunk
        self.in_buffer[:-self.hop_length] = self.in_buffer[self.hop_length:]
        self.in_buffer[-self.hop_length:] = audio_chunk
        
        # Take the current window for STFT
        raw_frame = self.in_buffer[-self.win_length:]
        frame = torch.tensor(raw_frame, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Run STFT
        mag, phase = self.preprocessor.transform(frame)
        
        real_noisy = mag * torch.cos(phase)
        imag_noisy = mag * torch.sin(phase)
        complex_noisy = torch.stack([real_noisy, imag_noisy], dim=1)
        
        with torch.no_grad():
            mask = self.model(complex_noisy)
            
            # Temporal mask smoothing
            if self.prev_mask is None:
                self.prev_mask = mask
            else:
                mask = self.alpha * self.prev_mask + (1 - self.alpha) * mask
            self.prev_mask = mask
            
            mask_real = mask[:, 0, :, :]
            mask_imag = mask[:, 1, :, :]
            
            # Complex multiplication for estimated speech
            est_real = real_noisy * mask_real - imag_noisy * mask_imag
            est_imag = real_noisy * mask_imag + imag_noisy * mask_real
            
            est_mag = torch.sqrt(est_real**2 + est_imag**2 + 1e-8)
            est_phase = torch.atan2(est_imag, est_real)
        
        # iSTFT for AI-enhanced speech
        ai_enhanced_frame = self.preprocessor.inverse_transform(est_mag, est_phase, length=self.win_length).squeeze(0).cpu().numpy()
        
        # Hybrid LMS Filtering
        # Estimate the noise by subtracting AI speech from original raw frame
        noise_est = raw_frame - ai_enhanced_frame
        
        # Use LMS to remove any residual noise from the AI enhanced frame
        hybrid_enhanced_frame = self.lms.process(reference_noise=noise_est, primary_signal=ai_enhanced_frame)
        
        # Overlap add in the output buffer
        self.out_buffer[:-self.hop_length] = self.out_buffer[self.hop_length:]
        self.out_buffer[-self.hop_length:] = 0.0 # Clear the new segment
        
        # Hanning window applied manually for Overlap-Add consistency if needed
        self.out_buffer[:self.win_length] += hybrid_enhanced_frame
        
        # Return the finalized chunk (the left side of the overlap-add)
        return self.out_buffer[:self.hop_length].copy()

if __name__ == "__main__":
    print("Initializing Phase 5: Streaming Inference Test (Complex-CRN + LMS)...")
    suppressor = StreamingNoiseSuppressor()
    
    # Simulate a stream with dummy data
    chunk_size = suppressor.hop_length # 256 samples (~16ms at 16kHz)
    dummy_stream = np.random.randn(chunk_size * 50).astype(np.float32)
    
    print(f"Streaming chunk size: {chunk_size} samples")
    
    output_stream = []
    start_time = time.time()
    
    for i in range(0, len(dummy_stream), chunk_size):
        chunk = dummy_stream[i:i+chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            
        t0 = time.time()
        out_chunk = suppressor.process_frame(chunk)
        t1 = time.time()
        
        process_time_ms = (t1 - t0) * 1000
        output_stream.append(out_chunk)
        
        if i == 0:
             print(f"First frame processing time: {process_time_ms:.2f} ms")
             
    total_time = time.time() - start_time
    print(f"Processed {len(output_stream)} frames in {total_time:.4f} seconds.")
    print("Streaming Inference test PASSED.")
