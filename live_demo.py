import sounddevice as sd
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

from streaming_inference import StreamingNoiseSuppressor
from lms_filter import LMSFilter
# Audio configuration
SAMPLE_RATE = 16000
BLOCK_SIZE = 256  # 16ms chunks

class LiveDemo:
    def __init__(self):
        print("Initializing AI Model (Hybrid Complex-CRN + LMS)...")
        self.suppressor = StreamingNoiseSuppressor()
        
        self.running = False
        
    def audio_callback(self, indata, outdata, frames, time_info, status):
        """This runs in a high-priority background thread by sounddevice."""
        if status:
            print(f"Audio Status: {status}")
            
        # Audio from microphone is 2D (frames, channels), we need 1D mono
        audio_in = indata[:, 0]
        
        # Process the frame through our real-time streaming engine
        # In a real deployed edge scenario, we would use the ONNX model here
        enhanced_audio = self.suppressor.process_frame(audio_in)
        
        # Write to output (headphones/speaker)
        outdata[:, 0] = enhanced_audio
        # If stereo output, duplicate to second channel
        if outdata.shape[1] > 1:
            outdata[:, 1] = enhanced_audio

    def start(self):
        print("\n" + "="*50)
        print("Live Microphone Demo Started!")
        print("="*50)
        print("Make some noise! Speak into the mic and listen to the output.")
        print("Press Ctrl+C to stop.")
        
        self.running = True
        try:
            # We open an audio stream connecting the default microphone to the default speakers
            with sd.Stream(samplerate=SAMPLE_RATE, 
                           blocksize=BLOCK_SIZE, 
                           channels=(1, 2), # 1 input (mono mic), 2 output (stereo headphones)
                           callback=self.audio_callback):
                while self.running:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping demo...")
            self.running = False
        except Exception as e:
            print(f"Error accessing microphone: {e}")
            print("Make sure your microphone privacy settings allow access.")

if __name__ == "__main__":
    demo = LiveDemo()
    demo.start()
