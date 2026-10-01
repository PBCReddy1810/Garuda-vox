import numpy as np

class LMSFilter:
    """
    Least Mean Squares (LMS) Adaptive Filter for residual noise suppression.
    Typically used to clean up persistent hums or predictable static that the neural network misses.
    """
    def __init__(self, num_taps=64, mu=0.01):
        self.num_taps = num_taps
        self.mu = mu
        self.weights = np.zeros(num_taps)
        self.buffer = np.zeros(num_taps)

    def process(self, reference_noise, primary_signal):
        """
        reference_noise: Estimated noise profile
        primary_signal: The signal containing speech + residual noise
        Returns: Filtered signal (residual noise canceled)
        """
        output = np.zeros_like(primary_signal)
        
        for n in range(len(primary_signal)):
            # Update buffer with new noise sample
            self.buffer[1:] = self.buffer[:-1]
            self.buffer[0] = reference_noise[n]
            
            # Predict noise using current filter weights
            y = np.dot(self.weights, self.buffer)
            
            # Error (this is actually the cleaned speech signal!)
            e = primary_signal[n] - y
            output[n] = e
            
            # Update weights to minimize error
            self.weights += 2 * self.mu * e * self.buffer
            
        return output

if __name__ == "__main__":
    print("Testing LMS Filter...")
    # Dummy signals
    t = np.linspace(0, 1, 16000)
    clean_speech = np.sin(2 * np.pi * 500 * t)  # 500 Hz tone
    noise = np.sin(2 * np.pi * 120 * t)         # 120 Hz hum
    
    mixed_signal = clean_speech + noise
    
    lms = LMSFilter(num_taps=32, mu=0.05)
    
    # We pass the pure noise as the reference (ideal scenario)
    cleaned = lms.process(noise, mixed_signal)
    print("LMS Filter passed basic execution test.")
