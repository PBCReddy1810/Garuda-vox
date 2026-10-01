# AI Real-Time Noise Suppression System (Complex-CRN + LMS)

An end-to-end, ultra-low latency AI noise suppression system built for mission-critical and defense speech communication. This system utilizes a highly optimized Complex Convolutional Recurrent Network (Complex-CRN) combined with an LMS Adaptive Filter for real-time edge inference at ~1ms per frame.

## Features
- **Complex-Domain AI Architecture:** A lightweight CNN encoder/decoder with a GRU bottleneck that processes both real and imaginary spectrogram components, preserving critical phase information.
- **Hybrid AI + LMS Pipeline:** The neural network's output is piped through a Least Mean Squares (LMS) adaptive filter to eliminate stubborn residual humming and stationary noise.
- **Ultra-Low Latency & Edge Optimization:** Compiled to ONNX with INT8 dynamic quantization for hardware-accelerated inference. Includes instructions for TensorRT deployment on NVIDIA Jetson devices.
- **Advanced Data Augmentation:** Automatically simulates room reverberation (RT60) and microphone clipping during dataset generation to improve generalization in dynamic defense scenarios.
- **Live Streaming:** Includes a real-time `sounddevice` engine to process your raw microphone feed and output cleaned audio instantly.

## Directory Structure
- `data/raw/clean_speech/`: Drop your clean voice recordings here.
- `data/raw/noise/`: Drop your background noise files here (e.g., helicopters, gunshots, wind, sirens).
- `data/processed/`: Automatically generated during the build process.
- `build.bat`: The master script. Automatically processes data, trains the model, and exports it to ONNX.
- `live_demo.py`: Run this to test the AI on your live microphone.

## How to Train the AI

To make the AI smarter or teach it to block out specific types of noise:
1. Record or download new audio files.
2. Place pure voice recordings into `data/raw/clean_speech`.
3. Place pure noise recordings into `data/raw/noise`.
4. Run `dataset_mixer.py` to generate the augmented dataset.
5. Run `train.py` to train the Complex-CRN model using the multi-objective hybrid loss (SI-SNR + Complex L1 + Spectral Convergence).
6. Run `export_model.py` to generate the INT8 quantized ONNX models.

## How to Use (Live Demo)

Once the model has been trained and `model.onnx` is generated, you can test it live:
```bash
python live_demo.py
```
*Put on your headphones, speak into the microphone, and listen to the real-time AI noise suppression.*

## Requirements
To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
pip install av onnx onnxruntime sounddevice scipy librosa
```
*(PyAV is used to natively decode complex audio files without requiring a system FFmpeg installation).*
