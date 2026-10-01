import torch
import os
import onnx
import onnxruntime as ort
import time
import numpy as np
import warnings
import sys
from onnxruntime.quantization import quantize_dynamic, QuantType

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore")

from model_architecture import SimpleCRN

def export_to_onnx(model_path="best_model.pth", output_path="model.onnx"):
    print("Exporting model to ONNX for edge optimization...")
    device = torch.device("cpu") # Export from CPU is standard
    model = SimpleCRN().to(device)
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            print(f"Loaded trained weights from {model_path}")
        except Exception as e:
            print(f"Failed to load {model_path} (likely an architecture change). Exporting untrained model: {e}")
    else:
        print(f"Warning: {model_path} not found. Exporting untrained model.")
    model.eval()
    
    # Dummy input matching the shape of a single STFT frame.
    # 2 channels (Real, Imaginary), 257 freq bins, 3 time frames.
    dummy_input = torch.randn(1, 2, 257, 3, device=device)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input_complex'],
        output_names=['output_mask_complex']
    )
    print(f"Model successfully exported to {output_path}")
    
    # Apply Dynamic Quantization for Edge Devices
    quant_output_path = output_path.replace(".onnx", "_quant.onnx")
    print(f"Applying INT8 Dynamic Quantization...")
    quantize_dynamic(
        model_input=output_path,
        model_output=quant_output_path,
        weight_type=QuantType.QUInt8
    )
    print(f"Quantized model saved to {quant_output_path}")

    # Instructions for TensorRT (NVIDIA Jetson)
    print("\n--- TensorRT Deployment Instructions ---")
    print("To compile this model for NVIDIA Jetson AGX Orin, run the following command on the target device:")
    print(f"  trtexec --onnx={output_path} --saveEngine=model.trt --fp16")
    print("----------------------------------------\n")

def benchmark_onnx(onnx_path="model.onnx"):
    print(f"Benchmarking ONNX model: {onnx_path}...")
    # Initialize ONNX Runtime session
    ort_session = ort.InferenceSession(onnx_path)
    
    # Dummy input representing one STFT frame (2 channels)
    dummy_input = np.random.randn(1, 2, 257, 3).astype(np.float32)
    
    # Warmup the execution engine
    for _ in range(10):
        ort_session.run(None, {'input_complex': dummy_input})
        
    # Benchmark
    times = []
    for _ in range(100):
        t0 = time.time()
        ort_session.run(None, {'input_complex': dummy_input})
        t1 = time.time()
        times.append((t1 - t0) * 1000)
        
    avg_time = np.mean(times)
    print(f"Average ONNX inference time per frame: {avg_time:.2f} ms")
    
    # Assuming hop length is 256 at 16kHz -> 16 milliseconds
    budget_ms = 16.0
    print(f"Time Budget per frame: {budget_ms:.2f} ms")
    
    if avg_time < budget_ms:
        print("Real-time capability VERIFIED. Model is fast enough for edge streaming.")
    else:
        print("WARNING: Inference is slower than real-time budget.")

if __name__ == "__main__":
    export_to_onnx()
    benchmark_onnx(onnx_path="model.onnx")
    benchmark_onnx(onnx_path="model_quant.onnx")
