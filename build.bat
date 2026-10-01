@echo off
set PYTHONIOENCODING=utf-8
echo ==========================================
echo   Building AI Noise Suppression System
echo ==========================================

echo [1/3] Generating synthetic dataset...
python dataset_mixer.py

echo [2/3] Training the AI model (This will take a while)...
python train.py

echo [3/3] Exporting trained model to ONNX...
python export_model.py

echo ==========================================
echo   Build Complete! You can now run:
echo   python live_demo.py
echo ==========================================
