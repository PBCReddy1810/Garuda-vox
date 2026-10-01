import torch
import torch.optim as optim
import os

from dataset_loader import get_dataloader
from model_architecture import SimpleCRN
from audio_utils import AudioPreprocessor
from losses import hybrid_loss

def main():
    print("Initializing Phase 3: AI Training Loop (Complex Domain)...")
    
    # Config
    batch_size = 8
    epochs = 15  # Fast recovery training run
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup model, processor, and optimizer
    preprocessor = AudioPreprocessor()
    model = SimpleCRN().to(device)
    
    if os.path.exists("best_model.pth"):
        try:
            model.load_state_dict(torch.load("best_model.pth", map_location=device))
            print("Loaded existing best_model.pth. Resuming training...")
        except Exception as e:
            print(f"Could not load previous model, starting fresh: {e}")
            
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Dataloader
    clean_path = "data/processed/train/clean"
    noisy_path = "data/processed/train/noisy"
    
    if not os.path.exists(clean_path):
        print("Data not found. Please complete Phase 1 first.")
        return
        
    dataloader = get_dataloader(clean_path, noisy_path, batch_size=batch_size, segment_length=32000)
    
    print("Starting dummy training loop...")
    model.train()
    
    for epoch in range(epochs):
        for batch_idx, (noisy_audio, clean_audio) in enumerate(dataloader):
            noisy_audio, clean_audio = noisy_audio.to(device), clean_audio.to(device)
            
            optimizer.zero_grad()
            
            # STFT Preprocessing
            mag_noisy, phase_noisy = preprocessor.transform(noisy_audio)
            
            # Calculate Real and Imaginary parts
            real_noisy = mag_noisy * torch.cos(phase_noisy)
            imag_noisy = mag_noisy * torch.sin(phase_noisy)
            
            # Forward pass (model takes real and imag, outputs complex mask)
            # Stack to shape (batch, 2, freq, time)
            complex_noisy = torch.stack([real_noisy, imag_noisy], dim=1)
            
            mask = model(complex_noisy)
            mask_real = mask[:, 0, :, :]
            mask_imag = mask[:, 1, :, :]
            
            # Apply Complex Ratio Mask (CRM)
            est_real = real_noisy * mask_real - imag_noisy * mask_imag
            est_imag = real_noisy * mask_imag + imag_noisy * mask_real
            
            # Convert back to mag and phase for iSTFT
            est_mag = torch.sqrt(est_real**2 + est_imag**2 + 1e-8)
            est_phase = torch.atan2(est_imag, est_real)
            
            # iSTFT to get audio waveform back
            estimated_audio = preprocessor.inverse_transform(est_mag, est_phase, length=noisy_audio.shape[-1])
            
            # Calculate target complex parts
            with torch.no_grad():
                clean_mag, clean_phase = preprocessor.transform(clean_audio)
                target_real = clean_mag * torch.cos(clean_phase)
                target_imag = clean_mag * torch.sin(clean_phase)
            
            # Hybrid Loss calculation
            loss = hybrid_loss(
                estimated_audio, clean_audio, 
                est_real, est_imag, 
                target_real, target_imag,
                alpha=0.1, beta=1.0
            )
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Skipping batch {batch_idx+1} due to unstable math.")
                optimizer.zero_grad()
                continue
            
            # Backpropagation with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Hybrid Loss: {loss.item():.4f}")
            
        # Save model at the end of each epoch
        torch.save(model.state_dict(), "best_model.pth")
        print(f"Epoch {epoch+1} complete. Model saved to best_model.pth\n")
            
    print("Training finished! You can now run evaluate.py or export_model.py")

if __name__ == "__main__":
    main()
