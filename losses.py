import torch
import torch.nn.functional as F

def si_snr_loss(estimated, target, eps=1e-8):
    """
    Scale-Invariant Signal-to-Noise Ratio (SI-SNR) loss.
    Lower is better (we return negative SI-SNR).
    estimated: (batch, time)
    target: (batch, time)
    """
    # Remove DC offset
    target = target - torch.mean(target, dim=-1, keepdim=True)
    estimated = estimated - torch.mean(estimated, dim=-1, keepdim=True)
    
    # Scale target to estimated (Scale invariance)
    target_energy = torch.sum(target ** 2, dim=-1, keepdim=True)
    dot_product = torch.sum(target * estimated, dim=-1, keepdim=True)
    
    scaling_factor = dot_product / (target_energy + eps)
    
    # Optimal scaled target
    target_scaled = scaling_factor * target
    
    # Noise component (anything in estimated that is NOT the target)
    noise = estimated - target_scaled
    
    # SNR calculation (in decibels)
    snr = 10 * torch.log10(
        torch.sum(target_scaled ** 2, dim=-1, keepdim=True) / 
        (torch.sum(noise ** 2, dim=-1, keepdim=True) + eps) + eps
    )
    
    # Return negative SNR as loss (minimizing negative SNR maximizes SNR)
    return -torch.mean(snr)

def complex_l1_loss(est_real, est_imag, target_real, target_imag):
    """L1 loss on complex spectrograms."""
    loss_real = F.l1_loss(est_real, target_real)
    loss_imag = F.l1_loss(est_imag, target_imag)
    return loss_real + loss_imag

def spectral_convergence_loss(est_mag, target_mag, eps=1e-8):
    """
    Spectral convergence loss acts as a simple perceptual loss,
    penalizing the difference in spectral envelope.
    """
    num = torch.norm(target_mag - est_mag, p="fro")
    den = torch.norm(target_mag, p="fro") + eps
    return num / den

def hybrid_loss(est_audio, target_audio, est_real, est_imag, target_real, target_imag, alpha=0.1, beta=1.0, gamma=1.0):
    """
    Combined loss function: SI-SNR + alpha * Complex L1 + beta * Spectral Convergence
    """
    loss_si_snr = si_snr_loss(est_audio, target_audio)
    loss_complex = complex_l1_loss(est_real, est_imag, target_real, target_imag)
    
    est_mag = torch.sqrt(est_real**2 + est_imag**2 + 1e-8)
    target_mag = torch.sqrt(target_real**2 + target_imag**2 + 1e-8)
    loss_sc = spectral_convergence_loss(est_mag, target_mag)
    
    return loss_si_snr + alpha * loss_complex + beta * loss_sc

if __name__ == "__main__":
    print("Testing Loss Functions...")
    target = torch.randn(2, 32000)
    estimated = target + 0.1 * torch.randn(2, 32000)
    
    target_real = torch.randn(2, 257, 126)
    target_imag = torch.randn(2, 257, 126)
    est_real = target_real + 0.1 * torch.randn(2, 257, 126)
    est_imag = target_imag + 0.1 * torch.randn(2, 257, 126)
    
    loss = hybrid_loss(estimated, target, est_real, est_imag, target_real, target_imag)
    print(f"Hybrid Loss: {loss.item():.4f}")
    print("Loss test PASSED.")
