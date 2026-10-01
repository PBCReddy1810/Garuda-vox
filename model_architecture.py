import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU()
        
    def forward(self, x):
        return self.elu(self.bn(self.conv(x)))

class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding=0):
        super().__init__()
        self.conv_t = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU()
        
    def forward(self, x):
        return self.elu(self.bn(self.conv_t(x)))

class SimpleCRN(nn.Module):
    """
    Highly Optimized Complex Convolutional Recurrent Network for Edge Devices.
    Uses GRU and halved channel counts to drastically lower latency.
    Accepts 2 channels (Real, Imaginary) and outputs 2 channels (Real Mask, Imag Mask).
    """
    def __init__(self):
        super().__init__()
        # Encoder (downsamples frequency dimension, preserves time)
        # Input channels: 2 (Real, Imag)
        self.enc1 = ConvBlock(2, 8, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))
        self.enc2 = ConvBlock(8, 16, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))
        self.enc3 = ConvBlock(16, 32, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))
        
        # Fast GRU (Recurrent bottleneck)
        # Expected freq bins at this point: 33 (if n_fft=512 -> 257 -> 129 -> 65 -> 33)
        self.gru = nn.GRU(input_size=32 * 33, hidden_size=128, num_layers=1, batch_first=True)
        self.gru_proj = nn.Linear(128, 32 * 33)
        
        # Decoder (upsamples frequency dimension using skip connections)
        self.dec3 = ConvTransposeBlock(64, 16, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), output_padding=(0, 0))
        self.dec2 = ConvTransposeBlock(32, 8, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), output_padding=(0, 0))
        # Output channels: 2 (Complex Mask: Real, Imag)
        self.dec1 = nn.ConvTranspose2d(16, 2, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), output_padding=(0, 0))

    def forward(self, x, h_in=None):
        # x shape: (batch, channels=2, freq, time)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # Prepare for GRU
        b, c, f, t = e3.shape
        # Permute to (batch, time, channels, freq) then flatten to (batch, time, features)
        gru_in = e3.permute(0, 3, 1, 2).reshape(b, t, -1)
        
        if h_in is not None:
            gru_out, h_out = self.gru(gru_in, h_in)
        else:
            gru_out, h_out = self.gru(gru_in)
            
        gru_out = self.gru_proj(gru_out)
        
        # Reshape back to CNN format
        gru_out = gru_out.reshape(b, t, c, f).permute(0, 2, 3, 1)
        
        # Skip connections
        d3 = self.dec3(torch.cat([gru_out, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        
        # Output complex mask (no sigmoid, allowing negative values for CRM)
        mask = d1
        
        if h_in is not None:
            return mask, h_out
        return mask

if __name__ == "__main__":
    print("Testing Complex-CRN Model Architecture...")
    # Dummy spectrogram input: batch_size=2, channel=2 (Real, Imag), freq_bins=257 (for n_fft=512), time_frames=126
    x = torch.randn(2, 2, 257, 126)
    model = SimpleCRN()
    mask = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Mask output shape: {mask.shape}")
    assert x.shape == mask.shape, "Mask shape must match input shape"
    print("Model Architecture test PASSED.")
