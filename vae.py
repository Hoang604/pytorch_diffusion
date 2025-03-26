import torch
import torch.nn as nn
from util import ResidualBlock, MultiHeadAttentionBlock


class VAEEncoder(nn.Module):

    def __init__(self, latent_dim: int = 4, in_channels: int=3, num_resnet_blocks: int=3, num_attention_heads: int=4):
        super().__init__()

        self.conv_in = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)

        self.resnet_block_in = self._make_layer(ResidualBlock, 64, 64, num_resnet_blocks, stride=1)
        self.resnet_block_1 = self._make_layer(ResidualBlock, 64, 128, num_resnet_blocks, stride=2)
        self.resnet_block_2 = self._make_layer(ResidualBlock, 128, 256, num_resnet_blocks, stride=2)
        self.resnet_block_3 = self._make_layer(ResidualBlock, 256, 512, num_resnet_blocks, stride=2)

        self.attention1 = MultiHeadAttentionBlock(512, num_attention_heads)

        self.conv_out = nn.Conv2d(512, 2 * latent_dim, kernel_size=3, padding=1)

    def _make_layer(self, block, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(block(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: the input tensor of shape (batch_size, in_channels, height, width)
        Returns:
            the mean and log variance of the latent space
        """
        # (batch_size, in_channels, height, width) -> (batch_size, 64, height, width)
        x = self.conv_in(x)
        # (batch_size, 64, height, width) -> (batch_size, 64, height, width)
        x = self.resnet_block_in(x)
        # (batch_size, 64, height, width) -> (batch_size, 512, height / 2, width / 2)
        x = self.resnet_block_1(x)
        # (batch_size, 512, height / 2, width / 2) -> (batch_size, 512, height / 4, width / 4))
        x = self.resnet_block_2(x)
        # (batch_size, 512, height / 4, width / 4) -> (batch_size, 512, height / 8, width / 8)
        x = self.resnet_block_3(x)

        # (batch_size, 512, height / 8, width / 8) -> (batch_size, 512, height / 8, width / 8)
        x = self.attention1(x)

        # (batch_size, 512, height / 8, width / 8) -> (batch_size, 2 * latent_dim, height / 8, width / 8)
        x = self.conv_out(x)
        # The output of the last conv layer contains both mean and log variance
        mean, log_var = torch.chunk(x, 2, dim=1)

        log_var = torch.clamp(log_var, -30, 20)  # limit the variance to avoid numerical instability

        return mean, log_var
    
class VAEDecoder(nn.Module):
    def __init__(self, latent_dim: int = 4, out_channels: int = 3, num_resnet_block=3, num_attention_heads=4):
        super().__init__()
        self.conv_in = nn.Conv2d(latent_dim, 512, kernel_size=3, padding=1)
        self.attention = MultiHeadAttentionBlock(512, num_attention_heads)

        # 4 upsample blocks, suppose the input size is 256x256
        self.upsample1 = self._make_upsample_block(512)  # 32x32 → 64x64
        self.resnet_block_1 = self._make_layer(ResidualBlock, 512, 256, num_resnet_block)

        self.upsample2 = self._make_upsample_block(256)  # 64x64 → 128x128
        self.resnet_block_2 = self._make_layer(ResidualBlock, 256, 128, num_resnet_block)

        self.upsample3 = self._make_upsample_block(128)  # 128x128 → 256x256
        self.resnet_block_3 = self._make_layer(ResidualBlock, 128, 64, num_resnet_block)

        self.resnet_block_4 = self._make_layer(ResidualBlock, 64, 64, num_resnet_block - 1)

        self.conv_out = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def _make_upsample_block(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.GroupNorm(32, in_channels),
            nn.SiLU()
        )

    def _make_layer(self, block, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(block(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, z):
        """
        Args:
            z: the input tensor of shape (batch_size, latent_dim, height / 16, width / 16)
            this vector is generated from the mean and log variance of the latent space
        Returns:
            the output tensor of shape (batch_size, out_channels, height, width)
        """

        # (batch_size, latent_dim, h/8, w/8) -> (batch_size, 512, h/8, w/8)
        x = self.conv_in(z)
        x = self.attention(x)  # [batch, 512, h/8, w/8]

        # --- Block 1: 32x32 -> 64x64 ---
        x = self.upsample1(x)  # [batch, 512, h/4, w/4] (PixelShuffle keep number of channels)
        x = self.resnet_block_1(x)  # [batch, 256, h/4, w/4] (ResNet reduce channels from 512->256)

        # --- Block 2: 64x64 -> 128x128 ---
        x = self.upsample2(x)  # [batch, 256, h/2, w/2]
        x = self.resnet_block_2(x)  # [batch, 128, h/2, w/2]

        # --- Block 3: 128x128 -> 256x256 ---
        x = self.upsample3(x)  # [batch, 128, h, w]
        x = self.resnet_block_3(x)  # [batch, 64, h, w]

        # --- Block 4: 256x256 -> 256x256 ---
        x = self.resnet_block_4(x)  # [batch, 64, h, w]

        # Output convolution
        x = self.conv_out(x)  # [batch, 3, h, w]
        return x

class VAE(nn.Module):
    
    def __init__(self, latent_dim: int, in_channels: int = 3, out_channels: int = 3, 
                 num_resnet_blocks: int = 3, num_attention_heads: int = 4):
        super().__init__()
        
        self.encoder = VAEEncoder(latent_dim, in_channels, num_resnet_blocks, num_attention_heads)
        self.decoder = VAEDecoder(latent_dim, out_channels, num_resnet_blocks, num_attention_heads)
        self.latent_dim = latent_dim
        
    def encode(self, x):
        """Encode input to mean and variance of latent distribution"""
        return self.encoder(x)

    def decode(self, z):
        """Decode latent vector to reconstruction"""
        return self.decoder(z)

    def reparameterize(self, mean, var, noise):
        """Sample from the latent distribution using the reparameterization trick"""
        std = torch.sqrt(var)
        if noise:
            eps = noise
        else:
            eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def forward(self, x):
        """
        Forward pass through the VAE
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
        Returns:
            Reconstructed tensor and parameters of the latent distribution (mean, var)
        """
        mean, var = self.encode(x)
        z = self.reparameterize(mean, var)
        x_recon = self.decode(z)
        
        return x_recon, mean, var

    def sample(self, num_samples: int, height: int, width: int, device: torch.device):
        """
        Sample from the latent distribution to generate new images
        Args:
            num_samples: Number of samples to generate
            height, width: Dimensions of the original input image
            device: Device to create tensor on
        """
        latent_h, latent_w = height // 8, width // 8
        z = torch.randn(num_samples, self.latent_dim, latent_h, latent_w, device=device)
        samples = self.decode(z)
        return samples