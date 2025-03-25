import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, dropout: float = 0.1):
        """
        Residual Block with configurable stride for downsampling.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel (default is 3).
            stride (int): Stride for the first convolutional layer. Allows for downsampling (default is 1).
            dropout (float): Dropout rate to apply after convolutions (default is 0.1).
        """
        super().__init__()

        # First normalization and convolution with stride for downsampling if needed.
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2)

        # Second normalization and convolution.
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2)

        self.dropout = nn.Dropout(dropout)

        # Adjust the residual connection if dimensions do not match or if downsampling is applied.
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.GroupNorm(32, out_channels)
        ) if in_channels != out_channels or stride != 1 else nn.Identity() 

    def forward(self, x):
        """
        Forward pass for the ResidualBlock.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, new_height, new_width)
                          where new_height and new_width depend on the stride.
        """
        # Save the original input for the residual connection.
        residue = x
        
        # First part: normalize, apply activation and convolution.
        x = F.silu(self.norm1(x))
        x = self.conv1(x)

        # Second part: normalize, apply activation and convolution.
        x = F.silu(self.norm2(x))
        x = self.conv2(x)

        # Apply dropout.
        x = self.dropout(x)

        # Add the residual connection and apply a final activation.
        x = F.silu(x + self.residual(residue))
        return x
    
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, channels: int, head: int = 4, dropout: float = 0.1):
        """
        Args:
            channels: the number of channels in the input tensor
            head: the number of heads
            dropout: the dropout value
        """
        super().__init__()
        self.head = head
        self.channels = channels

        self.head_dim = channels // head
        assert self.head_dim * head == channels, "channels must be divisible by head"

        # self.w_q = nn.Linear(channels, channels)    # W_q
        # self.w_k = nn.Linear(channels, channels)    # W_k
        # self.w_v = nn.Linear(channels, channels)    # W_v
        self.w_qkv = nn.Linear(channels, 3 * channels)    # W_q, W_k, W_v
        self.dropout = nn.Dropout(dropout)
        self.w_o = nn.Linear(channels, channels)    # W_o

    @staticmethod
    def scaled_dot_product_attention(head_dim, q, k, v, dropout: nn.Dropout):
        # (batch_size, h, pic_size^2, head_dim) x (batch_size, h, head_dim, pic_size^2) -> (batch_size, h, pic_size^2, pic_size^2)
        # Có vẻ vị trí của Q và K trong bảng attention đổi chỗ cho nhau so với video của 3blue1brown
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn / (head_dim ** 0.5)
        attn = torch.nn.functional.softmax(attn, dim=-1)  # softmax along the last dimension
        if dropout is not None:
            attn = dropout(attn)
        # (batch_size, h, pic_size^2, pic_size^2) x (batch_size, h, pic_size^2, head_dim) -> (batch_size, h, pic_size^2, head_dim)
        return torch.matmul(attn, v), attn

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: the input tensor of shape (batch_size, channels, height, width)
        Returns:
            the output tensor of shape (batch_size, channels, height, width)
        """
        # height and width are the same = pic_size
        batch_size, channels, pic_size, _ = x.size()

        # reshape the input tensor to (batch_size, pic_size^2, channels)
        x_reshape = x.view(batch_size, channels, pic_size ** 2).transpose(1, 2)  # (batch_size, pic_size^2, channels)
        # linear transformation
        qkv = self.w_qkv(x_reshape)  # batch_size, pic_size^2, 3 * channels
        q, k, v = qkv.chunk(3, dim=-1)  # batch_size, pic_size^2, channels

        # split the channels into h heads
        # (batch_size, pic_size^2, channels) -> (batch_size, pic_size^2, h, head_dim) -> (batch_size, h, pic_size^2, head_dim)
        q = q.contiguous().view(batch_size, pic_size ** 2, self.head, self.head_dim).transpose(1, 2)  # (batch_size, h, pic_size^2, head_dim)
        k = k.contiguous().view(batch_size, pic_size ** 2, self.head, self.head_dim).transpose(1, 2)  # (batch_size, h, pic_size^2, head_dim)
        v = v.contiguous().view(batch_size, pic_size ** 2, self.head, self.head_dim).transpose(1, 2)  # (batch_size, h, pic_size^2, head_dim)

        # scaled dot-product attention
        x, attn = self.scaled_dot_product_attention(self.head_dim, q, k, v, self.dropout)

        # (batch_size, h, pic_size^2, head_dim) -> (batch_size, pic_size^2, h, head_dim)
        x = x.transpose(1, 2)
        # (batch_size, pic_size^2, h, head_dim) -> (batch_size, pic_size^2, channels)
        # head concatenation
        x = x.contiguous().view(batch_size, pic_size ** 2, channels)
        # transpose the tensor back to the original shape, split pic_size^2 into pic_size x pic_size
        x = x.view(batch_size, pic_size, pic_size, channels)

        # Linear transformation in last dimension (channels)
        # (batch_size, pic_size, pic_size, channels) -> (batch_size, pic_size, pic_size, channels)
        x = self.w_o(x)

        # (batch_size, pic_size, pic_size, channels) -> (batch_size, channels, pic_size, pic_size)
        x = x.contiguous().view(batch_size, pic_size, pic_size, channels).permute(0, 3, 1, 2)

        return x

class VAEEncoder(nn.Module):

    def __init__(self, latent_dim: int, in_channels: int=3, num_resnet_blocks: int=3, num_attention_heads: int=4):
        super().__init__()

        self.resnet_block_in = self._make_layer(ResidualBlock, in_channels, 64, num_resnet_blocks, stride=2)

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
        # (batch_size, in_channels, height, width) -> (batch_size, 64, height / 2, width / 2)
        x = self.resnet_block_in(x)
        # (batch_size, 64, height / 2, width / 2) -> (batch_size, 512, height / 4, width / 4)
        x = self.resnet_block_1(x)
        # (batch_size, 512, height / 4, width / 4) -> (batch_size, 512, height / 8, width / 8)
        x = self.resnet_block_2(x)
        # (batch_size, 512, height / 8, width / 8) -> (batch_size, 512, height / 16, width / 16)
        x = self.resnet_block_3(x)

        # (batch_size, 512, height / 16, width / 16) -> (batch_size, 512, height / 16, width / 16)
        x = self.attention1(x)

        # (batch_size, 512, height / 16, width / 16) -> (batch_size, 2 * latent_dim, height / 16, width / 16)
        x = self.conv_out(x)
        # The output of the last conv layer contains both mean and log variance
        mean, log_var = torch.chunk(x, 2, dim=1)

        return mean, log_var
    
class VAEDecoder(nn.Module):
    def __init__(self, latent_dim: int, out_channels: int = 3, num_resnet_block=3, num_attention_heads=4):
        super().__init__()
        self.conv_in = nn.Conv2d(latent_dim, 512, kernel_size=3, padding=1)
        self.attention = MultiHeadAttentionBlock(512, num_attention_heads)

        # 4 upsample blocks, suppose the input size is 256x256
        self.upsample1 = self._make_upsample_block(512)  # 16x16 → 32x32
        self.resnet_block_1 = self._make_layer(ResidualBlock, 512, 256, num_resnet_block)

        self.upsample2 = self._make_upsample_block(256)  # 32x32 → 64x64
        self.resnet_block_2 = self._make_layer(ResidualBlock, 256, 128, num_resnet_block)

        self.upsample3 = self._make_upsample_block(128)  # 64x64 → 128x128
        self.resnet_block_3 = self._make_layer(ResidualBlock, 128, 64, num_resnet_block)

        self.upsample4 = self._make_upsample_block(64)  # 128x128 → 256x256
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

        # (batch_size, latent_dim, h/16, w/16) -> (batch_size, 512, h/16, w/16)
        x = self.conv_in(z)
        x = self.attention(x)  # [batch, 512, h/16, w/16]

        # --- Block 1: 16x16 -> 32x32 ---
        x = self.upsample1(x)  # [batch, 512, h/8, w/8] (PixelShuffle keep number of channels)
        x = self.resnet_block_1(x)  # [batch, 256, h/8, w/8] (ResNet reduce channels from 512->256)

        # --- Block 2: 32x32 -> 64x64 ---
        x = self.upsample2(x)  # [batch, 256, h/4, w/4]
        x = self.resnet_block_2(x)  # [batch, 128, h/4, w/4]

        # --- Block 3: 64x64 -> 128x128 ---
        x = self.upsample3(x)  # [batch, 128, h/2, w/2]
        x = self.resnet_block_3(x)  # [batch, 64, h/2, w/2]

        # --- Block 4: 128x128 -> 256x256 ---
        x = self.upsample4(x)  # [batch, 64, h, w]
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

    def reparameterize(self, mean, var):
        """Sample from the latent distribution using the reparameterization trick"""
        std = torch.sqrt(var)
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
        latent_h, latent_w = height // 16, width // 16
        z = torch.randn(num_samples, self.latent_dim, latent_h, latent_w, device=device)
        samples = self.decode(z)
        return samples