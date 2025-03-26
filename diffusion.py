import torch
from torch import nn
from torch.nn import functional as F
from util import MultiHeadAttentionBlock, CrossAttentionBlock

class TimeEmbedding(nn.Module):
    """
    Time encoding for diffusion models, encode time steps in to the 
    input tensors (batch_size, channels, height, width)
    d_time is the size of the time embedding vector, equal to the number of channels
    """
    def __init__(self, d_time: int, time: int, dropout: float = 0.1, batch_size: int = 1):
        """
        Args:
            d_time: the size of the encoding vector
            dropout: the dropout value
            time: the time step need to be encoded
            batch_size: the batch size
        """
        super().__init__()
        self.d_time = d_time
        self.dropout = nn.Dropout(dropout)

        # denominator is 10000^(2i/d_time)
        self.den = torch.exp((torch.arange(0, d_time, 2) / d_time) * torch.log(10000.0))
        
        self.register_buffer("den", self.den)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: the input tensor of shape (batch_size, channels, height, width)
            time: the time steps need to be encoded, shape (batch_size, 1)
        Returns:
            the input tensor with the time encoding added
        """
        # calculate the time encoding
        time_encoding = torch.zeros(x.shape[0], self.d_time)
        time_encoding[:, 0::2] = torch.sin(time / self.den)
        time_encoding[:, 1::2] = torch.cos(time / self.den)

        time_encoding = time_encoding.unsqueeze(-1).unsqueeze(-1)

        # add the time encoding to the input tensor
        x = x + time_encoding
        return self.dropout(x)
    

class Diffusion(nn.Module):

    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = Unet()
        self.final = Unet_Output(320, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: the latent tensor of shape (batch_size, latent_space_size (4), height / 16, width / 16)
            context: the context tensor of shape (batch_size, seq_len, token_dim)
            times: the time steps need to be encoded, shape (1, 320)
        Returns:
            the output tensor of shape (batch_size, channels, height, width)
        """
        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)
        # (batch_size, 4, height / 8, width / 8) -> (batch_size, 320, height / 8, width / 8)
        x = self.unet(latent, time, context)
        # (batch_size, 320, height / 8, width / 8) -> (batch_size, 4, height / 8, width / 8)
        final = self.final(x)
        return final