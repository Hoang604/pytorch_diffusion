import torch
from torch import nn
from torch.nn import functional as F
from vae import ResidualBlock


class TimeEncoding(nn.Module):
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