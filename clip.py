import torch
from torch import nn

class CLIPTextEmbedding(nn.Module):
    def __init__(self, n_vocab: int = 49408, n_embed: int = 768, n_tokens: int = 77):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_embed)
        # Learnable position embeddings
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embed))

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        # tokens: (batch_size, n_tokens)
        token_embed = self.token_embedding(tokens) # (batch_size, n_tokens, n_embed)
        # Add position embeddings (broadcasts along batch dimension)
        return token_embed + self.position_embedding

class CLIPLayer(nn.Module):
    def __init__(self, n_head: int = 12, n_embed: int = 768):
        super().__init__()
        # Pre-LayerNorm
        self.norm1 = nn.LayerNorm(n_embed)
        # Attention block (using MultiHeadAttentionBlock requires modification or a different implementation)
        # For compatibility with standard CLIP, let's use nn.MultiheadAttention here
        # self.attention = MultiHeadAttentionBlock(n_embed, n_head) # Your block assumes input (B,C,H,W)
        self.attention = nn.MultiheadAttention(n_embed, n_head, batch_first=True) # Use PyTorch's standard attention for text
        # Pre-LayerNorm
        self.norm2 = nn.LayerNorm(n_embed)
        # Feedforward MLP
        self.mlp = nn.Sequential(
            nn.Linear(n_embed, n_embed * 4),
            nn.GELU(),
            nn.Linear(n_embed * 4, n_embed)
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # x shape: (batch_size, n_tokens, n_embed)
        residual = x
        # Self-Attention part
        x_norm = self.norm1(x)
        # nn.MultiheadAttention expects Q, K, V. For self-attention, they are the same.
        # It also returns attn_output, attn_weights (optional)
        attn_output, _ = self.attention(x_norm, x_norm, x_norm, need_weights=False)
        x = residual + attn_output # Add residual

        # MLP part
        residual = x
        x_norm = self.norm2(x)
        mlp_output = self.mlp(x_norm)
        x = residual + mlp_output # Add residual

        return x

class CLIP(nn.Module):
    def __init__(self, n_vocab: int = 49408, n_embed: int = 768, n_tokens: int = 77, n_layers: int = 12, n_head: int = 12):
        super().__init__()
        self.embedding = CLIPTextEmbedding(n_vocab, n_embed, n_tokens)

        self.layers = nn.ModuleList([
            CLIPLayer(n_head, n_embed) for _ in range(n_layers)
        ])

        # Final LayerNorm
        self.norm = nn.LayerNorm(n_embed)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        # Ensure tokens are long type
        tokens = tokens.type(torch.long)

        # (batch_size, n_tokens) -> (batch_size, n_tokens, n_embed)
        x = self.embedding(tokens)

        # Pass through CLIP layers
        for layer in self.layers:
            x = layer(x)

        # Apply final LayerNorm
        # Output shape: (batch_size, n_tokens, n_embed)
        output = self.norm(x)

        return output
