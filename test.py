import torch
from torch import nn
from torch.nn import functional as F
from util import ResidualBlock, SinusoidalPositionEmbeddings
from clip import CLIP

class BasicTransformerBlock(nn.Module):
    """
    Combines Self-Attention, Cross-Attention, and FeedForward using nn.MultiheadAttention.
    Operates on inputs of shape (B, C, H, W).
    """
    def __init__(self, dim: int, context_dim: int, n_head: int, dropout: float = 0.1):
        """
        Args:
            dim (int): Input dimension (channels)
            context_dim (int): Dimension of context embeddings
            n_head (int): Number of attention heads
            dropout (float): Dropout rate
        """

        super().__init__()
        self.dim = dim
        # LayerNorms - Applied on the sequence representation (..., N, C)
        self.norm_self_attn = nn.LayerNorm(dim)
        self.norm_cross_attn = nn.LayerNorm(dim)
        self.norm_ff = nn.LayerNorm(dim)

        # Attention Layers using PyTorch's optimized implementation
        # Note: embed_dim is the dimension of the query (image features)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True # Expect input (B, N, C)
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,        # Query dimension (from image features x)
            kdim=context_dim,     # Key dimension (from context)
            vdim=context_dim,     # Value dimension (from context)
            num_heads=n_head,
            dropout=dropout,
            batch_first=True # Expect query(B, N_img, C), key/value(B, N_ctx, C_ctx)
        )

        # FeedForward Layer
        # Input/Output shape: (..., N, C)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) - Image features
        # context: (B, seq_len_ctx, C_context) - Text context embeddings
        batch_size, channels, height, width = x.shape
        n_tokens_img = height * width
        residual = x # Keep original shape for final addition

        # --- Reshape for Sequence Processing ---
        # (B, C, H, W) -> (B, C, N) -> (B, N, C)
        x_seq = x.view(batch_size, channels, n_tokens_img).transpose(1, 2)

        # --- Self-Attention ---
        # Norm applied on (B, N, C)
        x_norm = self.norm_self_attn(x_seq)
        # nn.MultiheadAttention expects Q, K, V. For self-attn, they are the same.
        self_attn_out, _ = self.self_attn(query=x_norm, key=x_norm, value=x_norm, need_weights=False)
        # Add residual connection in sequence format
        x_seq = x_seq + self_attn_out

        # --- Cross-Attention ---
        # Norm applied on (B, N, C)
        x_norm = self.norm_cross_attn(x_seq)
        # Q is from image (x_norm), K and V are from context
        cross_attn_out, _ = self.cross_attn(query=x_norm, key=context, value=context, need_weights=False)
        # Add residual connection in sequence format
        x_seq = x_seq + cross_attn_out

        # --- FeedForward ---
        # Norm applied on (B, N, C)
        x_norm = self.norm_ff(x_seq)
        # FF operates on (B, N, C)
        ff_out = self.ff(x_norm)
        # Add residual connection in sequence format
        x_seq = x_seq + ff_out

        # --- Reshape back to Image Format ---
        # (B, N, C) -> (B, C, N) -> (B, C, H, W)
        out = x_seq.transpose(1, 2).view(batch_size, channels, height, width)

        # The final residual connection should ideally be added *before* reshaping
        # Let's reconsider the residual connections slightly for shape consistency.
        # It's common to apply attention/FF and add residual *within* the sequence format.
        # The code above does this correctly.

        return out # Return shape (B, C, H, W)


# ------------------------------------------
# --- UNet with Attention and Conditioning ---
# ------------------------------------------

class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,              # Input image channels (e.g., 4 if VAE latent)
        out_channels=3,             # Output channels (usually same as input)
        base_dim=64,                # Base channel dimension
        dim_mults=(1, 2, 4, 8),     # Channel multipliers for each resolution level
        num_resnet_blocks=2,        # Number of ResNet blocks per level
        context_dim=768,            # Dimension of CLIP context embeddings
        attn_heads=8,               # Number of attention heads
        dropout=0.1
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.context_dim = context_dim
        self.num_resnet_blocks = num_resnet_blocks

        # --- Time Embedding ---
        time_proj_dim = base_dim * 4 # Dimension to project sinusoidal embedding to
        self.time_embeddings = SinusoidalPositionEmbeddings(base_dim) # Initial embedding dim = base_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(base_dim, time_proj_dim),
            nn.SiLU(),
            nn.Linear(time_proj_dim, time_proj_dim) # This will be used by ResNet blocks
        )
        actual_time_emb_dim = time_proj_dim # This is what ResNetBlock expects

        # --- Initial Convolution ---
        self.init_conv = nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1)

        # --- UNet Layers ---
        dims = [base_dim] + [base_dim * m for m in dim_mults] # e.g., [64, 64, 128, 256, 512]
        in_out_dims = list(zip(dims[:-1], dims[1:])) # e.g., [(64, 64), (64, 128), (128, 256), (256, 512)]
        num_resolutions = len(in_out_dims)

        # Helper modules
        def make_attn_block(dim, heads, ctx_dim):
            # return MultiHeadAttentionBlock(channels=dim, n_head=heads) # Only self-attention
            return BasicTransformerBlock(dim=dim, context_dim=ctx_dim, n_head=heads)

        def make_resnet_block(in_c, out_c, t_emb_dim):
            return ResidualBlock(in_channels=in_c, out_channels=out_c, time_emb_dim=t_emb_dim, dropout=dropout)

        def make_downsample():
            # Use Conv2d with stride 2 to downsample, mean resize image (not channels)
            return nn.Conv2d(dims[i+1], dims[i+1], kernel_size=4, stride=2, padding=1)

        def make_upsample(in_channels, out_channels):
             # Use ConvTranspose2d or Upsample + Conv
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )


        # -- Encoder --
        self.downs = nn.ModuleList([])
        for i, (dim_in, dim_out) in enumerate(in_out_dims):
            is_last = (i == num_resolutions - 1)
            stage_modules = nn.ModuleList([])
            # Add ResNet blocks
            stage_modules.append(make_resnet_block(dim_in, dim_out, actual_time_emb_dim))
            for _ in range(num_resnet_blocks - 1):
                stage_modules.append(make_resnet_block(dim_out, dim_out, actual_time_emb_dim))

            # Add Attention blocks (e.g., at lower resolutions)
            if dim_out in [dims[-1], dims[-2]]: # add attention at last 2 resolutions
                 stage_modules.append(make_attn_block(dim_out, attn_heads, context_dim))

            # Add Downsample layer if not the last stage
            if not is_last:
                stage_modules.append(make_downsample())
            else: # Add Identity if last stage (optional, for consistent structure)
                 stage_modules.append(nn.Identity())

            self.downs.append(stage_modules)

        # -- Bottleneck --
        mid_dim = dims[-1]
        self.mid_block1 = make_resnet_block(mid_dim, mid_dim, actual_time_emb_dim)
        self.mid_attn = make_attn_block(mid_dim, attn_heads, context_dim)
        self.mid_block2 = make_resnet_block(mid_dim, mid_dim, actual_time_emb_dim)

        # -- Decoder --
        self.ups = nn.ModuleList([])
        # Reverse dimensions for decoder, e.g., [(256, 512), (128, 256), (64, 128), (64, 64)]
        for i, (dim_out, dim_in) in enumerate(reversed(in_out_dims)): # Careful: dim_in/out are reversed role here
            is_last = (i == num_resolutions - 1)
            stage_modules = nn.ModuleList([])

            # Add ResNet blocks (Input channels = dim_in + skip_channels)
            skip_channels = dim_in # Channels from corresponding encoder stage
            stage_modules.append(make_resnet_block(dim_in + skip_channels, dim_in, actual_time_emb_dim))
            for _ in range(num_resnet_blocks - 1):
                stage_modules.append(make_resnet_block(dim_in, dim_in, actual_time_emb_dim))

            # Add Attention blocks
            if dim_in in [dims[-1], dims[-2]]: # Match encoder attention placement
                 stage_modules.append(make_attn_block(dim_in, attn_heads, context_dim))

            # Add Upsample layer if not the last stage (output stage)
            if not is_last:
                 stage_modules.append(make_upsample(dim_in, dim_out))
            else:
                 stage_modules.append(nn.Identity())

            self.ups.append(stage_modules)


        # --- Final Layer ---
        self.final_norm = nn.GroupNorm(32, base_dim) # Norm before final conv
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(base_dim, out_channels, kernel_size=1)


    def forward(self, x: torch.Tensor, time: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input noisy tensor (B, C_in, H, W)
            time (torch.Tensor): Time steps (B,)
            context (torch.Tensor): Context embeddings (B, seq_len, C_ctx)

        Returns:
            torch.Tensor: Predicted noise (B, C_out, H, W)
        """
        # 1. Initial Convolution
        x = self.init_conv(x) # (B, base_dim, H, W)

        # 2. Time Embedding
        t_emb = self.time_embeddings(time) # (B, base_dim)
        t_emb = self.time_mlp(t_emb)      # (B, actual_time_emb_dim)

        # 3. Encoder Path
        skip_connections = []
        for i, stage in enumerate(self.downs):
            res_block1 = stage[0] 
            res_blocks_rest = stage[1:self.num_resnet_blocks] # Indexing assumes fixed structure
            attn_block = stage[self.num_resnet_blocks] if len(stage) > self.num_resnet_blocks + 1 else None # Check if attn exists
            downsample = stage[-1]

            x = res_block1(x, t_emb)
            for res_block in res_blocks_rest:
                 x = res_block(x, t_emb)
            if attn_block is not None:
                x = attn_block(x, context)

            skip_connections.append(x) # Store output before downsampling
            x = downsample(x)

        # 4. Bottleneck
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x, context)
        x = self.mid_block2(x, t_emb)

        # 5. Decoder Path
        # Iterate through decoder stages and corresponding skip connections in reverse
        for i, stage in enumerate(self.ups):
            # Get skip connection from corresponding encoder level
            skip = skip_connections.pop() # Pop from the end

            res_block1 = stage[0]
            res_blocks_rest = stage[1:self.num_resnet_blocks]
            attn_block = stage[self.num_resnet_blocks] if len(stage) > self.num_resnet_blocks + 1 else None
            upsample = stage[-1]

            x = torch.cat((skip, x), dim=1) # Concatenate along channel dimension

            x = res_block1(x, t_emb)
            for res_block in res_blocks_rest:
                x = res_block(x, t_emb)
            if attn_block is not None:
                x = attn_block(x, context)
            
            x = upsample(x)


        # 6. Final Layers
        x = self.final_norm(x)
        x = self.final_act(x)
        x = self.final_conv(x)

        return x


# ------------------------------------------
# --- Example Usage ---
# ------------------------------------------
if __name__ == "__main__":
    x = [1, 2, 3, 4]
    # --- Config ---
    img_size = 32 # Example image size (UNet input size, e.g., VAE latent size)
    in_channels = 4 # Example: Latent channels from VAE
    out_channels = 4 # Example: Output latent channels
    batch_size = 2
    clip_seq_len = 77
    time_steps = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # --- Dummy Inputs ---
    dummy_x = torch.randn(batch_size, in_channels, img_size, img_size, device=device)
    dummy_time = torch.randint(0, time_steps, (batch_size,), device=device).long()
    dummy_tokens = torch.randint(0, 49408, (batch_size, clip_seq_len), device=device).long()

    # --- Instantiate Models ---
    print("Instantiating CLIP...")
    clip_model = CLIP().to(device)
    # Set to eval mode if using pretrained weights and not training CLIP
    clip_model.eval()
    print("CLIP instantiated.")

    print("Instantiating UNet...")
    unet_model = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_dim=128,              # Example base dimension
        dim_mults=(1, 2, 3, 4),    # Example multipliers
        num_resnet_blocks=2,
        context_dim=768,           # CLIP output dim
        attn_heads=4,
        dropout=0.1
    ).to(device)
    print("UNet instantiated.")

    # --- Run Inference ---
    print("Running inference...")
    with torch.no_grad(): # Use no_grad if only doing inference
        # 1. Get context from CLIP
        print("Getting context from CLIP...")
        context = clip_model(dummy_tokens)
        print(f"Context shape: {context.shape}")

        # 2. Run UNet
        print("Running UNet forward pass...")
        predicted_noise = unet_model(dummy_x, dummy_time, context)
        print(f"Predicted noise shape: {predicted_noise.shape}")

    # --- Verification ---
    assert predicted_noise.shape == dummy_x.shape, "Output shape must match input shape"
    print("\n>>> UNet with Time Embedding, Self/Cross-Attention, and CLIP Context ran successfully!")

    # Optional: Print model parameter count
    unet_params = sum(p.numel() for p in unet_model.parameters() if p.requires_grad)
    clip_params = sum(p.numel() for p in clip_model.parameters())
    print(f">>> UNet parameters: {unet_params:,}")
    print(f">>> CLIP parameters: {clip_params:,}")