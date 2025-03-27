import torch
from torch import nn
from torch.nn import functional as F
from util import ResidualBlock, SinusoidalPositionEmbeddings
from clip import CLIP
from typing import Optional
from torchinfo import summary

class BasicTransformerBlock(nn.Module):
    """
    Combines Self-Attention, Cross-Attention, and FeedForward using nn.MultiheadAttention.
    Operates on inputs of shape (B, C, H, W).
    Cross-Attention is applied conditionally based on context availability.
    """
    def __init__(self, dim: int, context_dim: int, n_head: int, dropout: float = 0.1):
        """
        Args:
            dim (int): Input dimension (channels)
            context_dim (int): Dimension of context embeddings (only used if context is provided)
            n_head (int): Number of attention heads
            dropout (float): Dropout rate
        """
        super().__init__()
        self.dim = dim
        # LayerNorms
        self.norm_self_attn = nn.LayerNorm(dim)
        self.norm_cross_attn = nn.LayerNorm(dim) # Norm before cross-attention
        self.norm_ff = nn.LayerNorm(dim)

        # Self-Attention Layer
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True # Expect input (B, N, C)
        )

        # Cross-Attention Layer (will be used conditionally)
        # We define it here, but only use it in forward if context is not None
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,        # Query dimension (from image features x)
            kdim=context_dim,     # Key dimension (from context)
            vdim=context_dim,     # Value dimension (from context)
            num_heads=n_head,
            dropout=dropout,
            batch_first=True # Expect query(B, N_img, C), key/value(B, N_ctx, C_ctx)
        )

        # FeedForward Layer
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, C, H, W) - Image features
        # context: Optional[(B, seq_len_ctx, C_context)] - Text context embeddings or None
        batch_size, channels, height, width = x.shape
        n_tokens_img = height * width
        # Note: No residual = x here, residuals are added after each block

        # --- Reshape for Sequence Processing ---
        # (B, C, H, W) -> (B, C, N) -> (B, N, C)
        x_seq = x.view(batch_size, channels, n_tokens_img).transpose(1, 2)

        # --- Self-Attention ---
        x_norm = self.norm_self_attn(x_seq)
        self_attn_out, _ = self.self_attn(query=x_norm, key=x_norm, value=x_norm, need_weights=False)
        x_seq = x_seq + self_attn_out # Add residual

        # --- Cross-Attention (Conditional) ---
        # Only perform cross-attention if context is provided
        if context is not None:
            x_norm = self.norm_cross_attn(x_seq)
            cross_attn_out, _ = self.cross_attn(query=x_norm, key=context, value=context, need_weights=False)
            x_seq = x_seq + cross_attn_out # Add residual only if cross-attn was performed
        # If context is None, this block is skipped

        # --- FeedForward ---
        x_norm = self.norm_ff(x_seq)
        ff_out = self.ff(x_norm)
        x_seq = x_seq + ff_out # Add residual

        # --- Reshape back to Image Format ---
        # (B, N, C) -> (B, C, N) -> (B, C, H, W)
        out = x_seq.transpose(1, 2).view(batch_size, channels, height, width)

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
        attn_heads=4,               # Number of attention heads
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
            if dim_out in [dims[-2], dims[-3]]: # add attention at last 2 resolutions
                 stage_modules.append(make_attn_block(dim_out, attn_heads, context_dim))

            if dim_out == dims[-1]: # Add more attention at the highest resolution
                stage_modules.append(make_attn_block(dim_out, attn_heads * 2, context_dim))
                stage_modules.append(make_attn_block(dim_out, attn_heads * 2, context_dim))

            # Add Downsample layer if not the last stage
            if not is_last:
                stage_modules.append(make_downsample())
            else: # Add Identity if last stage (optional, for consistent structure)
                 stage_modules.append(nn.Identity())

            self.downs.append(stage_modules)

        # -- Bottleneck --
        mid_dim = dims[-1]
        self.mid_block1 = make_resnet_block(mid_dim, mid_dim, actual_time_emb_dim)
        self.mid_attn1 = make_attn_block(mid_dim, attn_heads * 2, context_dim)
        self.mid_attn2 = make_attn_block(mid_dim, attn_heads * 2, context_dim)
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
            if dim_in in [dims[-2], dims[-3]]: # Match encoder attention placement
                 stage_modules.append(make_attn_block(dim_in, attn_heads, context_dim))
            
            if dim_in == dims[-1]:  # Add more attention at the highest resolution
                stage_modules.append(make_attn_block(dim_in, attn_heads * 2, context_dim))
                stage_modules.append(make_attn_block(dim_in, attn_heads * 2, context_dim))

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


    def forward(self, x: torch.Tensor, time: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
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
            attn_block1 = stage[self.num_resnet_blocks] if len(stage) > self.num_resnet_blocks + 1 else None # Check if attn exists
            attn_block2 = stage[self.num_resnet_blocks + 1] if len(stage) > self.num_resnet_blocks + 2 else None
            downsample = stage[-1]

            x = res_block1(x, t_emb)
            for res_block in res_blocks_rest:
                 x = res_block(x, t_emb)
            if attn_block1 is not None:
                x = attn_block1(x, context)
            if attn_block2 is not None:
                x = attn_block2(x, context)

            skip_connections.append(x) # Store output before downsampling
            x = downsample(x)

        # 4. Bottleneck
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn1(x, context)
        x = self.mid_attn2(x, context)
        x = self.mid_block2(x, t_emb)

        # 5. Decoder Path
        # Iterate through decoder stages and corresponding skip connections in reverse
        for i, stage in enumerate(self.ups):
            # Get skip connection from corresponding encoder level
            skip = skip_connections.pop() # Pop from the end

            res_block1 = stage[0]
            res_blocks_rest = stage[1:self.num_resnet_blocks]
            attn_block = stage[self.num_resnet_blocks] if len(stage) > self.num_resnet_blocks + 1 else None
            attn_block2 = stage[self.num_resnet_blocks + 1] if len(stage) > self.num_resnet_blocks + 2 else None
            upsample = stage[-1]

            x = torch.cat((skip, x), dim=1) # Concatenate along channel dimension

            x = res_block1(x, t_emb)
            for res_block in res_blocks_rest:
                x = res_block(x, t_emb)
            if attn_block is not None:
                x = attn_block(x, context)
            if attn_block2 is not None:
                x = attn_block2(x, context)
            
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
    dummy_x_shape = dummy_x.shape
    dummy_time_shape = dummy_time.shape
    dummy_tokens_shape = dummy_tokens.shape
    print(f"Input shape: {dummy_x_shape}")
    print(f"Time shape: {dummy_time_shape}")
    print(f"Tokens shape: {dummy_tokens_shape}")


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
    # Chỉ truyền input chính (hình ảnh) cho torchsummary
    print("Model Summary:")
    print(unet_model)    
    # Thêm vào đoạn code trước khi chạy forward pass
    summary(
        unet_model, 
        input_data=[
            dummy_x,  # hình ảnh
            dummy_time,  # timestep
            clip_model(dummy_tokens)  # context từ CLIP
        ],
        depth=3  # độ sâu hiển thị layers
    )

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