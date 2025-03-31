# train_vae.py
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter # For logging
import os
import argparse
from tqdm import tqdm # For progress bar
import lpips # Import LPIPS library
from util import SimpleImageDataset
from datetime import datetime
from diffusion import DiffusionModelPyTorch

try:
    from vae import VAE
except ImportError:
    raise ImportError("Error: Ensure VAE class definition is available (e.g., in vae_architecture.py)") # write error message on console

# --- VAE Loss Function ---
def vae_loss_function(recon_x, x, mu, log_var, lpips_loss_fn, kl_weight=1.0, l1_weight=1.0, lpips_weight=0.5):
    """
    Calculates the VAE loss: L1 Reconstruction Loss + LPIPS Loss + KL Divergence Loss.

    Args:
        recon_x: Reconstructed input data (output of decoder). [-1, 1] range.
        x: Original input data. [-1, 1] range.
        mu: Mean of the latent distribution (output of encoder).
        log_var: Log variance of the latent distribution (output of encoder).
        lpips_loss_fn: Instantiated LPIPS loss function.
        kl_weight: Weight factor for the KL divergence term.
        l1_weight: Weight factor for the L1 loss term.
        lpips_weight: Weight factor for the LPIPS loss term.

    Returns:
        Tuple: (Total Loss, L1 Loss, LPIPS Loss, KL Divergence Loss)
    """
    # L1 Reconstruction Loss (Mean Absolute Error)
    l1_loss = F.l1_loss(recon_x, x, reduction='mean') # Average over all elements and batch

    # Perceptual Loss (LPIPS)
    # LPIPS expects input in range [-1, 1], which matches our normalization
    perceptual_loss = lpips_loss_fn(recon_x, x).mean() # LPIPS outputs per-image loss, take mean over batch

    # Combined Reconstruction Loss
    recon_loss_combined = l1_weight * l1_loss + lpips_weight * perceptual_loss

    # KL Divergence Loss
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=[1, 2, 3]) # Sum over C, H, W
    kl_div = torch.mean(kl_div) # Average over batch size

    # Total Loss
    total_loss = recon_loss_combined + kl_weight * kl_div

    # Return individual components for logging
    return total_loss, l1_loss, perceptual_loss, kl_div

# --- Main Training Function ---
def train_vae(args):
    """
    Main function to train the VAE model with gradient accumulation.
    """
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}") # write message on console

    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.log_dir, time)
    checkpoint_dir = os.path.join(args.checkpoint_dir, time)
    # Create directories if they don't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # --- Dataset and DataLoader ---
    print("Loading dataset...") # write message on console
    train_dataset = SimpleImageDataset(folder_path=args.image_folder, img_size=args.img_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size, # This is the micro-batch size for accumulation
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    print(f"Dataset loaded with {len(train_dataset)} samples.") # write message on console
    print(f"Micro-batch size: {args.batch_size}") # write message on console
    print(f"Accumulation steps: {args.accumulation_steps}") # write message on console
    effective_batch_size = args.batch_size * args.accumulation_steps
    print(f"Effective batch size: {effective_batch_size}") # write message on console


    # --- Model ---
    print("Initializing VAE model...") # write message on console
    model = VAE(
        latent_dim=args.latent_dim,
        in_channels=3,
        out_channels=3
        # Add other args like num_resnet_blocks if your VAE class needs them
    ).to(device)
    print("VAE model initialized.") # write message on console

    # --- Optimizer ---
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    # Zero gradients initially before the loop starts
    optimizer.zero_grad()
    print(f"Optimizer: AdamW, LR: {args.learning_rate}") # write message on console

    # --- LPIPS Loss Setup ---
    print("Initializing LPIPS loss...") # write message on console
    lpips_loss_fn = lpips.LPIPS(net='alex').to(device) # Use AlexNet backbone for LPIPS
    # Freeze LPIPS model parameters (usually done by default, but good practice)
    for param in lpips_loss_fn.parameters():
        param.requires_grad = False
    print("LPIPS loss initialized.") # write message on console

    # --- Logging ---
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}") # write message on console

    # --- Training Loop ---
    import time
    start_time = time.time()
    print(f"Starting VAE training for {args.epochs} epochs...") # write message on console

    start_epoch, best_loss = DiffusionModelPyTorch.load_checkpoint_for_resume(device=device, model=model, optimizer=optimizer, checkpoint_path=os.path.join(checkpoint_dir, "vae_best.pth"))
    global_step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, args.epochs):
        model.train() # Set model to training mode
        epoch_loss = 0.0
        epoch_l1_loss = 0.0
        epoch_lpips_loss = 0.0
        epoch_kl_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True)

        # --- Main Training Loop ---
        for batch_idx, images in enumerate(progress_bar):
            images = images.to(device)

            # Forward pass through VAE
            recon_images, mean, log_var = model(images)

            # Calculate loss (using updated loss function)
            # Keep the original loss for logging
            total_loss, l1_loss, perceptual_loss, kl_loss = vae_loss_function(
                recon_images, images, mean, log_var, lpips_loss_fn,
                args.kl_weight, args.l1_weight, args.lpips_weight
            )

            # --- Gradient Accumulation ---
            # 1. Scale the loss
            # Divide the loss by accumulation steps *before* backward pass
            scaled_loss = total_loss / args.accumulation_steps

            # 2. Backward pass (accumulates gradients)
            # Gradients are added to the existing ones (or initialized if zero)
            scaled_loss.backward()

            # 3. Optimizer step (conditional)
            # Check if we have processed enough steps for one effective batch
            # or if it's the last batch of the epoch
            is_accumulation_step = (batch_idx + 1) % args.accumulation_steps == 0
            is_last_batch = (batch_idx + 1) == len(train_loader)

            if is_accumulation_step or is_last_batch:
                optimizer.step()      # Apply accumulated gradients
                optimizer.zero_grad() # Reset gradients for the next accumulation cycle
            # --- End Gradient Accumulation ---


            # --- Logging (use unscaled loss) ---
            epoch_loss += total_loss.item() # Log the original unscaled loss
            epoch_l1_loss += l1_loss.item()
            epoch_lpips_loss += perceptual_loss.item()
            epoch_kl_loss += kl_loss.item()

            # Log instantaneous losses (unscaled) to TensorBoard
            writer.add_scalar('Loss/Total_Step', total_loss.item(), global_step)
            writer.add_scalar('Loss/L1_Step', l1_loss.item(), global_step)
            writer.add_scalar('Loss/LPIPS_Step', perceptual_loss.item(), global_step)
            writer.add_scalar('Loss/KL_Step', kl_loss.item(), global_step)

            # Update progress bar description (with unscaled loss)
            progress_bar.set_postfix({
                "Total Loss": f"{total_loss.item():.4f}",
                "L1": f"{l1_loss.item():.4f}",
                "LPIPS": f"{perceptual_loss.item():.4f}",
                "KL": f"{kl_loss.item():.4f}"
            })

            global_step += 1

        # --- End of Epoch ---
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_l1_loss = epoch_l1_loss / len(train_loader)
        avg_lpips_loss = epoch_lpips_loss / len(train_loader)
        avg_kl_loss = epoch_kl_loss / len(train_loader)

        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Avg Loss: {avg_epoch_loss:.4f}, "
              f"Avg L1: {avg_l1_loss:.4f}, "
              f"Avg LPIPS: {avg_lpips_loss:.4f}, "
              f"Avg KL: {avg_kl_loss:.4f}") # write message on console

        # Log average epoch losses to TensorBoard
        writer.add_scalar('Loss/Total_Epoch', avg_epoch_loss, epoch)
        writer.add_scalar('Loss/L1_Epoch', avg_l1_loss, epoch)
        writer.add_scalar('Loss/LPIPS_Epoch', avg_lpips_loss, epoch)
        writer.add_scalar('Loss/KL_Epoch', avg_kl_loss, epoch)

        # --- Save Best Model ---
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            print(f"New best loss! Saving model checkpoint to {checkpoint_dir}")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, os.path.join(checkpoint_dir, "vae_best.pth"))

        # --- Save Checkpoint ---
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
            checkpoint_path = os.path.join(checkpoint_dir, f"vae_epoch_{epoch+1}.pth")
            # Ensure model state is saved correctly (no DDP wrapper here)
            model_state_to_save = model.state_dict()
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model_state_to_save,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}") # write message on console

    # --- End of Training ---
    end_time = time.time()
    print(f"\nTraining finished in {(end_time - start_time)/60:.2f} minutes.") # write message on console
    writer.close() # Close TensorBoard writer

# --- Script Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Variational Autoencoder (VAE) with L1+LPIPS Loss and Gradient Accumulation")

    # Dataset and Model Args
    parser.add_argument('--image_folder', type=str, default='/media/hoangdv/simpler_data', required=True, help='Path to the folder containing training images')
    parser.add_argument('--img_size', type=int, default=256, help='Target image size (resized to img_size x img_size)')
    parser.add_argument('--latent_dim', type=int, default=4, help='Dimension of the VAE latent space (encoder output channels will be 2*latent_dim)')
    # Add args for VAE architecture if needed

    # Training Args
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Micro-batch size (for gradient accumulation)')
    parser.add_argument('--accumulation_steps', type=int, default=8, help='Number of steps to accumulate gradients over. Effective batch size = batch_size * accumulation_steps') # New Arg
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Optimizer learning rate')
    parser.add_argument('--kl_weight', type=float, default=1.0, help='Weight for the KL divergence term in the VAE loss')
    parser.add_argument('--l1_weight', type=float, default=1.0, help='Weight for the L1 reconstruction loss term')
    parser.add_argument('--lpips_weight', type=float, default=0.5, help='Weight for the LPIPS perceptual loss term')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader worker processes')

    # Logging and Saving Args
    parser.add_argument('--checkpoint_dir', type=str, default='/media/hoangdv/vae_checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='/media/hoangdv/vae_logs', help='Directory for TensorBoard logs')
    parser.add_argument('--save_interval', type=int, default=10, help='Save checkpoint every N epochs')

    args = parser.parse_args()

    # --- Print Configuration ---
    print(f"--- VAE Training Configuration ---") # write message on console
    for arg, value in vars(args).items():
        print(f"{arg}: {value}") # write message on console
    print(f"Effective Batch Size: {args.batch_size * args.accumulation_steps}") # write message on console
    print(f"---------------------------------") # write message on console

    # Call the main training function
    train_vae(args)
