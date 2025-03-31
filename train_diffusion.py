import torch
from torch.utils.data import DataLoader
from util import SimpleImageDataset
import os
import bitsandbytes as bnb
import argparse
from unet import UNet
from diffusion import DiffusionModelPyTorch, GradientAccumulatorPyTorch, ImageGenerator
from vae import VAE

            
def train_diffusion(args):
    """
    Main function to set up and run the diffusion model training.
    Uses a hardcoded path for the dataset image folder and the simplified Dataset class.
    """
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}") # write message on console


    # --- Load Pre-trained VAE ---
    print("Loading pre-trained VAE model...") # write message on console
    vae = VAE(latent_dim=args.latent_dim).to(device)
    try:
        checkpoint = torch.load(args.vae_weights_path, map_location=device)
        # Adjust loading based on how checkpoint was saved in train_vae.py
        if 'model_state_dict' in checkpoint:
            vae.load_state_dict(checkpoint['model_state_dict'])
        else:
            vae.load_state_dict(checkpoint) # Assume raw state dict was saved
        print(f"Loaded VAE weights from {args.vae_weights_path}") # write message on console
    except FileNotFoundError:
        print(f"Error: VAE weights file not found at {args.vae_weights_path}. Please train VAE first.") # write error message on console
        return
    except Exception as e:
        print(f"Error loading VAE weights: {e}") # write error message on console
        return

    # --- Freeze VAE ---
    vae.eval() # Set VAE to evaluation mode
    for param in vae.parameters():
        param.requires_grad = False
    print("VAE model frozen.") # write message on console

    # Create the dataset with the specified folder
    train_dataset = SimpleImageDataset(folder_path='./simpler_data', img_size=args.img_size)
    print(f"Loaded {len(train_dataset)} images from ./simpler_data")

    # Create the DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # --- Diffusion Helper ---
    diffusion_helper = DiffusionModelPyTorch(
        img_size=args.img_size,
        img_channels=args.img_channels,
        timesteps=args.timesteps,
        device=device
    )
    unet_model = UNet(
        in_channels=args.latent_dim,
        out_channels=args.latent_dim,
        base_dim=args.unet_base_dim,
        dim_mults=tuple(args.unet_dim_mults),
        num_resnet_blocks=3
    ).to(device)
    print(f"Front UNet model initialized with base_dim={args.unet_base_dim}, dim_mults={tuple(args.unet_dim_mults)}") # write message on console

    print("Initializing model with random weights")

    optimizer = bnb.optim.AdamW8bit(
        unet_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    print(f"Optimizer: AdamW, LR: {args.learning_rate}, Weight Decay: {args.weight_decay}") # write message on console

    # --- Gradient Accumulator ---
    accumulator = GradientAccumulatorPyTorch(
        model=unet_model,
        optimizer=optimizer,
        steps=args.accumulation_steps,
    )
    print(f"Gradient Accumulation Steps: {args.accumulation_steps}") # write message on console

    if os.path.exists(args.weights_path):
        start_epoch, best_loss = diffusion_helper.load_checkpoint_for_resume(unet_model, optimizer=optimizer, checkpoint_path=args.weights_path)
        print(f"Loaded model weights from {args.weights_path}") # write message on console
        
    def visualize_diffusion():
        print("Generating sample images...")
        image_generator = ImageGenerator()
        images = []
        for i in range(15):
            images.append(image_generator.generate_images(unet_model).squeeze(0))
            print(images[i].shape)

        import matplotlib.pyplot as plt
        x = plt.subplots(1, 15, figsize=(15, 1))
        for i in range(15):
            plt.subplot(1, 15, i+1)
            plt.imshow(images[i].permute(1, 2, 0).detach().cpu().numpy())
            plt.axis('off')
        plt.show()
    
    visualize_diffusion()

    diffusion_helper.visualize_diffusion_steps(unet_model, train_dataset[0])
    input("Press Enter to continue...")

    # --- Start Training ---
    print("\nStarting training process...") # write message on console
    try:
        diffusion_helper.train_with_accumulation(
            dataset=train_loader,
            vae=vae,
            model=unet_model,
            accumulator=accumulator,
            optimizer=optimizer,
            epochs=args.epochs,
            start_epoch=start_epoch,
            best_loss=best_loss,
            log_dir_base=args.log_dir,
            log_dir='/home/hoang/python/pytorch_diffusion/logs/20250329-142504',
            checkpoint_dir='/home/hoang/python/pytorch_diffusion/temp_checkpoints/20250329-142504',
            checkpoint_dir_base=args.checkpoint_dir
        )
    except Exception as train_e:
        # Catch potential errors during training (e.g., from DataLoader if __getitem__ fails)
        # write error message on console
        print(f"\nERROR occurred during training: {train_e}")
        print("This might be due to issues reading image files or shape mismatches.") 

# --- Script Entry Point ---
if __name__ == "__main__": 
    # Set up argument parser (same as before, without --image_folder)
    parser = argparse.ArgumentParser(description="Train Diffusion Model (Simplified Dataset, Hardcoded Path)")
    # Dataset args
    parser.add_argument('--img_size', type=int, default=32, help='Target image size')
    parser.add_argument('--laten_dim', type=int, default=3, help='Number of image channels')
    # Training args
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--accumulation_steps', type=int, default=256, help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Optimizer weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader worker processes')
    # Diffusion args
    parser.add_argument('--timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    parser.add_argument('--weights_path', type=str, default="/media/hoangdv/temp_checkpoints/20250329-142504/diffusion_model_best.pth", help='Path to pre-trained model weights')
    # VAE args
    parser.add_argument('--vae_weights_path', type=str, default="/media/hoangdv/vae_checkpoints/20250329-142504/vae_model_best.pth", help='Path to pre-trained VAE weights')
    # UNet args
    parser.add_argument('--unet_base_dim', type=int, default=256, help='Base channel dimension for UNet')
    parser.add_argument('--unet_dim_mults', type=int, nargs='+', default=[1, 2, 4], help='Channel multipliers for UNet')
    # Logging/Saving args
    parser.add_argument('--log_dir', type=str, default='.', help='Base directory for TensorBoard logs')
    parser.add_argument('--checkpoint_dir', type=str, default='.', help='Base directory for checkpoints')
    # Loading model args
    parser.add_argument('--verbose', action='store_true', help='Print detailed information about weight loading')

    args = parser.parse_args()

    # --- Print Configuration ---
    effective_batch_size = args.batch_size * args.accumulation_steps
    print(f"--- Configuration ---") # write message on console
    # Image folder path is hardcoded and printed inside main()
    print(f"Image Size: {args.img_size}x{args.img_size}") # write message on console
    print(f"Batch Size (per device): {args.batch_size}") # write message on console
    print(f"Accumulation Steps: {args.accumulation_steps}") # write message on console
    print(f"Effective Batch Size: {effective_batch_size}") # write message on console
    print(f"Epochs: {args.epochs}") # write message on console
    print(f"Learning Rate: {args.learning_rate}") # write message on console
    print(f"Timesteps: {args.timesteps}") # write message on console
    print(f"UNet Base Dim: {args.unet_base_dim}") # write message on console
    print(f"UNet Dim Mults: {tuple(args.unet_dim_mults)}") # write message on console
    print(f"Log Directory Base: {args.log_dir}") # write message on console
    print(f"Checkpoint Directory Base: {args.checkpoint_dir}") # write message on console
    print(f"--------------------") # write message on console

    # Call the main function
    train_diffusion(args)