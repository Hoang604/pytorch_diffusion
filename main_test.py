# main_train_minimal_dataset.py
import torch
import torch.optim as optim
# Import Dataset from torch.utils.data
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
import bitsandbytes as bnb
import argparse
# Import PIL.Image for loading images
from PIL import Image
from unet import UNet
from diffusion import DiffusionModelPyTorch, GradientAccumulatorPyTorch

def main(args):
    """
    Main function to set up and run the diffusion model training.
    Uses a hardcoded path for the dataset image folder and the simplified Dataset class.
    """
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}") # write message on console

    # Simple dataset for loading images from a folder
    class SimpleImageDataset(Dataset):
        def __init__(self, folder_path, img_size):
            self.folder_path = folder_path
            self.image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
        def __len__(self):
            return len(self.image_files)
        
        def __getitem__(self, idx):
            img_path = os.path.join(self.folder_path, self.image_files[idx])
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
            return image

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

    unet_model = UNet(
        in_channels=args.img_channels,
        out_channels=args.img_channels,
        base_dim=args.unet_base_dim,
        dim_mults=tuple(args.unet_dim_mults),
    ).to(device)
    print(f"UNet model initialized with base_dim={args.unet_base_dim}, dim_mults={tuple(args.unet_dim_mults)}") # write message on console

    print("Initializing model with random weights")
    # --- Diffusion Helper ---
    diffusion_helper = DiffusionModelPyTorch(
        img_size=args.img_size,
        img_channels=args.img_channels,
        timesteps=args.timesteps,
        device=device
    )

    if os.path.exists(args.weights_path):
        diffusion_helper.load_model_weights(unet_model, args.weights_path, verbose=args.verbose)
        print(f"Loaded model weights from {args.weights_path}") # write message on console
    # --- Optimizer ---
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

    diffusion_helper.visualize_diffusion_steps(unet_model, train_dataset[100])

    # --- Start Training ---
    print("\nStarting training process...") # write message on console
    try:
        diffusion_helper.train_with_accumulation(
            dataset=train_loader,
            model=unet_model,
            accumulator=accumulator,
            optimizer=optimizer,
            epochs=args.epochs,
            log_dir_base=args.log_dir,
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
    parser.add_argument('--img_size', type=int, default=128, help='Target image size')
    parser.add_argument('--img_channels', type=int, default=3, help='Number of image channels')
    # Training args
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--accumulation_steps', type=int, default=128, help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Optimizer weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader worker processes')
    # Diffusion args
    parser.add_argument('--timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    parser.add_argument('--weights_path', type=str, default="/home/hoang/python/pytorch_diffusion/temp_checkpoints/20250327-004444/diffusion_model_best.pth", help='Path to pre-trained model weights')
    # UNet args
    parser.add_argument('--unet_base_dim', type=int, default=64, help='Base channel dimension for UNet')
    parser.add_argument('--unet_dim_mults', type=int, nargs='+', default=[1, 2, 4, 8], help='Channel multipliers for UNet')
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
    main(args)