# main_train_minimal_dataset.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# Import Dataset from torch.utils.data
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime
import math
import argparse
# Import PIL.Image for loading images
from PIL import Image
from test import UNet
from test_diffusion import DiffusionModelPyTorch, GradientAccumulatorPyTorch

# --- Simplified StanfordCarsCustom Dataset Class ---
class StanfordCarsCustom(Dataset):
    def __init__(self, image_folder, transform = None):
        """
        Args:
            image_folder (str): Path to the folder containing ONLY car images.
            transform (callable, optional): Optional transform to be applied.
        """
        super().__init__()
        # Directly list all entries in the directory. Assumes all are image files.
        # WARNING: This will fail if the directory doesn't exist or contains non-files (subdirectories).
        self.images = [os.path.join(image_folder, file) for file in os.listdir(image_folder)]
        self.transform = transform
        # write message on console about the number of files found (treats everything as an image)
        print(f"Found {len(self.images)} files in {image_folder}.")

    def __len__(self):
        """Returns the total number of files found."""
        return len(self.images)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the image file to retrieve.

        Returns:
            torch.Tensor: Transformed image tensor with an added dimension [1, C, H, W].
                         Returning just 'image' (shape [C, H, W]) is usually preferred for DataLoader.
            int: A dummy label (0).
        """
        image_file = self.images[index]
        # WARNING: This will fail if image_file is not a valid image or is corrupted.
        image = Image.open(image_file).convert("RGB") # Ensure 3 channels (RGB)

        # Apply transformations if they exist
        if self.transform:
            image = self.transform(image)

        # Return the transformed image with an added leading dimension and a dummy label
        # Note: The standard DataLoader expects [C, H, W] from __getitem__.
        # Returning image[None] makes it [1, C, H, W], which might cause shape mismatches later.
        return image[None], 0


# --- Main Training Function (uses hardcoded path) ---
def main(args):
    """
    Main function to set up and run the diffusion model training.
    Uses a hardcoded path for the dataset image folder and the simplified Dataset class.
    """
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}") # write message on console

    # --- <<< HARDCODED IMAGE FOLDER PATH >>> ---
    hardcoded_image_folder = './data/stanford-cars-kaggle/car_data/car_data/train/'
    print(f"Using hardcoded image folder: {hardcoded_image_folder}") # write message on console
    # --- <<< END HARDCODING >>> ---

    # --- Dataset and DataLoader ---
    img_transforms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(), # Converts PIL Image [0, 255] to Tensor [0.0, 1.0]
    ])

    # --- Load using Simplified Custom Dataset ---
    try:
        # Instantiate the simplified custom dataset
        train_dataset = StanfordCarsCustom(
            image_folder=hardcoded_image_folder,
            transform=img_transforms
        )
        # Check if any files were listed (doesn't guarantee they are valid images)
        if len(train_dataset) == 0:
            # write error message on console if directory was empty or inaccessible
            print(f"ERROR: No files found in the directory: {hardcoded_image_folder}. Check the path.")
            return
    except FileNotFoundError:
        # write error message on console if the hardcoded directory doesn't exist
        print(f"ERROR: Directory not found at '{hardcoded_image_folder}'. Ensure it exists.")
        return
    except Exception as e:
        # Catch other potential errors during init (e.g., permission denied)
        # write error message on console
        print(f"ERROR: An unexpected error occurred while initializing the dataset: {e}")
        return
    # --- End Dataset Loading ---

    # Create the DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # --- Model Definition (Imported) ---
    unet_model = UNet(
        in_channels=args.img_channels,
        out_channels=args.img_channels,
        base_dim=args.unet_base_dim,
        dim_mults=tuple(args.unet_dim_mults),
    ).to(device)
    print(f"UNet model initialized with base_dim={args.unet_base_dim}, dim_mults={tuple(args.unet_dim_mults)}") # write message on console

    # --- Diffusion Helper (Imported) ---
    diffusion_helper = DiffusionModelPyTorch(
        img_size=args.img_size,
        img_channels=args.img_channels,
        timesteps=args.timesteps,
        device=device
    )

    # --- Optimizer ---
    optimizer = optim.AdamW(
        unet_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    print(f"Optimizer: AdamW, LR: {args.learning_rate}, Weight Decay: {args.weight_decay}") # write message on console

    # --- Gradient Accumulator (Imported) ---
    accumulator = GradientAccumulatorPyTorch(
        model=unet_model,
        optimizer=optimizer,
        steps=args.accumulation_steps,
        device=device
    )
    print(f"Gradient Accumulation Steps: {args.accumulation_steps}") # write message on console

    # --- Start Training ---
    print("\nStarting training process...") # write message on console
    try:
        diffusion_helper.train_with_accumulation(
            dataset_loader=train_loader,
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
    parser.add_argument('--img_size', type=int, default=32, help='Target image size')
    parser.add_argument('--img_channels', type=int, default=3, help='Number of image channels')
    # Training args
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per device')
    parser.add_argument('--accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Optimizer weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader worker processes')
    # Diffusion args
    parser.add_argument('--timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    # UNet args
    parser.add_argument('--unet_base_dim', type=int, default=64, help='Base channel dimension for UNet')
    parser.add_argument('--unet_dim_mults', type=int, nargs='+', default=[1, 2, 4], help='Channel multipliers for UNet')
    # Logging/Saving args
    parser.add_argument('--log_dir', type=str, default='.', help='Base directory for TensorBoard logs')
    parser.add_argument('--checkpoint_dir', type=str, default='.', help='Base directory for checkpoints')

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