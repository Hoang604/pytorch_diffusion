# evaluate_loss_per_timestep.py (Grouped Timesteps Version)

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF # For functional transforms like hflip
from PIL import Image as PilImage # For opening images
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter # Optional: For TensorBoard logging
import numpy as np
import argparse
import os
import pickle
import collections # For defaultdict
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# --- Constants ---
GROUP_SIZE = 10 # Group timesteps into blocks of this size

# --- Dataset Definition (copied from user's last version) ---
class SimpleImageDataset(Dataset):
    def __init__(self, folder_path, img_size):
        """
        Initializes the dataset. Now includes logic to handle flipped images.

        Args:
            folder_path (str): Path to the folder containing images.
            img_size (int): Target size for the images (height and width).
        """
        self.folder_path = folder_path
        # Find image files - consider adding more extensions if needed
        try:
            self.image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]) # Added webp
            if not self.image_files:
                raise FileNotFoundError(f"No valid image files found in {folder_path}") # Raise error if no images
        except FileNotFoundError as e:
            print(f"Error: {e}") # write error message on console
            raise # Re-raise the error to stop execution
        except Exception as e:
            print(f"An unexpected error occurred while listing files: {e}") # write error message on console
            raise # Re-raise

        self.original_len = len(self.image_files) # Store original number of files
        self.img_size = img_size # Store img_size for potential error handling

        # Define the main transformations (excluding the flip)
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        print(f"Found {self.original_len} original images in {folder_path}.") # write message on console
        print(f"Total dataset size (with flips): {self.__len__()}") # write message on console


    def __len__(self):
        """ Returns the total size of the dataset (original + flipped). """
        return self.original_len * 2 # Report double the length

    def __getitem__(self, idx):
        """
        Gets an image by index. Handles returning either the original or a flipped version.

        Args:
            idx (int): Index of the item to retrieve (0 to 2*original_len - 1).

        Returns:
            torch.Tensor: The transformed image tensor.
        """
        if not 0 <= idx < self.__len__():
            raise IndexError(f"Index {idx} out of range for dataset size {self.__len__()}") # Handle index out of bounds

        # Determine if this index corresponds to a flipped image
        should_flip = idx >= self.original_len

        # Calculate the index of the original image file
        original_idx = idx % self.original_len

        # Construct the image path
        img_path = os.path.join(self.folder_path, self.image_files[original_idx])

        try:
            # Load the PIL image
            image = PilImage.open(img_path).convert("RGB")

            # Apply horizontal flip if needed, *before* other transforms
            if should_flip:
                image = TF.hflip(image) # Apply horizontal flip functionally

            # Apply the main transformations (Resize, ToTensor, Normalize)
            transformed_image = self.transform(image)

            return transformed_image

        except Exception as e:
            # Handle potential errors during loading or transforming
            print(f"\nError loading or processing image at index {idx} (original file: {img_path}): {e}") # write message on console
            # Return a dummy tensor of the correct size to avoid crashing the batch collation
            print("Returning dummy tensor to continue batch processing.") # write message on console
            return torch.zeros((3, self.img_size, self.img_size))
            # Or re-raise the exception if stopping is preferred:
            # raise RuntimeError(f"Failed to process image: {img_path}") from e


# --- Import required classes from project ---
try:
    from unet import UNet
    from diffusion import DiffusionModelPyTorch
except ImportError as e:
    print(f"Error: Could not import required modules (UNet, DiffusionModelPyTorch).") # write error message on console
    print(f"Ensure unet_test.py and diffusion_test.py are accessible.") # write error message on console
    print(f"Details: {e}") # write error message on console
    exit()

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Diffusion Model Loss per Timestep Group")

    # --- Required Arguments ---
    # Provide default values based on user's last input, but ideally these should be required without defaults
    parser.add_argument('--weights_path', type=str, default="./temp_checkpoints/20250328-215458/diffusion_model_second_best.pth",
                        help='Path to pre-trained model weights (.pth)')
    parser.add_argument('--dataset_path', type=str, default="./simpler_data",
                        help='Path to the image dataset folder')

    # --- Optional Arguments with Defaults ---
    parser.add_argument('--output_dir', type=str, default=f"./evaluation_results_grouped_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                        help='Directory to save results (default: ./evaluation_results_grouped_TIMESTAMP)')
    parser.add_argument('--num_samples', type=int, default=20000,
                        help=f'Maximum number of samples (image instances) to evaluate (default: 1000). Dataset size is doubled by flips.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation (default: 64)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use: cuda or cpu (default: auto)')
    parser.add_argument('--save_plot', action='store_true', default=True,
                        help='Save the loss vs timestep group plot (default: True)')
    parser.add_argument('--show_plot', action='store_true', default=True,
                        help='Show the loss vs timestep group plot immediately after evaluation (default: True)')
    parser.add_argument('--log_tensorboard', action='store_true', default=True, # User had True here
                        help='Log results to TensorBoard (default: True)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of DataLoader workers (default: 4)')

    # --- Model/Diffusion Parameters (should match training) ---
    parser.add_argument('--img_size', type=int, default=32,
                        help='Target image size (default: 32)')
    parser.add_argument('--img_channels', type=int, default=3,
                        help='Number of image channels (default: 3)')
    parser.add_argument('--timesteps', type=int, default=1000,
                        help='Number of diffusion timesteps (default: 500)')
    parser.add_argument('--unet_base_dim', type=int, default=256,
                        help='Base channel dimension for UNet (default: 256)')
    parser.add_argument('--unet_dim_mults', type=int, nargs='+', default=[1, 2, 4],
                        help='Channel multipliers for UNet (default: [1, 2, 4])')

    args = parser.parse_args()

    # --- Validation for timesteps vs group_size ---
    if args.timesteps % GROUP_SIZE != 0:
        print(f"Warning: Number of timesteps ({args.timesteps}) is not perfectly divisible by GROUP_SIZE ({GROUP_SIZE}). "
              f"The last group will be smaller.") # write warning message on console

    return args

def main(args):
    """Main evaluation function."""
    # 1. Setup: Create output dir, set device
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"Using device: {device}") # write message on console
    print(f"Evaluation results will be saved in: {args.output_dir}") # write message on console

    # 2. Initialize Model and Diffusion Helper
    print("Initializing model and diffusion helper...") # write message on console
    try:
        model = UNet(
            in_channels=args.img_channels,
            out_channels=args.img_channels,
            base_dim=args.unet_base_dim,
            dim_mults=tuple(args.unet_dim_mults),
        ).to(device)

        diffusion_helper = DiffusionModelPyTorch(
            img_size=args.img_size,
            img_channels=args.img_channels,
            timesteps=args.timesteps,
            device=device
        )
    except Exception as e:
        print(f"Error during model or diffusion helper initialization: {e}") # write error message on console
        return

    # 3. Load Weights
    print(f"Loading weights from: {args.weights_path}") # write message on console
    if os.path.exists(args.weights_path):
         try:
             diffusion_helper.load_model_weights(model, args.weights_path, verbose=True)
             print(f"Successfully loaded model weights.") # write message on console
         except Exception as e:
             print(f"Error loading weights: {e}") # write error message on console
             print("Please ensure the weights file is compatible and the model architecture matches.") # write error message on console
             return
    else:
         print(f"Error: Weights path not found: {args.weights_path}") # write error message on console
         return

    model.eval() # Set model to evaluation mode (important!)

    # 4. Load Dataset
    print(f"Loading dataset from: {args.dataset_path}") # write message on console
    try:
        eval_dataset = SimpleImageDataset(folder_path=args.dataset_path, img_size=args.img_size)
    except Exception as e: # Catch errors during dataset init
        print(f"Error initializing dataset: {e}") # write error message on console
        return

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False, # Keep order consistent for evaluation
        num_workers=args.num_workers,
        pin_memory=True if device != torch.device('cpu') else False # Pin memory only for GPU
    )

    # 5. Initialize Loss Storage (Grouped) and Optional TensorBoard
    # Use defaultdict for convenience when adding to lists for new keys
    losses_per_group = collections.defaultdict(list)
    writer = None
    if args.log_tensorboard:
        log_dir = os.path.join(args.output_dir, 'tensorboard_logs')
        try:
            writer = SummaryWriter(log_dir)
            print(f"Logging TensorBoard to: {log_dir}") # write message on console
        except Exception as e:
            print(f"Error initializing TensorBoard SummaryWriter: {e}") # write error message on console
            print("TensorBoard logging will be disabled.") # write message on console
            writer = None # Ensure writer is None if init fails


    # 6. Evaluation Loop
    total_samples_processed = 0
    # Adjust max_samples based on dataset length if num_samples is None or too large
    max_samples = len(eval_dataset) if args.num_samples is None else min(args.num_samples, len(eval_dataset))

    print(f"Starting evaluation for up to {max_samples} samples...") # write message on console

    # Disable gradient calculations for efficiency
    with torch.no_grad():
        progress_bar = tqdm(total=max_samples, desc="Evaluating Batches", unit="sample")
        batch_num = 0
        samples_processed_in_loop = 0
        for x_0_batch in eval_loader:
            # Check if max samples reached before processing batch
            if samples_processed_in_loop >= max_samples:
                break

            x_0_batch = x_0_batch.to(device)
            actual_batch_size = x_0_batch.shape[0]

            # Determine how many samples from this batch to process
            remaining_samples = max_samples - samples_processed_in_loop
            samples_to_process_this_batch = min(actual_batch_size, remaining_samples)

            if samples_to_process_this_batch != actual_batch_size:
                x_0_batch = x_0_batch[:samples_to_process_this_batch]
                actual_batch_size = samples_to_process_this_batch # Update actual batch size for this iteration

            if actual_batch_size == 0: # Should not happen if logic is correct, but safe check
                continue

            # Sample timesteps randomly for the batch
            t_batch = torch.randint(0, args.timesteps, (actual_batch_size,), device=device, dtype=torch.long)

            # --- Generate x_t and target ---
            # Assuming the model was trained with v-prediction
            try:
                x_t, noise_added = diffusion_helper.q_sample(x_0_batch, t_batch)

                # Calculate target_v
                sqrt_alphas_cumprod_t = diffusion_helper._extract(diffusion_helper.sqrt_alphas_cumprod, t_batch, x_t.shape)
                sqrt_one_minus_alphas_cumprod_t = diffusion_helper._extract(diffusion_helper.sqrt_one_minus_alphas_cumprod, t_batch, x_t.shape)
                target_v = sqrt_alphas_cumprod_t * noise_added - sqrt_one_minus_alphas_cumprod_t * x_0_batch
                target_for_loss = target_v
            except Exception as e:
                print(f"\nError during q_sample or target calculation in batch {batch_num}: {e}") # write error message on console
                continue # Skip this batch

            # --- Model prediction ---
            try:
                predicted_output = model(x_t, t_batch)
            except Exception as e:
                print(f"\nError during model prediction in batch {batch_num}: {e}") # write error message on console
                continue # Skip this batch


            # --- Calculate per-item loss ---
            try:
                per_item_loss = F.mse_loss(predicted_output, target_for_loss, reduction='none').mean(dim=[1, 2, 3]) # Average over C, H, W
            except Exception as e:
                 print(f"\nError calculating loss in batch {batch_num}: {e}") # write error message on console
                 continue # Skip this batch


            # --- Store losses in the dictionary based on timestep group ---
            for i in range(actual_batch_size):
                timestep_i = t_batch[i].item()
                loss_i = per_item_loss[i].item()

                # Basic check for valid loss values
                if np.isfinite(loss_i):
                    if 0 <= timestep_i < args.timesteps:
                        # Calculate the group key
                        group_index = timestep_i // GROUP_SIZE
                        start_t = group_index * GROUP_SIZE
                        # Ensure end_t doesn't exceed max timestep
                        end_t = min(start_t + GROUP_SIZE - 1, args.timesteps - 1)
                        group_key = f"{start_t}-{end_t}"

                        # Append loss to the corresponding group list (defaultdict handles creation)
                        losses_per_group[group_key].append(loss_i)
                # else: # Optional warning for non-finite losses
                    # print(f"\nWarning: Encountered non-finite loss ({loss_i}) at timestep {timestep_i}. Skipping.") # write warning message on console

            samples_processed_in_loop += actual_batch_size
            progress_bar.update(actual_batch_size)
            # progress_bar.set_postfix({"Processed": f"{samples_processed_in_loop}/{max_samples}"}) # Removed redundant postfix
            batch_num += 1

        # Ensure progress bar reaches its total if loop finishes early or exactly
        progress_bar.n = samples_processed_in_loop
        progress_bar.refresh()
        progress_bar.close()
        total_samples_processed = samples_processed_in_loop # Use the actual count

    print(f"\nEvaluation finished. Processed {total_samples_processed} samples.") # write message on console

    # 7. Analyze and Save Results
    print("Analyzing results and saving data...") # write message on console
    stats_per_group = {}
    # Sort keys numerically based on the starting timestep of the group
    group_keys_sorted = sorted(losses_per_group.keys(), key=lambda k: int(k.split('-')[0]))

    mean_losses_grouped = []
    median_losses_grouped = []
    group_centers = [] # Use center of group for plotting x-axis
    valid_groups_count = 0

    for group_key in group_keys_sorted:
        if losses_per_group[group_key]: # Check if any losses were collected for this group
            valid_groups_count += 1
            losses_np = np.array(losses_per_group[group_key])
            mean_loss = np.mean(losses_np)
            median_loss = np.median(losses_np)
            std_loss = np.std(losses_np)
            count = len(losses_np)
            stats_per_group[group_key] = {'mean': mean_loss, 'median': median_loss, 'std': std_loss, 'count': count}

            # Append for plotting
            mean_losses_grouped.append(mean_loss)
            median_losses_grouped.append(median_loss)
            start_t = int(group_key.split('-')[0])
            end_t = int(group_key.split('-')[1])
            group_centers.append((start_t + end_t) / 2.0) # Mid-point of the group range

            # Log to TensorBoard if enabled
            if writer:
                try:
                    writer.add_histogram(f'Loss_Distribution_Grouped/Group_{group_key}', losses_np, global_step=0)
                    # Log stats per group - use group start timestep as global step for ordering
                    writer.add_scalar(f'Loss_Stats_Grouped/Mean', mean_loss, global_step=start_t)
                    writer.add_scalar(f'Loss_Stats_Grouped/Median', median_loss, global_step=start_t)
                    writer.add_scalar(f'Loss_Stats_Grouped/StdDev', std_loss, global_step=start_t)
                    writer.add_scalar(f'Loss_Stats_Grouped/Count', count, global_step=start_t)
                except Exception as e:
                     print(f"\nWarning: Error logging to TensorBoard for group {group_key}: {e}") # write warning message on console


    print(f"Analysis complete for {valid_groups_count} timestep groups with data.") # write message on console

    # --- Save the raw loss data (grouped) using pickle ---
    loss_data_filename = 'losses_per_group.pkl'
    loss_data_path = os.path.join(args.output_dir, loss_data_filename)
    try:
        with open(loss_data_path, 'wb') as f:
            # Convert defaultdict back to dict for standard pickle compatibility if needed
            pickle.dump(dict(losses_per_group), f)
        print(f"Saved raw grouped loss data to: {loss_data_path}") # write message on console
    except Exception as e:
        print(f"\nError saving raw grouped loss data: {e}") # write error message on console

    # --- Save the calculated stats (grouped) using pickle ---
    stats_data_filename = 'stats_per_group.pkl'
    stats_data_path = os.path.join(args.output_dir, stats_data_filename)
    try:
        with open(stats_data_path, 'wb') as f:
            pickle.dump(stats_per_group, f)
        print(f"Saved calculated grouped stats to: {stats_data_path}") # write message on console
    except Exception as e:
        print(f"\nError saving grouped stats data: {e}") # write error message on console


    # --- Generate, Show, and Save Plot (using grouped data) ---
    plot_generated = False
    if group_centers: # Only plot if we have data
        try:
            print("Generating plot...") # write message on console
            plt.figure(figsize=(14, 7))
            # Use group_centers for x-axis
            plt.plot(group_centers, mean_losses_grouped, label='Mean Loss per Group', marker='o', linestyle='-', markersize=5)
            plt.plot(group_centers, median_losses_grouped, label='Median Loss per Group', marker='x', linestyle='--', markersize=5, alpha=0.8)
            plt.xlabel(f'Diffusion Timestep (Grouped by {GROUP_SIZE}, Center Point)')
            plt.ylabel('Loss (MSE)')
            plt.title('Grouped Loss vs. Diffusion Timestep')
            plt.legend()
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.tight_layout() # Adjust layout

            if args.save_plot:
                plot_filename = 'grouped_loss_vs_timestep_plot.png'
                plot_path = os.path.join(args.output_dir, plot_filename)
                plt.savefig(plot_path)
                print(f"Saved plot to: {plot_path}") # write message on console
                plot_generated = True

                # Log plot to TensorBoard if enabled and saved
                if writer and plot_generated:
                     try:
                         # Re-open the image to log it
                         with PilImage.open(plot_path) as img:
                             img_tensor = TF.to_tensor(img) # Use functional transform
                         writer.add_image('Grouped_Loss_vs_Timestep_Plot', img_tensor, global_step=0)
                         print("Logged plot to TensorBoard.") # write message on console
                     except Exception as tb_e:
                         print(f"\nWarning: Could not log plot to TensorBoard: {tb_e}") # write warning message on console

            if args.show_plot:
                print("Displaying plot...") # write message on console
                plt.show() # Display the plot
                # save image
                plot_filename = 'grouped_loss_vs_timestep_plot.png'
                plot_path = os.path.join(args.output_dir, plot_filename)
                plt.savefig(plot_path)
                print(f"Saved plot to: {plot_path}")
                

            plt.close() # Close the figure window after saving/showing

        except Exception as e:
            print(f"\nError generating or showing plot: {e}") # write error message on console
            plt.close() # Ensure figure is closed even if error occurs
    else:
        print("No data available from any timestep group to generate plot.") # write message on console

    # Close TensorBoard writer
    if writer:
        writer.close()
        print("TensorBoard logging finished.") # write message on console

    print("Evaluation script finished.") # write message on console

# --- Script Entry Point ---
if __name__ == "__main__":
    args = parse_args()
    main(args)