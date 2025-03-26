import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter # Use PyTorch's TensorBoard
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime
import math
import matplotlib.pyplot as plt
from test import UNet
from util import SinusoidalPositionEmbeddings

# --- PyTorch Gradient Accumulator ---
class GradientAccumulatorPyTorch:
    def __init__(self, model, optimizer, steps=32, device='cpu'):
        """
        Initializes the Gradient Accumulator for PyTorch.

        Args:
            model (nn.Module): The PyTorch model.
            optimizer (optim.Optimizer): The PyTorch optimizer.
            steps (int): The number of steps to accumulate gradients over.
            device (str or torch.device): The device to run on ('cpu' or 'cuda').
        """
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.device = device
        self.current_step = 0
        # No need to manually store accumulated grads like in TF;
        # PyTorch accumulates in .grad attribute until optimizer.zero_grad() is called.

    def train_step(self, x_batch, t_batch, noise):
        """
        Performs one training step with gradient accumulation logic.

        Args:
            x_batch (torch.Tensor): Batch of noisy images.
            t_batch (torch.Tensor): Batch of timesteps.
            noise (torch.Tensor): The noise added to create x_batch.

        Returns:
            torch.Tensor: The unscaled loss value for the current batch.
        """
        self.model.train() # Set model to training mode

        # Move data to the correct device
        x_batch = x_batch.to(self.device)
        t_batch = t_batch.to(self.device)
        noise = noise.to(self.device)

        # Forward pass
        predicted_noise = self.model(x_batch, t_batch)

        # Calculate loss
        loss = F.mse_loss(predicted_noise, noise) # Use PyTorch's MSE loss

        # Scale loss for accumulation
        # Loss is averaged over accumulated steps IN TOTAL before backward()
        # Each backward() call adds gradients, so we scale before calling it.
        scaled_loss = loss / self.steps

        # Backward pass - gradients accumulate in model parameters' .grad attribute
        scaled_loss.backward()

        self.current_step += 1

        # Apply gradients if accumulation steps are reached
        if self.current_step >= self.steps:
            self.optimizer.step() # Apply accumulated gradients
            self.optimizer.zero_grad() # Reset gradients for the next accumulation cycle
            self.current_step = 0

        # Return the unscaled loss for logging purposes
        # Detach to prevent holding onto computation graph unnecessarily for logging
        return loss.detach()

# --- PyTorch Diffusion Model ---
class DiffusionModelPyTorch:
    def __init__(
        self,
        img_size=32,
        img_channels=3,
        timesteps=1000,
        device='cpu' # Add device parameter
    ):
        """
        Initialize the diffusion model in PyTorch.

        Args:
            img_size (int): Image dimensions (assumed square).
            img_channels (int): Number of image channels.
            timesteps (int): Number of diffusion steps.
            device (str or torch.device): Device for tensors ('cpu' or 'cuda').
        """
        self.img_size = img_size
        self.img_channels = img_channels
        self.timesteps = timesteps
        self.device = device

        # Define cosine noise schedule using PyTorch tensors
        def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype=dtype)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1. - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0., 0.999) # Use PyTorch's clip

        self.betas = cosine_beta_schedule(timesteps).to(self.device)

        # Pre-calculate diffusion parameters using PyTorch tensors
        self.alphas = (1.0 - self.betas).to(self.device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(self.device)
        # Use torch.cat instead of np.append for PyTorch tensors
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), self.alphas_cumprod[:-1]], dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(self.device)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas).to(self.device)

        # Parameters for sampling
        self.posterior_variance = (self.betas *
            (1.0 - self.alphas_cumprod_prev) /
            (1.0 - self.alphas_cumprod)).to(self.device)
        # Ensure no NaN from division by zero (can happen at t=0 if not careful)
        self.posterior_variance = torch.nan_to_num(self.posterior_variance, nan=0.0, posinf=0.0, neginf=0.0)


    def _extract(self, a, t, x_shape):
        """ Helper function to extract specific coefficients at time t and reshape """
        batch_size = t.shape[0]
        # Use tensor indexing instead of tf.gather
        out = a[t]
        # Use view for reshaping
        return out.view(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def q_sample(self, x_0, t):
        """
        Add noise to images following the forward diffusion process (PyTorch version).

        Args:
            x_0 (torch.Tensor): Input images, shape [B, C, H, W]. Assumed to be in [0, 255] range initially.
            t (torch.Tensor): Timesteps, shape [B].

        Returns:
            torch.Tensor: Noisy images x_t at timestep t, range [-1, 1].
            torch.Tensor: The noise added to the images.
        """
        # Cast to float32, normalize to [0, 1], then scale to [-1, 1]
        x_0 = x_0.float() / 255.0
        x_0 = x_0 * 2.0 - 1.0
        x_0 = torch.clamp(x_0, -1.0, 1.0) # Use PyTorch's clamp

        # Create random noise
        noise = torch.randn_like(x_0, device=x_0.device)

        # No need to slice t if it's already generated with the correct batch size

        # Get pre-calculated parameters using the helper function
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        # Forward diffusion equation
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        x_t = torch.clamp(x_t, -1.0, 1.0)

        # NO clear_session equivalent or need in PyTorch

        return x_t, noise

    def train_with_accumulation(self, dataset, model, accumulator, optimizer, epochs=30, log_dir_base='/home/hoang/python/pytorch_diffusion', checkpoint_dir_base='/home/hoang/python/pytorch_diffusion'):
        """
        Train with gradient accumulation in PyTorch.
        Assumes dataset yields batches of images (e.g., DataLoader).
        """
        model.to(self.device) # Ensure model is on the correct device

        # Create directories for checkpoints and logs
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(log_dir_base, 'logs', timestamp)
        checkpoint_dir = os.path.join(checkpoint_dir_base, 'temp_checkpoints', timestamp)
        best_checkpoint_path = os.path.join(checkpoint_dir, 'diffusion_model_best.pth')
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Set up TensorBoard writer
        writer = SummaryWriter(log_dir)

        global_step = 0 # Counter for optimizer steps (after accumulation)
        batch_step = 0  # Counter for individual batches processed
        best_loss = float('inf')

        print(f"Starting training on device: {self.device}")
        print(f"Logging to: {log_dir}")
        print(f"Saving checkpoints to: {checkpoint_dir}")

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            model.train() # Set model to training mode for the epoch
            progress_bar = tqdm(total=len(dataset), desc="Training")
            epoch_losses = []

            for batch_idx, x_batch in enumerate(dataset):
                # --- Data Loading ---
                # Assume dataset provides image tensors directly
                # Move initial batch data to device if not already done by DataLoader
                x_batch = x_batch.to(self.device)
                actual_batch_size = x_batch.shape[0]

                # --- Timestep Sampling ---
                # Sample random timesteps CORRECTLY
                t = torch.randint(0, self.timesteps, (actual_batch_size,), device=self.device, dtype=torch.long)

                # --- Forward Diffusion ---
                x_t, noise_added = self.q_sample(x_batch, t)

                # --- Accumulator Step ---
                # Pass data to accumulator (handles model forward, loss, backward)
                loss_value_tensor = accumulator.train_step(x_t, t, noise_added)
                loss_value = loss_value_tensor.item() # Get Python float for logging/printing
                epoch_losses.append(loss_value)

                # --- Logging (per batch) ---
                writer.add_scalar('Loss/batch', loss_value, batch_step)

                # Log images periodically (less frequently than every batch usually)
                if batch_step % 500 == 0:
                     # Log original images (normalize [0, 255] to [0, 1])
                    writer.add_images("Original Images", x_batch.float() / 255.0, global_step=batch_step, dataformats='NCHW')
                     # Log noisy images (normalize [-1, 1] to [0, 1])
                    writer.add_images("Noisy Images", (x_t + 1.0) / 2.0, global_step=batch_step, dataformats='NCHW')

                # --- Update Counters and Progress Bar ---
                batch_step += 1
                if accumulator.current_step == 0: # Check if optimizer step was just performed
                    global_step += 1

                progress_bar.update(1)
                current_t_display = t[0].item() # Display first timestep in batch
                progress_bar.set_description(f"Loss: {loss_value:.4f} (t={current_t_display})")

            # --- End of Epoch ---
            mean_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch+1} Average Loss: {mean_loss:.4f}")
            writer.add_scalar('Loss/epoch', mean_loss, epoch)

            # Save best model based on epoch loss
            if mean_loss < best_loss:
                best_loss = mean_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, best_checkpoint_path)
                print(f"Saved new best model checkpoint to {best_checkpoint_path}")

            # Generate and log images periodically
            if (epoch + 1) % 5 == 0:
                print("Generating images...")
                generated = self.generate_images(model, num_images=4) # Use num_images=4 for grid
                writer.add_images("Generated Images", generated, global_step=epoch, dataformats='NCHW')
                print("Generated images logged to TensorBoard.")

            # Save model weights periodically (optional, keep last few)
            if (epoch + 1) % 10 == 0:
                 epoch_checkpoint_path = os.path.join(checkpoint_dir, f'diffusion_model_epoch_{epoch+1}.pth')
                 torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': mean_loss,
                 }, epoch_checkpoint_path)
                 print(f"Saved epoch checkpoint to {epoch_checkpoint_path}")

            progress_bar.close()

        writer.close()
        print("Training finished.")


    @torch.no_grad() # Decorator to disable gradient calculation during inference
    def p_sample(self, model, x_t, t_int):
        """
        Sample from the model at timestep t (PyTorch version).

        Args:
            model (nn.Module): The neural network model.
            x_t (torch.Tensor): Current noisy image, shape [B, C, H, W].
            t_int (int): Current timestep as an integer.

        Returns:
            torch.Tensor: Predicted less noisy image at timestep t-1.
        """
        model.eval() # Set model to evaluation mode
        batch_size = x_t.shape[0]
        # Create tensor for timestep t
        t_tensor = torch.full((batch_size,), t_int, dtype=torch.long, device=self.device)

        # Predict the noise component
        predicted_noise = model(x_t, t_tensor)

        # Get parameters for timestep t using tensor indexing
        alpha_t = self.alphas[t_int] # scalar
        alpha_cumprod_t = self.alphas_cumprod[t_int] # scalar
        beta_t = self.betas[t_int] # scalar
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_int] # scalar
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t_int] # scalar

        # Calculate the mean for sampling using the predicted noise
        # Equation: 1/sqrt(alpha_t) * (x_t - beta_t / sqrt(1 - alpha_cumprod_t) * predicted_noise)
        coeff = beta_t / sqrt_one_minus_alpha_cumprod_t
        mean_pred = sqrt_recip_alpha_t * (x_t - coeff * predicted_noise)
        # No need to predict x_0 separately if using this DDPM sampling eq directly

        # Get posterior variance
        posterior_variance_t = self.posterior_variance[t_int] # scalar

        # Sample noise z ~ N(0, I) only if t > 0
        noise = torch.randn_like(x_t) if t_int > 0 else torch.zeros_like(x_t)

        # Calculate final sample x_{t-1}
        x_t_minus_1 = mean_pred + torch.sqrt(posterior_variance_t) * noise
        x_t_minus_1 = torch.clamp(x_t_minus_1, -1.0, 1.0)

        return x_t_minus_1

    @torch.no_grad() # Decorator for inference mode
    def generate_images(self, model, num_images=4):
        """
        Generate images using the diffusion model (PyTorch version).

        Args:
            model (nn.Module): The neural network model.
            num_images (int): Number of images to generate.

        Returns:
            torch.Tensor: Generated images in [0, 1] range, shape [N, C, H, W].
        """
        model.eval() # Set model to evaluation mode
        print(f"Generating {num_images} images on device {self.device}...")
        # Start with pure noise on the correct device
        x_t = torch.randn(
            (num_images, self.img_channels, self.img_size, self.img_size),
            device=self.device
        )

        # Sample step by step, from t=T-1 down to t=0
        for t in tqdm(range(self.timesteps - 1, -1, -1), desc="Generating"):
            x_t = self.p_sample(model, x_t, t)

        # Convert from [-1, 1] range to [0, 1]
        generated_images = (x_t + 1.0) / 2.0
        generated_images = torch.clamp(generated_images, 0.0, 1.0)

        print("Image generation complete.")
        return generated_images

    def save_model_weights(self, model, save_path):
        """
        Save only the model's state_dict (weights).

        Args:
            model (nn.Module): The trained PyTorch model.
            save_path (str): Path to save the weights file (e.g., 'model.pth').
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model weights saved to {save_path}")

    def load_model_weights(self, model, model_path):
        """
        Load model weights from a state_dict file.

        Args:
            model (nn.Module): The PyTorch model instance (must have the same architecture).
            model_path (str): Path to the saved weights file.
        """
        if os.path.exists(model_path):
             # Load state dict, mapping to the model's device
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device) # Ensure model is on the correct device
            print(f"Model weights loaded from {model_path}")
        else:
            print(f"Warning: Model weights path not found: {model_path}")


    def convert_to_raw_images(self, images_tensor):
        """
        Convert generated PyTorch tensors to numpy arrays in uint8 format.

        Args:
            images_tensor (torch.Tensor): Generated images tensor ([N, C, H, W], range [0, 1]).

        Returns:
            np.ndarray: Raw images in uint8 format ([N, H, W, C]).
        """
        # Ensure tensor is on CPU and detach from graph
        images_tensor = images_tensor.detach().cpu()
        # Permute channels from NCHW to NHWC for standard image format
        images_tensor = images_tensor.permute(0, 2, 3, 1)
        # Scale to [0, 255] and convert to uint8
        images_np = (images_tensor * 255.0).clamp(0, 255).byte().numpy()
        return images_np

    @torch.no_grad()
    def visualize_diffusion_steps(self, model, x_0_tensor, num_steps_to_show=10):
        """
        Visualize the diffusion process (PyTorch version).

        Args:
            model (nn.Module): The neural network model.
            x_0_tensor (torch.Tensor): Original clean image tensor ([C, H, W], range [0, 255]).
            num_steps_to_show (int): Number of intermediate steps to display.
        """
        model.eval()
        if x_0_tensor.dim() == 3:
             x_0_batch = x_0_tensor.unsqueeze(0).to(self.device) # Add batch dim, move to device
        elif x_0_tensor.dim() == 4:
             x_0_batch = x_0_tensor.to(self.device) # Assume already has batch dim
        else:
             raise ValueError("Input tensor x_0_tensor must have 3 or 4 dimensions")

        # Select timesteps
        step_indices = torch.linspace(0, self.timesteps - 1, num_steps_to_show, dtype=torch.long, device='cpu') # Use cpu for indices

        # Forward process
        forward_images = []
        print("Visualizing forward process...")
        for t_int in step_indices.tolist():
            t = torch.tensor([t_int], device=self.device) # Create tensor for timestep
            noisy_image_t, _ = self.q_sample(x_0_batch, t) # Use q_sample
            # Convert [-1, 1] to [0, 1] for display, move to CPU, remove batch dim
            img_display = (noisy_image_t[0].cpu() + 1.0) / 2.0
            forward_images.append(img_display.permute(1, 2, 0).clamp(0, 1).numpy()) # CHW to HWC

        # Reverse process
        reverse_images = []
        x_t = torch.randn_like(x_0_batch) # Start with noise on the correct device
        print("Visualizing reverse process...")
        timesteps_for_reverse_vis = step_indices.tolist()[::-1] # Reverse order for display matching
        current_vis_idx = 0

        for t_int in tqdm(range(self.timesteps - 1, -1, -1), desc="Reverse"):
            x_t = self.p_sample(model, x_t, t_int)
            # Store image if the timestep matches one we want to visualize
            if current_vis_idx < len(timesteps_for_reverse_vis) and t_int == timesteps_for_reverse_vis[current_vis_idx]:
                img_display = (x_t[0].cpu() + 1.0) / 2.0 # Convert [-1, 1] to [0, 1]
                reverse_images.append(img_display.permute(1, 2, 0).clamp(0, 1).numpy()) # CHW to HWC
                current_vis_idx += 1

        # Reverse the collected reverse images to show T -> 0 order
        reverse_images = reverse_images[::-1]

        # Plotting (similar to TF version)
        num_forward = len(forward_images)
        num_reverse = len(reverse_images)
        total_plots = num_forward + num_reverse
        if total_plots == 0:
             print("No images generated for visualization.")
             return

        plt.figure(figsize=(total_plots * 2, 4))

        # Plot forward
        for i, img in enumerate(forward_images):
            ax = plt.subplot(2, max(num_forward, num_reverse), i + 1)
            ax.imshow(img)
            ax.set_title(f"Forward t={step_indices[i].item()}")
            ax.axis('off')

        # Plot reverse
        for i, img in enumerate(reverse_images):
            ax = plt.subplot(2, max(num_forward, num_reverse), i + max(num_forward, num_reverse) + 1)
            ax.imshow(img)
            # Titles correspond to the step index list in reverse order
            ax.set_title(f"Reverse t={step_indices[num_reverse-1-i].item()}")
            ax.axis('off')

        plt.tight_layout()
        plt.savefig('diffusion_visualization_pytorch.png')
        print("Saved visualization to diffusion_visualization_pytorch.png")
        plt.show()