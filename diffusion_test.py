import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter # Use PyTorch's TensorBoard
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime
import matplotlib.pyplot as plt

# --- PyTorch Gradient Accumulator ---
class GradientAccumulatorPyTorch:
    def __init__(self, model, optimizer, steps=32, device='cuda'):
        """
        Initializes the Gradient Accumulator for PyTorch.

        Args:
            model (nn.Module): The PyTorch model.
            optimizer (optim.Optimizer): The PyTorch optimizer.
            steps (int): The number of steps to accumulate gradients over.
            device (str or torch.device): The device to run on ('cuda' or 'cpu').
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
        device='cuda' # Add device parameter
    ):
        """
        Initialize the diffusion model in PyTorch.

        Args:
            img_size (int): Image dimensions (assumed square).
            img_channels (int): Number of image channels.
            timesteps (int): Number of diffusion steps.
            device (str or torch.device): Device for tensors ('cuda' or 'cpu').
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
            x_0 (torch.Tensor): Input images, shape [B, C, H, W]. Assumed to be in [-1 , 1] range initially.
            t (torch.Tensor): Timesteps, shape [B,].

        Returns:
            torch.Tensor: Noisy images x_t at timestep t, range [-1, 1].
            torch.Tensor: The noise added to the images.
        """
        x_0 = x_0.to(self.device) # Move to correct device
        x_0 = x_0.float()

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

    def train_with_accumulation(self, dataset, model, accumulator, optimizer, epochs=30, start_epoch=0, best_loss=float('inf'), log_dir=None, checkpoint_dir=None, log_dir_base='/home/hoang/python/pytorch_diffusion', checkpoint_dir_base='/home/hoang/python/pytorch_diffusion'):
        """
        Train with gradient accumulation in PyTorch.
        Assumes dataset yields batches of images (e.g., DataLoader).
        """
        model.to(self.device) # Ensure model is on the correct device
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        

        if log_dir is not None:
            log_dir = log_dir
        else:
            log_dir = os.path.join(log_dir_base, 'logs', timestamp)

        if checkpoint_dir is not None:
            checkpoint_dir = checkpoint_dir
        else:
            checkpoint_dir = os.path.join(checkpoint_dir_base, 'temp_checkpoints', timestamp)

        best_checkpoint_path = os.path.join(checkpoint_dir, 'diffusion_model_best.pth')
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Set up TensorBoard writer
        writer = SummaryWriter(log_dir)

        global_step = start_epoch * len(dataset) # Counter for optimizer steps (after accumulation)
        batch_step = start_epoch * len(dataset)  # Counter for individual batches processed
        best_loss = float('inf')

        print(f"Starting training on device: {self.device}")
        print(f"Logging to: {log_dir}")
        print(f"Saving checkpoints to: {checkpoint_dir}")

        for epoch in range(start_epoch, epochs):
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
                sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
                sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

                # v = sqrt(alpha_bar_t) * epsilon - sqrt(1 - alpha_bar_t) * x_0
                target_v = sqrt_alphas_cumprod_t * noise_added - sqrt_one_minus_alphas_cumprod_t * x_batch

                # --- Accumulator Step ---
                # Pass data to accumulator (handles model forward, loss, backward)
                loss_value_tensor = accumulator.train_step(x_t, t, target_v)
                loss_value = loss_value_tensor.item() # Get Python float for logging/printing
                epoch_losses.append(loss_value)

                # --- Logging (per batch) ---
                writer.add_scalar('Loss/batch', loss_value, batch_step)

                # Log images periodically (less frequently than every batch usually)
                if batch_step % 5000 == 0:
                     # Log original images (normalize [-1, 1] to [0, 1])
                    writer.add_images("Original Images", (x_batch.float() + 1) / 2, global_step=batch_step, dataformats='NCHW')
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
                generated = self.generate_images_v_prediction(model, num_images=4) # Use num_images=4 for grid
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
    
    @torch.no_grad() # Decorator to disable gradient calculation during inference
    def p_sample_v_prediction(self, model, x_t, t_int):
        """
        Sample from the model at timestep t using v-prediction (PyTorch version).

        Args:
            model (nn.Module): The neural network model (predicts v_theta).
            x_t (torch.Tensor): Current noisy image, shape [B, C, H, W].
            t_int (int): Current timestep as an integer.

        Returns:
            torch.Tensor: Predicted less noisy image at timestep t-1.
        """
        model.eval() # Set model to evaluation mode
        batch_size = x_t.shape[0]
        device = x_t.device
        # Create tensor for timestep t
        t_tensor = torch.full((batch_size,), t_int, dtype=torch.long, device=device)

        # --- Modification for v-prediction starts ---
        # 1. Predict the velocity component v_theta
        predicted_v = model(x_t, t_tensor) # Assuming model output is v_theta directly

        # 2. Convert predicted v_theta to predicted noise epsilon_pred
        # Need sqrt_alphas_cumprod_t and sqrt_one_minus_alphas_cumprod_t
        # Use the _extract helper to get coefficients for the current timestep t_int
        # Note: _extract expects a tensor t, so we use t_tensor
        sqrt_alpha_prod_t = self._extract(self.sqrt_alphas_cumprod, t_tensor, x_t.shape)
        sqrt_one_minus_alpha_prod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t_tensor, x_t.shape)

        # epsilon_pred = sqrt(1 - alpha_bar_t) * x_t + sqrt(alpha_bar_t) * v_theta
        predicted_noise = sqrt_one_minus_alpha_prod_t * x_t + sqrt_alpha_prod_t * predicted_v
        # --- Modification for v-prediction ends ---

        # Get other parameters for timestep t (mostly scalars for DDPM)
        alpha_t = self.alphas[t_int].to(device)
        # alpha_cumprod_t = self.alphas_cumprod[t_int].to(device) # Not directly needed below, but related
        beta_t = self.betas[t_int].to(device)
        # Use scalar coefficient directly from pre-calculated tensor for sqrt(1-alpha_bar_t)
        sqrt_one_minus_alpha_cumprod_scalar_t = self.sqrt_one_minus_alphas_cumprod[t_int].to(device)
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t_int].to(device)

        # 3. Calculate the mean using the *original* DDPM formula, but with epsilon_pred
        # Equation: 1/sqrt(alpha_t) * (x_t - beta_t / sqrt(1 - alpha_cumprod_t) * epsilon_pred)
        coeff = beta_t / sqrt_one_minus_alpha_cumprod_scalar_t # Use scalar version here
        mean_pred = sqrt_recip_alpha_t * (x_t - coeff * predicted_noise) # Use the derived predicted_noise

        # 4. Add noise based on posterior variance (same as before)
        posterior_variance_t = self.posterior_variance[t_int].to(device)

        # Sample noise z ~ N(0, I) only if t > 0
        noise = torch.randn_like(x_t) if t_int > 0 else torch.zeros_like(x_t)

        # Calculate final sample x_{t-1}
        x_t_minus_1 = mean_pred + torch.sqrt(posterior_variance_t) * noise
        x_t_minus_1 = torch.clamp(x_t_minus_1, -1.0, 1.0) # Clamp to valid range

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
    
    @torch.no_grad() # Decorator for inference mode
    def generate_images_v_prediction(self, model, num_images=4):
        """
        Generate images using the v-prediction diffusion model (PyTorch version).

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

        # # Sample step by step, from t=T-1 down to t=0
        for t in tqdm(range(self.timesteps - 1, -1, -1), desc="Generating"):
            x_t = self.p_sample_v_prediction(model, x_t, t)

        # Convert from [-1, 1] range to [0, 1]
        generated_images = (x_t + 1.0) / 2.0
        generated_images = torch.clamp(generated_images, 0.0, 1.0)

        print("Image generation complete.")
        return generated_images

    def save_model(self, model, save_path, optimizer=None, epoch=None, loss=None, save_weights_only=False):
        """
        Save model - either weights only or full checkpoint including training state.
    
        Args:
            model (nn.Module): The trained PyTorch model.
            save_path (str): Path to save the model file (e.g., 'model.pth').
            optimizer (torch.optim.Optimizer, optional): The optimizer to save state.
            epoch (int, optional): Current training epoch.
            loss (float, optional): Current/best loss value.
            save_weights_only (bool, optional): If True, save only weights. Default: False.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        if save_weights_only:
            # Save only model weights (state_dict)
            torch.save(model.state_dict(), save_path)
            print(f"Model weights saved to {save_path}")
        else:
            # Save full checkpoint with training state
            checkpoint = {
                'model_state_dict': model.state_dict(),
            }
            
            # Add optional components if provided
            if optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
            if epoch is not None:
                checkpoint['epoch'] = epoch
                
            if loss is not None:
                checkpoint['loss'] = loss
                
            torch.save(checkpoint, save_path)
            print(f"Full model checkpoint saved to {save_path}")
            
            # Print additional info about what was saved
            components = []
            if 'model_state_dict' in checkpoint: components.append("model weights")
            if 'optimizer_state_dict' in checkpoint: components.append("optimizer state")
            if 'epoch' in checkpoint: components.append(f"epoch {epoch}")
            if 'loss' in checkpoint: components.append(f"loss {loss:.6f}")
            
            print(f"Saved: {', '.join(components)}")

    def load_model_weights(self, model, model_path, verbose=False):
        """
        Load model weights from a checkpoint, supporting different architectures.
    
        Args:
            model (nn.Module): The PyTorch model instance.
            model_path (str): Path to the saved weights file.
            verbose (bool): Whether to print detailed information about missing/unexpected keys.
        """
        if os.path.exists(model_path):
            # Tải toàn bộ checkpoint
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Trích xuất state_dict (tùy cấu trúc của checkpoint)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            
            # Tải với strict=False để bỏ qua các parameters không khớp
            incompatible_keys = model.load_state_dict(state_dict, strict=False)
            
            # In thông tin về các keys đã được load và keys bị bỏ qua
            if incompatible_keys.missing_keys:
                print(f"Warning: {len(incompatible_keys.missing_keys)} keys in model were not loaded")
                if verbose:
                    print(f"Missing keys: {incompatible_keys.missing_keys}")
                    
            if incompatible_keys.unexpected_keys:
                print(f"Info: {len(incompatible_keys.unexpected_keys)} keys in checkpoint were not used")
                if verbose:
                    print(f"Unused keys: {incompatible_keys.unexpected_keys}")
            
            # Đảm bảo model ở đúng device
            # model.to(self.device)
            
            # In thông tin tổng quát
            print(f"Partial weights loaded from {model_path}")
            print(f"Successfully loaded {len(state_dict) - len(incompatible_keys.unexpected_keys)} compatible parameters")
        else:
            print(f"Warning: Model weights path not found: {model_path}")
    
    # Inside the DiffusionModelPyTorch class...

    def load_checkpoint_for_resume(self, model, optimizer, checkpoint_path):
        """
        Load a full checkpoint for resuming training, performing loading
        operations on the CPU to avoid GPU OOM, then moving the model
        back to the target device (self.device).

        Args:
            model (nn.Module): The model instance (should ideally be on CPU initially).
            optimizer (torch.optim.Optimizer): The optimizer instance.
            checkpoint_path (str): Path to the checkpoint file (.pth).

        Returns:
            int: The epoch number to start training from (saved_epoch + 1).
            float: The loss value saved in the checkpoint.
                   Returns float('inf') if checkpoint doesn't exist/fails to load.
        """
        start_epoch = 0
        loaded_loss = float('inf')
        original_device = next(model.parameters()).device # Remember the model's original device (likely CPU if called correctly)

        # Ensure model is on CPU before loading state dicts from a CPU-loaded checkpoint
        model.to('cpu')
        print(f"Temporarily moved model to CPU for loading.")


        # Check if the checkpoint file exists
        if os.path.isfile(checkpoint_path):
            print(f"Loading checkpoint for resume from: {checkpoint_path} onto CPU")
            try:
                # --- Step 1: Load the entire checkpoint dictionary onto CPU ---
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location='cpu', # <<< Load all tensors in the checkpoint to CPU memory
                    weights_only=False   # <<< Keep this False to load optimizer state etc.
                )
                print("Checkpoint dictionary loaded to CPU memory.")

                # --- Step 2: Load model state dict while model is on CPU ---
                model.load_state_dict(checkpoint['model_state_dict'])
                print("Model state loaded successfully onto CPU model.")

                # --- Step 3: Load optimizer state dict ---
                # The optimizer state is associated with model parameters.
                # Since the model is currently on CPU, loading the optimizer state
                # (which was also loaded to CPU by map_location='cpu') works here.
                if 'optimizer_state_dict' in checkpoint and optimizer is not None:
                    try:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        print("Optimizer state loaded successfully (references CPU parameters for now).")
                    except Exception as optim_load_err:
                         print(f"Error loading optimizer state: {optim_load_err}")
                         # Decide if you want to proceed without optimizer state or re-raise
                         print("Warning: Optimizer state loading failed. Optimizer will start from scratch.")
                elif optimizer is None:
                     print("Warning: Optimizer not provided, skipping optimizer state loading.")
                else:
                    print("Warning: Optimizer state not found in checkpoint. Optimizer starts from scratch.")

                # --- Step 4: Load epoch and loss (these are usually small scalars) ---
                if 'epoch' in checkpoint:
                    saved_epoch = checkpoint['epoch']
                    start_epoch = saved_epoch + 1
                    print(f"Resuming from epoch: {start_epoch}")
                else:
                    print("Warning: Epoch number not found in checkpoint. Starting from epoch 0.")

                if 'loss' in checkpoint:
                    loaded_loss = checkpoint['loss']
                    print(f"Loaded loss from checkpoint: {loaded_loss:.6f}")
                else:
                     print("Info: Loss value not found in checkpoint.")

                # --- Step 5: Move the model back to the target device (e.g., CUDA) ---
                # This needs to happen AFTER successfully loading state dicts.
                model.to(self.device)
                print(f"Model moved back to target device: {self.device}")
                # NOTE: The optimizer's state (buffers) will implicitly move with the
                # parameters when the model is moved. No explicit optimizer.to(self.device) is usually needed.

            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Starting training from scratch due to error during loading.")
                # Attempt to move model back to original device even if loading failed
                try:
                     model.to(self.device)
                     print(f"Model moved back to target device '{self.device}' after loading error.")
                except Exception as move_err:
                     print(f"Could not move model back to target device after error: {move_err}")
                start_epoch = 0
                loaded_loss = float('inf')
        else:
            print(f"Checkpoint file not found at {checkpoint_path}. Starting training from scratch.")
            # Ensure model is on the correct device if starting from scratch
            model.to(self.device)
            print(f"Model ensured to be on target device '{self.device}' when starting fresh.")
            start_epoch = 0
            loaded_loss = float('inf')

        return start_epoch, loaded_loss

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
        step_indices = torch.linspace(0, self.timesteps - 1, num_steps_to_show, dtype=torch.long, device='cuda') # Use cuda for indices

        # Forward process
        forward_images = []
        print("Visualizing forward process...")
        for t_int in step_indices.tolist():
            t = torch.tensor([t_int], device=self.device) # Create tensor for timestep
            noisy_image_t, _ = self.q_sample(x_0_batch, t) # Use q_sample
            # Convert [-1, 1] to [0, 1] for display, move to cuda, remove batch dim
            img_display = (noisy_image_t[0].to(self.device) + 1.0) / 2.0
            forward_images.append(img_display.permute(1, 2, 0).clamp(0, 1).cpu().numpy()) # CHW to HWC

        # Reverse process
        reverse_images = []
        # x_t = torch.randn_like(x_0_batch) # Start with noise on the correct device
        x_t = noisy_image_t # Start with the last noisy image from the forward process
        print("Visualizing reverse process...")
        timesteps_for_reverse_vis = step_indices.tolist()[::-1] # Reverse order for display matching
        current_vis_idx = 0

        for t_int in tqdm(range(self.timesteps - 1, -1, -1), desc="Reverse"):
            x_t = self.p_sample_v_prediction(model, x_t, t_int)
            # Store image if the timestep matches one we want to visualize
            if current_vis_idx < len(timesteps_for_reverse_vis) and t_int == timesteps_for_reverse_vis[current_vis_idx]:
                img_display = (x_t[0].to(self.device) + 1.0) / 2.0 # Convert [-1, 1] to [0, 1]
                reverse_images.append(img_display.permute(1, 2, 0).clamp(0, 1).cpu().numpy()) # CHW to HWC
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

# Assume the cosine_beta_schedule function exists somewhere if needed for reference,
# but we will rely on the scheduler's internal implementation.

# Import a suitable scheduler, DDIM is a good start, DPM++ is often faster
from diffusers import DDIMScheduler
class ImageGenerator:
    def __init__(self, img_channels=3, img_size=32, device='cuda', num_train_timesteps=1000):
        """
        Initializes the Image Generator using a diffusers Scheduler.

        Args:
            img_channels (int): Number of image channels.
            img_size (int): Image height/width.
            device (str or torch.device): Device to run on.
            num_train_timesteps (int): The number of diffusion steps used during training (e.g., 1000).
            # Note: We don't need beta_start, beta_end when using named schedules like squaredcos_cap_v2
        """
        self.img_channels = img_channels
        self.img_size = img_size
        self.device = device
        self.num_train_timesteps = num_train_timesteps
        def cosine_beta_schedule(num_train_timesteps, s=0.008, dtype=torch.float32):
            steps = num_train_timesteps + 1
            x = torch.linspace(0, num_train_timesteps, steps, dtype=dtype)
            alphas_cumprod = torch.cos(((x / num_train_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1. - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0., 0.999) # Use PyTorch's clip

        self.betas = cosine_beta_schedule(num_train_timesteps).to(self.device)

        # --- 1. Initialize the Scheduler ---
        # Choose a scheduler. DDIMScheduler is faster than DDPM.
        # DPMSolverMultistepScheduler is often even faster.
        self.scheduler = DDIMScheduler( # Or DPMSolverMultistepScheduler(...)
            num_train_timesteps=self.num_train_timesteps,
            trained_betas=self.betas,
            prediction_type="v_prediction",        # <<< Specify that the model predicts velocity (v)
            clip_sample=False,                 # Often recommended for better generation quality
            set_alpha_to_one=False,
            steps_offset=1,
        )
        # You can set a default number of inference steps here if you like
        # self.scheduler.set_timesteps(50)

        print(f"ImageGenerator initialized with {type(self.scheduler).__name__}, "
              f"cosine schedule (squaredcos_cap_v2), "
              f"and prediction_type='velocity'.")

    @torch.no_grad()
    def generate_images(self, model, num_images=1, num_inference_steps=50):
        """
        Generate images using the diffusion model with the configured scheduler.

        Args:
            model (nn.Module): The neural network model (trained with v-prediction).
            num_images (int): Number of images to generate.
            num_inference_steps (int): Number of steps for the sampler.

        Returns:
            torch.Tensor: Generated images in [0, 1] range, shape [N, C, H, W].
        """
        model.eval() # Set model to evaluation mode
        print(f"Generating {num_images} images using {num_inference_steps} steps on device {self.device}...")

        # Set the number of inference steps for the scheduler
        self.scheduler.set_timesteps(num_inference_steps)

        # Start with pure noise (initial sample x_T)
        latents = torch.randn(
            (num_images, self.img_channels, self.img_size, self.img_size),
            device=self.device
        )
        # Scale initial noise if required by the specific scheduler (DDIM/DDPM usually need this)
        # Check the scheduler's documentation or examples for init_noise_sigma
        # Example scaling (might vary): latents = latents * self.scheduler.init_noise_sigma

        # Sampling Loop using the scheduler
        for t in tqdm(self.scheduler.timesteps, desc="Generating"):
            t = t.unsqueeze(0).to(self.device) # Match batch size
            # Prepare input for the model (e.g., scaling for noise schedule)
            # Some schedulers might require scaling the input latents
            # latent_model_input = self.scheduler.scale_model_input(latents, t) # If needed
            latent_model_input = latents # Often sufficient

            # Call Model to predict velocity v_theta
            # The model was trained to output v, so its output is v_theta
            model_output_v = model(latent_model_input, t)

            # Call Scheduler Step
            # Pass the model's velocity prediction (model_output_v)
            # The scheduler internally handles the conversion (v -> epsilon or v -> x0)
            # based on its prediction_type and the sampling algorithm (DDIM etc.)
            latents = self.scheduler.step(model_output_v, t, latents).prev_sample

        # Post-processing: Convert from [-1, 1] range to [0, 1]
        generated_images = (latents + 1.0) / 2.0
        generated_images = torch.clamp(generated_images, 0.0, 1.0)

        print("Image generation complete.")
        return generated_images