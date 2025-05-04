from diffusion import DiTWrapper, CausalDitWrapper
import torch.nn.functional as F
from typing import Tuple
from torch import nn
import torch


class ODERegression(nn.Module):
    def __init__(self, args, device):
        """
        Initialize the ODERegression module.
        This class is self-contained and compute generator losses
        in the forward pass given precomputed ode solution pairs.
        This class supports the ode regression loss for both causal and bidirectional models.
        See Sec 4.3 of CausVid https://arxiv.org/abs/2412.07772 for details
        """
        super().__init__()

        # Step 1: Initialize all models

        self.generator = CausalDiTWrapper(**config["model"]['diffusion'], timestep_embed_dim=24)

        if getattr(args, "generator_ckpt", False):
            print(f"Loading pretrained generator from {args.generator_ckpt}")
            state_dict = torch.load(args.generator_ckpt, map_location="cpu")[
                'generator']
            self.generator.load_state_dict(
                state_dict, strict=True
            )

        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()

        self.text_encoder = get_text_encoder_wrapper(
            model_name=args.model_name)()
        self.text_encoder.requires_grad_(False)

        # Step 2: Initialize all hyperparameters

        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long, device=device)

        self.args = args
        self.device = device
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32

        # for latent frame with zero noise, we probablistically perturb it with an extra small noise
        # self.extra_noise_step = getattr(args, "extra_noise_step", 0)
        # self.scheduler = self.generator.get_scheduler()

    def _process_timestep(self, timestep):
        """
        Pre-process the randomly generated timestep based on the generator's task type (always audio).
        Input:
            - timestep: [batch_size, num_frame] tensor containing the randomly generated timestep.

        Output Behavior:
            - causal_audio: broadcast the timestep to be the same for all frames **in a block**.
        """
        timestep = timestep.reshape(timestep.shape[0], -1, self.num_frame_per_block)
        timestep[:, :, 1:] = timestep[:, :, 0:1]
        timestep = timestep.reshape(timestep.shape[0], -1)
        return timestep

    @torch.no_grad()
    def _prepare_generator_input(self, ode_latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a tensor containing the whole ODE sampling trajectories,
        randomly choose an intermediate timestep and return the latent as well as the corresponding timestep.
        Input:
            - ode_latent: a tensor containing the whole ODE sampling trajectories [batch_size, num_denoising_steps, num_frames, latent_dim].
        Output:
            - noisy_input: a tensor containing the selected latent [batch_size, num_frames, latent_dim].
            - timestep: a tensor containing the corresponding timestep [batch_size].
        """
        batch_size, num_denoising_steps, num_frames, latent_dim = ode_latent.shape

        # Step 1: Randomly choose a timestep for each frame
        index = torch.randint(0, len(self.denoising_step_list), [
            batch_size, num_frames], device=self.device, dtype=torch.long)

        index = self._process_timestep(index)

        # take the timestep collected per frame and filter ode_latent to include only 
        # the selected timesteps per frame

        noisy_input = torch.gather(
            ode_latent, dim=1,
            index=index.reshape(batch_size, 1, num_frames, 1).expand(
                -1, -1, -1, latent_dim)
        ).squeeze(1)

        timestep = self.denoising_step_list[index]

        return noisy_input, timestep

    def generator_loss(self, ode_latent: torch.Tensor, conditional_dict: dict) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noisy latents and compute the ODE regression loss.
        Input:
            - ode_latent: a tensor containing the ODE latents [batch_size, num_denoising_steps, num_frames, latent_dim].
            They are ordered from most noisy to clean latents.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - loss: a scalar tensor representing the generator loss.
            - log_dict: a dictionary containing additional information for loss timestep breakdown.
        """
        # Step 1: Run generator on noisy latents
        target_latent = ode_latent[:, -1]

        noisy_input, timestep = self._prepare_generator_input(
            ode_latent=ode_latent)

        output = self.generator(x=noisy_input, t=timestep, cond=conditional_dict)

        # Step 2: Compute the regression loss
        mask = timestep != 0

        loss = F.mse_loss(
            output[mask], target_latent[mask], reduction="mean")

        log_dict = {
            "unnormalized_loss": F.mse_loss(pred_audio, target_latent, reduction='none').mean(dim=[1, 2]).detach(),
            "timestep": timestep.float().mean(dim=1).detach()
        }

        return loss, log_dict
