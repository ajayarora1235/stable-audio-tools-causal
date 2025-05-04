import torch.nn.functional as F
from typing import Tuple
from torch import nn
import torch
from stable_audio_tools.inference.sampling import get_alphas_sigmas, sample, sample_discrete_euler, truncated_logistic_normal_rescaled, DistributionShift


class DMD(nn.Module):
    def __init__(self, args, device):
        with open('/Users/cameronfranz/Documents/Projects/AudioTrain/causal_audio/configs/txt2audio/stable_audio_open_1_0.json') as f:
            config = json.load(f)

        self.generator = DiffusionTransformer(**config["model"]['diffusion'], timestep_embed_dim=24)
        # self.generator = get_diffusion_wrapper(
            # model_name=self.generator_model_name)()
        
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.real_score = get_diffusion_wrapper(
            model_name=self.real_model_name)()
        self.real_score.requires_grad_(False)

        self.fake_score = get_diffusion_wrapper(
            model_name=self.fake_model_name)()

        self.text_encoder = get_text_encoder_wrapper(
            model_name=args.model_name)()
        self.text_encoder.requires_grad_(False)

        self.vae = get_vae_wrapper(model_name=args.model_name)()
        self.vae.requires_grad_(False)

        # Step 2: Initialize all dmd hyperparameters
        self.timestep_sampler = getattr(args, "timestep_sampler", "uniform")
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long, device=device)
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        self.real_guidance_scale = args.real_guidance_scale
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)

        self.args = args
        self.device = device
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32
        self.scheduler = self.generator.get_scheduler()
        self.denoising_loss_func = MSELoss("output", "targets", weight=1.0, mask_key="padding_mask" if self.mask_padding else None, name="mse_loss")

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

    def _compute_kl_grad(
        self, noisy_audio: torch.Tensor,
        estimated_clean_audio: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict, unconditional_dict: dict,
        normalization: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the KL grad (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - noisy_image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - estimated_clean_image_or_video: a tensor with shape [B, F, C, H, W] representing the estimated clean image or video.
            - timestep: a tensor with shape [B, F] containing the randomly generated timestep.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - normalization: a boolean indicating whether to normalize the gradient.
        Output:
            - kl_grad: a tensor representing the KL grad.
            - kl_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Step 1: Compute the fake score
        pred_fake_audio = self.fake_score(
            noisy_audio=noisy_audio,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        # Step 2: Compute the real score
        # We compute the conditional and unconditional prediction
        # and add them together to achieve cfg (https://arxiv.org/abs/2207.12598)
        pred_real_audio = self.real_score(
            x=noisy_audio,
            cond=conditional_dict,
            t=timestep,
            cfg_scale=self.real_guidance_scale
        )

        # Step 3: Compute the DMD gradient (DMD paper eq. 7).
        grad = (pred_fake_audio - pred_real_audio)

        # TODO: Change the normalizer for causal teacher
        if normalization:
            # Step 4: Gradient normalization (DMD paper eq. 8).
            p_real = (estimated_clean_audio - pred_real_audio)
            normalizer = torch.abs(p_real).mean(dim=[1, 2], keepdim=True)
            grad = grad / normalizer
        grad = torch.nan_to_num(grad)

        return grad, {
            "dmdtrain_clean_latent": estimated_clean_audio.detach(),
            "dmdtrain_noisy_latent": noisy_audio.detach(),
            "dmdtrain_pred_real_audio": pred_real_audio.detach(),
            "dmdtrain_pred_fake_audio": pred_fake_audio.detach(),
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "timestep": timestep.detach()
        }

    def compute_distribution_matching_loss(
        self, audio: torch.Tensor, conditional_dict: dict,
        unconditional_dict: dict, gradient_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the DMD loss (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - audio: a tensor with shape [B, F, L] where L is the latent dimension.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - gradient_mask: a boolean tensor with the same shape as audio indicating which frames to compute loss .
        Output:
            - dmd_loss: a scalar tensor representing the DMD loss.
            - dmd_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        original_latent = audio

        batch_size, num_frame = audio.shape[:2]

        with torch.no_grad():
            # Step 1: Randomly sample timestep based on the given schedule
            if self.timestep_sampler == "uniform":
                # Draw uniformly distributed continuous timesteps
                t = torch.rand(batch_size, num_frame, device=self.device)
            elif self.timestep_sampler == "logit_normal":
                t = torch.sigmoid(torch.randn(batch_size, num_frame, device=self.device))
            elif self.timestep_sampler == "trunc_logit_normal":
                t = truncated_logistic_normal_rescaled(batch_size * num_frame).to(self.device)
                t = 1 - t
                t = t.reshape(batch_size, num_frame)
            else:
                raise ValueError(f"Invalid timestep_sampler: {self.timestep_sampler}")

            # Process timestep for causal audio
            t = self._process_timestep(t)

            # Clamp timesteps to min/max range
            t = t.clamp(self.min_step / self.num_train_timestep, self.max_step / self.num_train_timestep)

            # Calculate noise schedule parameters
            alphas, sigmas = get_alphas_sigmas(t)

            # Add noise to the input
            alphas = alphas[:, :, None]  # Add channel dimension
            sigmas = sigmas[:, :, None]  # Add channel dimension
            noise = torch.randn_like(audio)
            noisy_latent = audio * alphas + noise * sigmas

            # Step 2: Compute the KL grad
            grad, dmd_log_dict = self._compute_kl_grad(
                noisy_audio=noisy_latent,
                estimated_clean_audio=original_latent,
                timestep=t * self.num_train_timestep,  # Scale back to discrete timesteps
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict
            )

        if gradient_mask is not None:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            )[gradient_mask], (original_latent.double() - grad.double()).detach()[gradient_mask], reduction="mean")
        else:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            ), (original_latent.double() - grad.double()).detach(), reduction="mean")
        return dmd_loss, dmd_log_dict


    def _run_generator(self, audio_shape, conditional_dict: dict, unconditional_dict: dict, clean_latent: torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optionally simulate the generator's input from noise using backward simulation
        and then run the generator for one-step.
        Input:
            - audio_shape: a list containing the shape of the audio [B, F, L].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, L]. Need to be passed when no backward simulation is used.
        Output:
            - pred_audio: a tensor with shape [B, F, L].
        """
        # Step 1: Sample noise and backward simulate the generator's input
        simulated_noisy_input = []
        for timestep in self.denoising_step_list:
            noise = torch.randn(
                audio_shape, device=self.device, dtype=self.dtype)

            noisy_timestep = timestep * torch.ones(
                audio_shape[:2], device=self.device, dtype=torch.long)
            if timestep != 0:
                # Calculate noise schedule parameters for this timestep
                t = timestep / self.num_train_timestep
                alphas, sigmas = get_alphas_sigmas(torch.tensor([t], device=self.device).expand(audio_shape[0], audio_shape[1]))
                
                # Add channel dimension
                alphas = alphas[:, :, None]
                sigmas = sigmas[:, :, None]
                
                # Add noise to the clean latent
                noisy_audio = clean_latent * alphas + noise * sigmas
            else:
                noisy_audio = clean_latent

            simulated_noisy_input.append(noisy_audio)

        simulated_noisy_input = torch.stack(simulated_noisy_input, dim=1)

        # Step 2: Randomly sample a timestep and pick the corresponding input
        index = torch.randint(0, len(self.denoising_step_list), [
                              audio_shape[0], audio_shape[1]], device=self.device, dtype=torch.long)
        index = self._process_timestep(index)

        # select the corresponding timestep's noisy input from the stacked tensor [B, T, F, L]
        noisy_input = torch.gather(
            simulated_noisy_input, dim=1,
            index=index.reshape(index.shape[0], 1, index.shape[1], 1).expand(
                -1, -1, -1, audio_shape[2])
        ).squeeze(1)

        timestep = self.denoising_step_list[index]

        pred_audio = self.generator(
            noisy_audio=noisy_input,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        gradient_mask = None  # timestep != 0

        # pred_image_or_video = noisy_input * \
        #     (1-gradient_mask.float()).reshape(*gradient_mask.shape, 1, 1, 1) + \
        #     pred_image_or_video * gradient_mask.float().reshape(*gradient_mask.shape, 1, 1, 1)

        pred_audio = pred_audio.type_as(noisy_input)

        return pred_audio, gradient_mask

    def generator_loss(self, audio_shape, conditional_dict: dict, unconditional_dict: dict, clean_latent: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Generate audio from noise and compute the DMD loss.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - audio_shape: a list containing the shape of the audio [B, F, L].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, L]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - generator_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Step 1: Run generator on backward simulated noisy input
        pred_audio, gradient_mask = self._run_generator(
            audio_shape=audio_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent
        )

        # Step 2: Compute the DMD loss
        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            audio=pred_audio,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask
        )

        return dmd_loss, dmd_log_dict

    def critic_loss(self, audio_shape, conditional_dict: dict, unconditional_dict: dict, clean_latent: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Generate audio from noise and train the critic with generated samples.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - audio_shape: a list containing the shape of the audio [B, F, L].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, L]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - critic_log_dict: a dictionary containing the intermediate tensors for logging.
        """

        # Step 1: Run generator on backward simulated noisy input
        with torch.no_grad():
            generated_audio, _ = self._run_generator(
                audio_shape=audio_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent
            )

        # Step 2: Compute the fake prediction
        critic_timestep = torch.randint(
            0,
            self.num_train_timestep,
            audio_shape[:2],
            device=self.device,
            dtype=torch.long
        )
        critic_timestep = self._process_timestep(
            critic_timestep, type=self.fake_task_type)

        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

        # Sample noise and add it to the generated audio
        critic_noise = torch.randn_like(generated_audio)
        
        # Get alpha_t and sigma_t from the noise schedule
        t = critic_timestep / self.num_train_timestep
        alpha_t, sigma_t = get_alphas_sigmas(t)
        alpha_t = alpha_t[:, :, None]  # Add channel dimension
        sigma_t = sigma_t[:, :, None]  # Add channel dimension
        
        # Add noise to the generated audio using alpha_t and sigma_t
        noisy_generated_audio = alpha_t * generated_audio + sigma_t * critic_noise
        
        # Pass the noisy audio to the fake score model
        pred_fake_audio = self.fake_score(
            noisy_audio=noisy_generated_audio,
            conditional_dict=conditional_dict,
            timestep=critic_timestep
        )

        # Step 3: Compute the denoising loss for the fake critic

        if self.diffusion_objective == "v":
            targets = critic_noise * alpha_t - generated_audio * sigma_t

        denoising_loss = self.denoising_loss_func({
            "output": pred_fake_audio,
            "targets": targets,
            "padding_mask": None
        })

        # Step 4: TODO: Compute the GAN loss

        # Step 5: Debugging Log
        critic_log_dict = {
            "critictrain_latent": generated_audio.detach(),
            "critictrain_noisy_latent": noisy_generated_audio.detach(),
            "critictrain_pred_audio": pred_fake_audio.detach(),
            "critic_timestep": critic_timestep.detach()
        }

        return denoising_loss, critic_log_dict