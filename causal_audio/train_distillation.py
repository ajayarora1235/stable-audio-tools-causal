from safetensors.torch import save_file
import torch.distributed as dist
from omegaconf import OmegaConf
from causal_audio.dmd import DMD
import argparse
import torch
import wandb
import time
import os
import json
from causal_video.data import ODERegressionLMDBDataset
from causal_video.util import barrier, cycle
import torch.nn.functional as F
from stable_audio_tools.inference.sampling import get_alphas_sigmas, sample, sample_discrete_euler, truncated_logistic_normal_rescaled, DistributionShift
from accelerate import DistributedDataParallelKwargs, Accelerator

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(log_with="wandb", kwargs_handlers=[ddp_kwargs])
device = accelerator.device

class DiffusionDistillationTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.cuda.current_device()
        self.is_main_process = dist.get_rank() == 0
        self.diffusion_objective = getattr(config, "diffusion_objective", "v")
        self.validation_timesteps = getattr(config, "validation_timesteps", [0.1, 0.5, 0.9])

        if config.distillation_loss == "dmd":
            self.distillation_model = DMD(config, device=self.device)
        else:
            raise ValueError("Invalid distillation loss type")

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.distillation_model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2)
        )

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.distillation_model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2)
        )

        dataset = ODERegressionLMDBDataset(
                config.data_path, max_pair=int(1e8))
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, sampler=sampler)
        self.dataloader = cycle(dataloader)

        self.step = 0
        self.max_grad_norm = 10.0
        self.previous_time = None
        self.backward_simulation = getattr(config, "backward_simulation", False)
        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.validation_step_outputs = {f'val/loss_{t:.1f}': [] for t in self.validation_timesteps}
        self.unconditional_dict = None

    def save(self, use_safetensors=False):
        print("Start gathering distributed model states...")
        generator_state_dict = self.distillation_model.generator.state_dict()
        critic_state_dict = self.distillation_model.fake_score.state_dict()
        state_dict = {
            "generator": generator_state_dict,
            "critic": critic_state_dict
        }

        if self.is_main_process:
            save_path = os.path.join(
                self.config.output_dir, f"distillation_model_{self.step}")
            
            if use_safetensors:
                save_file(state_dict, save_path + ".safetensors")
                print(f"Model saved to {save_path}.safetensors")
            else:
                torch.save(state_dict, save_path + ".pt")
                print(f"Model saved to {save_path}.pt")

    def train_one_step(self):
        # Step 1: Get the next batch of text prompts
        if not self.backward_simulation:
            batch = next(self.dataloader)
            text_prompts = batch["prompts"]
            clean_latent = batch["ode_latent"][:, -1].to(
                device=self.device, dtype=self.dtype)
        else:
            text_prompts = next(self.dataloader)
            clean_latent = None

        batch_size = len(text_prompts)
        audio_shape = list(self.config.audio_shape)
        audio_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            # Use the model's built-in conditioner to handle all conditioning
            conditional_dict = self.distillation_model.generator.conditioner(text_prompts, self.device)
            
            # For unconditional generation (used in classifier-free guidance)
            if not getattr(self, "unconditional_dict", None):
                # Create a copy of metadata with empty/null prompts
                unconditional_prompts = [""] * batch_size
                
                # Get unconditional embeddings using the same conditioner
                unconditional_dict = self.distillation_model.generator.conditioner(unconditional_prompts, self.device)
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 3: Train the generator
        TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0
        
        if TRAIN_GENERATOR:
            generator_loss, generator_log_dict = self.distillation_model.generator_loss(
                audio_shape=audio_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent
            )

            self.generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_grad_norm = self.distillation_model.generator.clip_grad_norm_(
                self.max_grad_norm)
            self.generator_optimizer.step()
            
            # Log generator metrics
            if self.is_main_process:
                log_dict = {
                    "generator/loss": generator_loss.item(),
                    "generator/grad_norm": generator_grad_norm
                }
                for key, value in generator_log_dict.items():
                    log_dict[f"generator/{key}"] = value
                
                wandb.log(log_dict, step=self.step)
        
        # Step 4: Train the critic
        critic_loss, critic_log_dict = self.distillation_model.critic_loss(
            audio_shape=audio_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = self.distillation_model.fake_score.clip_grad_norm_(
            self.max_grad_norm)
        self.critic_optimizer.step()
        
        # Log critic metrics
        if self.is_main_process:
            log_dict = {
                "critic/loss": critic_loss.item(),
                "critic/grad_norm": critic_grad_norm
            }
            for key, value in critic_log_dict.items():
                log_dict[f"critic/{key}"] = value
            
            wandb.log(log_dict, step=self.step)

    def validation_step(self):
        batch = next(self.dataloader)
        
        if not self.backward_simulation:
            text_prompts = batch["prompts"]
            diffusion_input = batch["ode_latent"][:, -1].to(device=self.device, dtype=self.dtype)
        else:
            text_prompts = batch
            diffusion_input = torch.randn(len(text_prompts), *self.config.audio_shape[1:], device=self.device, dtype=self.dtype)

        with torch.no_grad():
            conditioning = self.distillation_model.generator.conditioner(text_prompts, self.device)

        for validation_timestep in self.validation_timesteps:
            t = torch.full((diffusion_input.shape[0],), validation_timestep, device=self.device)

            # Calculate the noise schedule parameters for those timesteps
            if self.diffusion_objective in ["v"]:
                alphas, sigmas = get_alphas_sigmas(t)
            elif self.diffusion_objective == "rectified_flow":
                alphas, sigmas = 1-t, t

            # Combine the ground truth data and the noise
            alphas = alphas[:, None, None]
            sigmas = sigmas[:, None, None]
            noise = torch.randn_like(diffusion_input)
            noised_inputs = diffusion_input * alphas + noise * sigmas

            if self.diffusion_objective == "v":
                targets = noise * alphas - diffusion_input * sigmas
            elif self.diffusion_objective == "rectified_flow":
                targets = noise - diffusion_input

            extra_args = {}

            with torch.no_grad():
                output = self.distillation_model.generator(noised_inputs, t, cond=conditioning, cfg_dropout_prob=0, **extra_args)
                val_loss = F.mse_loss(output, targets)
                self.validation_step_outputs[f'val/loss_{validation_timestep:.1f}'].append(val_loss.item())


    def on_validation_epoch_start(self):
        # Reset validation losses
        for validation_timestep in self.validation_timesteps:
            self.validation_step_outputs[f'val/loss_{validation_timestep:.1f}'] = []

    def on_validation_epoch_end(self):
        if not self.is_main_process:
            return
            
        log_dict = {}
        for validation_timestep in self.validation_timesteps:
            outputs_key = f'val/loss_{validation_timestep:.1f}'
            if len(self.validation_step_outputs[outputs_key]) > 0:
                val_loss = sum(self.validation_step_outputs[outputs_key]) / len(self.validation_step_outputs[outputs_key])
                log_dict[outputs_key] = val_loss

        # Get average over all timesteps
        all_losses = []
        for losses in self.validation_step_outputs.values():
            all_losses.extend(losses)
        
        if all_losses:
            avg_loss = sum(all_losses) / len(all_losses)
            log_dict['val/avg_loss'] = avg_loss
        
        wandb.log(log_dict, step=self.step)

    def train(self):
        while True:
            self.train_one_step()
            
            # Run validation at specified intervals
            if self.step % self.config.validation_interval == 0:
                self.on_validation_epoch_start()
                self.validation_step()
                self.on_validation_epoch_end()
                
            if (not self.config.no_save) and self.step % self.config.log_iters == 0:
                self.save()
                torch.cuda.empty_cache()

            barrier()
            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time

            self.step += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize

    trainer = DiffusionDistillationTrainer(config)
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()
