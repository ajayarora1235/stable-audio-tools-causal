from safetensors.torch import save_file
import torch.distributed as dist
from omegaconf import OmegaConf
from causal_audio.ode_regression import ODERegression
import argparse
import torch
import wandb
import time
import os
from causvid.data import ODERegressionLMDBDataset
from causvid.util import barrier, cycle
import torch.nn.functional as F


class ODERegressionTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.cuda.current_device()
        self.is_main_process = dist.get_rank() == 0
        
        # Initialize the model
        self.ode_model = ODERegression(config, device=self.device)

        # Initialize optimizer
        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.distillation_model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2)
        )

        # Initialize dataset and dataloader
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
        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.validation_timesteps = getattr(config, "validation_timesteps", [0.1, 0.5, 0.9])
        self.validation_step_outputs = {f'val/loss_{t:.1f}': [] for t in self.validation_timesteps}

    def save(self, use_safetensors=False):
        print("Start gathering distributed model states...")
        generator_state_dict = self.ode_model.generator.state_dict()
        state_dict = {
            "generator": generator_state_dict
        }

        if self.is_main_process:
            save_path = os.path.join(
                self.config.output_dir, f"ode_model_{self.step}")
            
            if use_safetensors:
                save_file(state_dict, save_path + ".safetensors")
                print(f"Model saved to {save_path}.safetensors")
            else:
                torch.save(state_dict, save_path + ".pt")
                print(f"Model saved to {save_path}.pt")

    def train_one_step(self):
        # Step 1: Get the next batch of text prompts
        batch = next(self.dataloader)
        text_prompts = batch["prompts"]
        ode_latent = batch["ode_latent"].to(
            device=self.device, dtype=self.dtype)

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.ode_model.generator.conditioner(
                text_prompts=text_prompts)

        # Step 3: Train the generator
        generator_loss, log_dict = self.ode_model.generator_loss(
            ode_latent=ode_latent,
            conditional_dict=conditional_dict
        )

        self.generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_grad_norm = self.ode_model.generator.clip_grad_norm_(
            self.max_grad_norm)
        self.generator_optimizer.step()

        # Step 4: Logging
        if self.is_main_process:
            log_dict = {
                "generator/loss": generator_loss.item(),
                "generator/grad_norm": generator_grad_norm
            }
            
            # Add per-timestep loss breakdowns
            if "unnormalized_loss" in log_dict and "timestep" in log_dict:
                unnormalized_loss = log_dict["unnormalized_loss"]
                timestep = log_dict["timestep"]
                
                loss_breakdown = {}
                for index, t in enumerate(timestep):
                    t_bucket = str(int(t.item()) // 250 * 250)
                    if t_bucket not in loss_breakdown:
                        loss_breakdown[t_bucket] = []
                    loss_breakdown[t_bucket].append(unnormalized_loss[index].item())
                
                for key_t, losses in loss_breakdown.items():
                    log_dict[f"generator/loss_at_time_{key_t}"] = sum(losses) / len(losses)
            
            wandb.log(log_dict, step=self.step)

    def validation_step(self):
        batch = next(self.dataloader)
        text_prompts = batch["prompts"]
        ode_latent = batch["ode_latent"].to(device=self.device, dtype=self.dtype)

        with torch.no_grad():
            conditional_dict = self.ode_model.generator.conditioner(
                text_prompts=text_prompts)
            
            # Use the clean latent as the target
            target_latent = ode_latent[:, -1]
            
            for validation_timestep in self.validation_timesteps:
                # Create a batch with the same timestep for all examples
                batch_size = ode_latent.shape[0]
                timestep = torch.full((batch_size,), validation_timestep, device=self.device)
                
                # Get the corresponding noisy latent
                step_index = int(validation_timestep * len(self.ode_model.denoising_step_list))
                noisy_input = ode_latent[:, step_index]
                
                # Run the model
                output = self.ode_model.generator(
                    x=noisy_input, 
                    t=timestep, 
                    cond=conditional_dict
                )
                
                # Compute loss
                val_loss = F.mse_loss(output, target_latent)
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

    trainer = ODERegressionTrainer(config)
    trainer.train()

    wandb.finish()

if __name__ == "__main__":
    main()
