import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from pathlib import Path

from ..models.diffusion import ConditionedDiffusionModel
from ..models.autoencoders import AutoencoderPretransform
from ..interface.encoders import TextEncoder
from ..interface.wrappers import ConditionalDiffusionWrapper
from ..training.callbacks import DemoCallback

class NoiseScheduler:
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = "cuda"
    ):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        
        # Create noise schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def get_noise(self, x, t):
        noise = torch.randn_like(x)
        alpha_t = self.alphas_cumprod[t]
        return torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise

class DMDWrapper(nn.Module):
    def __init__(
        self,
        model: ConditionedDiffusionModel,
        text_encoder: TextEncoder,
        vae: AutoencoderPretransform,
        num_timesteps: int = 1000
    ):
        super().__init__()
        
        self.model = model
        self.text_encoder = text_encoder
        self.vae = vae
        self.num_timesteps = num_timesteps
        
        # Create wrapper
        self.diffusion_wrapper = ConditionalDiffusionWrapper(
            model=model,
            text_encoder=text_encoder,
            vae=vae
        )
        
        # Create noise scheduler
        self.noise_scheduler = NoiseScheduler(num_timesteps=num_timesteps)
        
    def forward(self, batch):
        # Get text embeddings
        text_embeddings = self.text_encoder(batch["text"])
        
        # Get latents
        latents = self.vae.encode(batch["audio"])
        
        # Sample timesteps
        t = torch.randint(0, self.num_timesteps, (latents.shape[0],), device=latents.device)
        
        # Get noisy latents
        noisy_latents = self.noise_scheduler.get_noise(latents, t)
        
        # Get real score
        real_score = self.diffusion_wrapper.get_score(noisy_latents, t, text_embeddings)
        
        # Get fake score
        fake_score = self.model(noisy_latents, t, text_embeddings)
        
        return {
            "real_score": real_score,
            "fake_score": fake_score,
            "t": t
        }

class DMDDemoCallback(DemoCallback):
    def __init__(
        self,
        model: ConditionedDiffusionModel,
        text_encoder: TextEncoder,
        vae: AutoencoderPretransform,
        demo_dir: str,
        demo_freq: int = 1000,
        num_demos: int = 4
    ):
        super().__init__(
            model=model,
            text_encoder=text_encoder,
            vae=vae,
            demo_dir=demo_dir,
            demo_freq=demo_freq,
            num_demos=num_demos
        )
        
    def generate_demo(self, batch):
        # Get text embeddings
        text_embeddings = self.text_encoder(batch["text"])
        
        # Get latents
        latents = self.vae.encode(batch["audio"])
        
        # Sample timesteps
        t = torch.randint(0, self.num_timesteps, (latents.shape[0],), device=latents.device)
        
        # Get noisy latents
        noisy_latents = self.noise_scheduler.get_noise(latents, t)
        
        # Get real score
        real_score = self.diffusion_wrapper.get_score(noisy_latents, t, text_embeddings)
        
        # Get fake score
        fake_score = self.model(noisy_latents, t, text_embeddings)
        
        # Decode latents
        audio = self.vae.decode(latents)
        noisy_audio = self.vae.decode(noisy_latents)
        
        return {
            "audio": audio,
            "noisy_audio": noisy_audio,
            "real_score": real_score,
            "fake_score": fake_score,
            "t": t,
            "text": batch["text"]
        } 