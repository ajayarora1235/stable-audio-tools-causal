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

class ODERegressionWrapper(nn.Module):
    def __init__(
        self,
        model: ConditionedDiffusionModel,
        text_encoder: TextEncoder,
        vae: AutoencoderPretransform,
        num_timesteps: int = 4
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
        
    def forward(self, batch):
        # Get text embeddings
        text_embeddings = self.text_encoder(batch["text"])
        
        # Get latents
        latents = self.vae.encode(batch["audio"])
        
        # Get timesteps
        timesteps = torch.linspace(0, 1, self.num_timesteps, device=latents.device)
        
        # Get noisy latents and scores
        noise = torch.randn_like(latents)
        noisy_latents = self.diffusion_wrapper.q_sample(latents, timesteps, noise)
        scores = self.diffusion_wrapper.get_score(noisy_latents, timesteps, text_embeddings)
        
        return {
            "noisy_latents": noisy_latents,
            "scores": scores,
            "timesteps": timesteps
        }

class ODERegressionDemoCallback(DemoCallback):
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
        
        # Get timesteps
        timesteps = torch.linspace(0, 1, 4, device=latents.device)
        
        # Get noisy latents and scores
        noise = torch.randn_like(latents)
        noisy_latents = self.diffusion_wrapper.q_sample(latents, timesteps, noise)
        scores = self.diffusion_wrapper.get_score(noisy_latents, timesteps, text_embeddings)
        
        # Decode latents
        audio = self.vae.decode(latents)
        noisy_audio = self.vae.decode(noisy_latents)
        
        return {
            "audio": audio,
            "noisy_audio": noisy_audio,
            "scores": scores,
            "timesteps": timesteps,
            "text": batch["text"]
        } 