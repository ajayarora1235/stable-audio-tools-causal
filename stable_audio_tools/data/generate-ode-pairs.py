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

def generate_ode_pairs(
    model_path: str,
    text_encoder_path: str,
    vae_path: str,
    dataset_path: str,
    output_dir: str,
    num_samples: int = 1000,
    batch_size: int = 8,
    device: str = "cuda"
):
    # Initialize models
    model = ConditionedDiffusionModel.load_from_checkpoint(model_path)
    text_encoder = TextEncoder.load_from_checkpoint(text_encoder_path)
    vae = AutoencoderPretransform.load_from_checkpoint(vae_path)
    
    # Move to device
    model = model.to(device)
    text_encoder = text_encoder.to(device)
    vae = vae.to(device)
    
    # Create wrapper
    diffusion_wrapper = ConditionalDiffusionWrapper(
        model=model,
        text_encoder=text_encoder,
        vae=vae
    )
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset = torch.load(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Generate pairs
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        if batch_idx * batch_size >= num_samples:
            break
            
        # Get text embeddings
        text_embeddings = text_encoder(batch["text"])
        
        # Get latents
        latents = vae.encode(batch["audio"])
        
        # Get diffusion steps
        timesteps = torch.linspace(0, 1, 4, device=device)
        
        # Generate pairs
        for t in timesteps:
            # Get noisy latents
            noise = torch.randn_like(latents)
            noisy_latents = diffusion_wrapper.q_sample(latents, t, noise)
            
            # Get score
            score = diffusion_wrapper.get_score(noisy_latents, t, text_embeddings)
            
            # Save pairs
            for i in range(batch_size):
                sample_idx = batch_idx * batch_size + i
                if sample_idx >= num_samples:
                    break
                    
                pair = {
                    "noisy_latent": noisy_latents[i].cpu(),
                    "score": score[i].cpu(),
                    "t": t.cpu(),
                    "text": batch["text"][i]
                }
                
                torch.save(pair, output_dir / f"pair_{sample_idx}.pt")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--text_encoder_path", type=str, required=True)
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    generate_ode_pairs(
        model_path=args.model_path,
        text_encoder_path=args.text_encoder_path,
        vae_path=args.vae_path,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=args.device
    ) 