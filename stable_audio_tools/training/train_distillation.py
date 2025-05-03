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
from .dmd import DMDWrapper, DMDDemoCallback
from .trainer import Trainer

def train_distillation(
    model_path: str,
    text_encoder_path: str,
    vae_path: str,
    train_dataset_path: str,
    val_dataset_path: str,
    output_dir: str,
    num_epochs: int = 100,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
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
    wrapper = DMDWrapper(
        model=model,
        text_encoder=text_encoder,
        vae=vae
    )
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    train_dataset = torch.load(train_dataset_path)
    val_dataset = torch.load(val_dataset_path)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create demo callback
    demo_callback = DMDDemoCallback(
        model=model,
        text_encoder=text_encoder,
        vae=vae,
        demo_dir=str(output_dir / "demos")
    )
    
    # Create trainer
    trainer = Trainer(
        model=wrapper,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        output_dir=output_dir,
        learning_rate=learning_rate,
        callbacks=[demo_callback]
    )
    
    # Train
    trainer.train(num_epochs=num_epochs)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--text_encoder_path", type=str, required=True)
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--train_dataset_path", type=str, required=True)
    parser.add_argument("--val_dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    train_distillation(
        model_path=args.model_path,
        text_encoder_path=args.text_encoder_path,
        vae_path=args.vae_path,
        train_dataset_path=args.train_dataset_path,
        val_dataset_path=args.val_dataset_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device
    ) 