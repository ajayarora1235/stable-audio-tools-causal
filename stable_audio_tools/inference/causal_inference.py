import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

from ..models.causal_model import CausalTransformer
from ..models.autoencoders import AutoencoderPretransform
from ..interface.encoders import TextEncoder
from ..interface.wrappers import ConditionalDiffusionWrapper

class CausalInference:
    def __init__(
        self,
        model: CausalTransformer,
        text_encoder: TextEncoder,
        vae: AutoencoderPretransform,
        num_frames_per_block: int = 4,
        device: str = "cuda"
    ):
        self.model = model
        self.text_encoder = text_encoder
        self.vae = vae
        self.num_frames_per_block = num_frames_per_block
        self.device = device
        
        # Move to device
        self.model = self.model.to(device)
        self.text_encoder = self.text_encoder.to(device)
        self.vae = self.vae.to(device)
        
        # Initialize KV cache
        self.kv_cache = None
        self.crossattn_cache = None
        
    def _prepare_blockwise_causal_attn_mask(self, i, j, device):
        mask = torch.ones((i, j), device=device, dtype=torch.bool)
        mask = torch.tril(mask)
        return mask
        
    def _process_block(self, x, text_embeddings, mask=None):
        # Get block size
        block_size = x.shape[1]
        
        # Create causal mask
        causal_mask = self._prepare_blockwise_causal_attn_mask(block_size, block_size, x.device)
        
        # Apply model
        x = self.model(x, mask=causal_mask)
        
        return x
        
    def generate(
        self,
        text: str,
        num_frames: int,
        num_steps: int = 100,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None
    ):
        # Get text embeddings
        text_embeddings = self.text_encoder(text)
        
        # Initialize latents
        latents = torch.randn((1, num_frames, self.vae.latent_dim), device=self.device)
        
        # Process in blocks
        for i in range(0, num_frames, self.num_frames_per_block):
            # Get block
            block = latents[:, i:i+self.num_frames_per_block]
            
            # Process block
            block = self._process_block(block, text_embeddings)
            
            # Update latents
            latents[:, i:i+self.num_frames_per_block] = block
            
        # Decode latents
        audio = self.vae.decode(latents)
        
        return audio
        
    def sample(
        self,
        text: str,
        num_frames: int,
        num_steps: int = 100,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None
    ):
        # Get text embeddings
        text_embeddings = self.text_encoder(text)
        
        # Initialize latents
        latents = torch.randn((1, num_frames, self.vae.latent_dim), device=self.device)
        
        # Process in blocks
        for i in range(0, num_frames, self.num_frames_per_block):
            # Get block
            block = latents[:, i:i+self.num_frames_per_block]
            
            # Process block
            block = self._process_block(block, text_embeddings)
            
            # Sample from logits
            if top_k is not None:
                v, _ = torch.topk(block, top_k)
                block[block < v[:, [-1]]] = float('-inf')
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(block, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                block[indices_to_remove] = float('-inf')
                
            probs = F.softmax(block / temperature, dim=-1)
            block = torch.multinomial(probs, num_samples=1)
            
            # Update latents
            latents[:, i:i+self.num_frames_per_block] = block
            
        # Decode latents
        audio = self.vae.decode(latents)
        
        return audio 