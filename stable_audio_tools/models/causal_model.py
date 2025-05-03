import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

from .transformer import ContinuousTransformer, TransformerBlock, RotaryEmbedding

class CausalWanSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_heads = 64,
        causal = True,
        zero_init_output = True,
        **kwargs
    ):
        super().__init__()
        
        self.dim = dim
        self.dim_heads = dim_heads
        self.causal = causal
        
        self.to_qkv = nn.Linear(dim, dim_heads * 3, bias=False)
        self.to_out = nn.Linear(dim_heads, dim, bias=False)
        
        if zero_init_output:
            nn.init.zeros_(self.to_out.weight)
            
    def _prepare_blockwise_causal_attn_mask(self, i, j, device):
        mask = torch.ones((i, j), device=device, dtype=torch.bool)
        if self.causal:
            mask = torch.tril(mask)
        return mask
        
    def forward(self, x, mask=None):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        
        # Reshape for attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.dim_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.dim_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.dim_heads)
        
        # Compute attention
        dots = torch.einsum('bhid,bhjd->bhij', q, k) / (self.dim_heads ** 0.5)
        
        # Apply causal mask
        if self.causal:
            causal_mask = self._prepare_blockwise_causal_attn_mask(dots.shape[-2], dots.shape[-1], dots.device)
            dots = dots.masked_fill(~causal_mask, float('-inf'))
            
        # Apply input mask
        if mask is not None:
            dots = dots.masked_fill(~mask, float('-inf'))
            
        # Softmax
        attn = F.softmax(dots, dim=-1)
        
        # Apply attention
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        
        # Reshape back
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        # Project out
        out = self.to_out(out)
        
        return out

class CausalWanAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_heads = 64,
        causal = True,
        zero_init_branch_outputs = True,
        **kwargs
    ):
        super().__init__()
        
        self.dim = dim
        self.dim_heads = dim_heads
        self.causal = causal
        
        self.attn = CausalWanSelfAttention(
            dim=dim,
            dim_heads=dim_heads,
            causal=causal,
            zero_init_output=zero_init_branch_outputs
        )
        
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
        if zero_init_branch_outputs:
            nn.init.zeros_(self.ff[-1].weight)
            
    def forward(self, x, mask=None):
        # Self attention
        x = x + self.attn(x, mask=mask)
        
        # Feed forward
        x = x + self.ff(x)
        
        return x

class CausalHead(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        zero_init_output = True
    ):
        super().__init__()
        
        self.dim = dim
        self.dim_out = dim_out
        
        self.to_out = nn.Linear(dim, dim_out, bias=False)
        
        if zero_init_output:
            nn.init.zeros_(self.to_out.weight)
            
    def forward(self, x):
        return self.to_out(x)

class CausalTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        dim_heads = 64,
        causal = True,
        zero_init_branch_outputs = True,
        **kwargs
    ):
        super().__init__()
        
        self.dim = dim
        self.depth = depth
        self.dim_heads = dim_heads
        self.causal = causal
        
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(
                dim=dim,
                dim_heads=dim_heads,
                causal=causal,
                zero_init_branch_outputs=zero_init_branch_outputs
            )
            for _ in range(depth)
        ])
        
        self.head = CausalHead(
            dim=dim,
            dim_out=dim,
            zero_init_output=zero_init_branch_outputs
        )
        
    def forward(self, x, mask=None):
        # Apply blocks
        for block in self.blocks:
            x = block(x, mask=mask)
            
        # Apply head
        x = self.head(x)
        
        return x 