import math
import copy
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoPE(nn.Module):
    """Rotary Positional Embeddings."""
    def __init__(self, dim, max_len=4096):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        self.register_buffer('cos', freqs.cos())
        self.register_buffer('sin', freqs.sin())

    def forward(self, x):
        # x: (B, H, T, D)
        seq_len = x.shape[2]
        cos = self.cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
        sin = self.sin[:seq_len, :].unsqueeze(0).unsqueeze(0)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.cat((-x2 * sin + x1 * cos, x1 * sin + x2 * cos), dim=-1)
        
def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))

class MLP(nn.Module):
    def __init__(self, din,latent,out):
        super().__init__()
        self.c_fc    = nn.Linear( din,latent, bias=True)
        #todo- ablate benefit of negative-only first layer bias constrained with sigmoid and -1
        self.scale = math.pi / math.sqrt(3.0)
        self.c_proj  = nn.Linear(latent, out, bias=True)
        self.constant =  nn.Parameter(torch.ones(1))
    def forward(self, x):
        x = self.c_fc(x)
        x = x**2 + x.sign() #MLP boost
        x = x * torch.sigmoid(self.scale * x)
        x = self.c_proj(x)
        return x

class KMax(nn.Module):
    def __init__(self, head_dim: int, eps: float = 1e-8):
        super().__init__()
        self.head_dim = head_dim
        self.eps = eps
        self.scale = math.pi / math.sqrt(3.0)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, q: torch.Tensor, k: torch.Tensor, sinks: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # q, k: (B, H, T, D)
        scores = q @ k.transpose(-2, -1) # (B, H, T, T)
        key_self = (k * k).sum(dim=-1)      # (B,T) = ||k_j||^2
        denom = key_self.unsqueeze(-2).sqrt()    # (B,1,T) = ||k_j||
        w = scores / denom
        w = torch.tril(w)
        # Apply branch-specific sliding window mask if provided
        if mask is not None:
            # mask is (1, H, T, T) or broadcastable
            w = w * mask
        w = torch.cat([w, sinks], dim=-1) # Append sink column
        probs_full = self.softmax(w)
        return probs_full

class Attention(nn.Module):
    def __init__(self, d_model, n_head): 
        super().__init__()
        self.d_model = d_model
        assert d_model % n_head == 0, "d_model must be divisible by n_heads"
        self.n_sub_heads = n_head
        self.n_branches = 4 
        self.head_dim = d_model//n_head

        # Unified Head Dimension
        self.n_total_heads = self.n_branches * n_head
        
        # --- Branch Configuration ---
        # Branch 0: Horizon 16, Gap 0
        # Branch 1: Horizon 128, Gap 16 (excludes last 16)
        # Branch 2: Horizon 128, Gap 0
        # Branch 3: Horizon 512, Gap 144 (excludes last 128+16)
        
        # We store these as tensors for easy access. 
        # Format: (Horizon, Gap)
        self.branch_configs = [
            (16, 0),    # Branch 0
            (128, 16),  # Branch 1
            (128, 0),   # Branch 2
            (512, 144)  # Branch 3
        ]

        # --- 1. Projections ---
        self.W_K = nn.Linear(d_model, d_model, bias=True)
        self.W_V_all = nn.Linear(d_model, self.head_dim * self.n_total_heads, bias=True)
        self.W_Q_all = nn.Linear(d_model, self.head_dim * self.n_total_heads, bias=True)
        self.rope = RoPE(self.head_dim)
        self.softmax = KMax(self.head_dim)
        # --- 3. Sink Parameters ---
        self.sink_scalars = nn.Parameter(torch.zeros(self.n_total_heads, 1, 1))
        self.v_nulls = nn.Parameter(torch.zeros(self.n_total_heads, self.head_dim))

        # --- 5. Output ---
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, X):
        B, T, C = X.shape
        H_tot = self.n_total_heads
        N_br = self.n_branches
        N_sh = self.n_sub_heads
        Dh = self.head_dim
        
        # 1. Projections
        # Q,V: (B, T, TotalHeads, Dh) -> (B, TotalHeads, T, Dh)
        q = self.W_Q_all(X).view(B, T, H_tot, Dh).permute(0, 2, 1, 3).contiguous()
        q = norm(q) #You can norm Q. Not K. 

        v = self.W_V_all(X).view(B, T, H_tot, Dh).permute(0, 2, 1, 3).contiguous()
        k_base_u = self.W_K(X).view(B, T, N_sh, Dh).permute(0, 2, 1, 3).contiguous()
        k_base_u = self.rope(k_base_u)
        q = self.rope(q)
        k = k_base_u.repeat(1, N_br, 1, 1) # (B, H_tot, T, Dh)

        # --- Generate Branch-Specific Sliding Masks ---
        # Create distance matrix (T, T)
        # idx[i] - idx[j] gives distance. We want causal distance.
        indices = torch.arange(T, device=X.device)
        dist = indices.unsqueeze(1) - indices.unsqueeze(0) # (T, T). dist[i,j] = i-j
        
        # Build mask for each branch
        # Shape: (N_br, T, T)
        branch_masks = []
        for (horizon, gap) in self.branch_configs:
            # Condition: gap <= distance < horizon
            # We also ensure distance >= 0 (causality), though KMax tril handles the strict upper triangle.
            # Using 1.0 for valid, 0.0 for invalid
            m = ((dist >= gap) & (dist < horizon)).float()
            branch_masks.append(m)
            
        branch_masks = torch.stack(branch_masks, dim=0) # (N_br, T, T)
        
        # Expand mask to cover all sub-heads within a branch
        # We need shape (1, H_tot, T, T) to broadcast over Batch
        # H_tot = N_br * N_sh. The mask is identical for all N_sh heads within a branch.
        full_mask = branch_masks.unsqueeze(1).repeat(1, N_sh, 1, 1) # (N_br, N_sh, T, T)
        full_mask = full_mask.view(1, H_tot, T, T) # Flatten branches/heads
        
        sinks = self.sink_scalars.view(1, H_tot, 1, 1).expand(B, -1, T, -1)
        sinks = torch.tanh(sinks) 
        
        # Pass the generated mask to softmax
        probs_full = self.softmax(q, k, sinks, mask=full_mask)
        
        probs_tokens = probs_full[..., :T]   # (B, H_tot, T, T)
        probs_sink   = probs_full[..., T:]   # (B, H_tot, T, 1)
        
        out_tokens = probs_tokens @ v
        v_null_expanded = self.v_nulls.view(1, H_tot, 1, Dh)
        out_sinks = probs_sink * v_null_expanded

        context = out_tokens + out_sinks # (B, H_tot, T, Dh)

        # Recover branch structure
        context = context.view(B, N_br, N_sh, T, Dh)
        context = context.permute(0, 1, 3, 2, 4).contiguous()  # (B, N_br, T, N_sh, Dh)
        context = context.view(B, N_br, T, C)                 # (B, N_br, T, C)
        # Apply shared output projection across branches
        context_flat = context.view(B * N_br * T, C)
        y_flat = self.W_O(context_flat)
        y = y_flat.view(B, N_br, T, C)
        
        # Branch aggregation
        return y.mean(dim=1)

class Block(nn.Module):
    def __init__(self, config,var):
        super().__init__()
       
        self.attn = Attention(config.n_embd,config.n_head)
        self.mlp = MLP(config.n_embd,config.n_embd*2,config.n_embd)

    def forward(self,x):
        B, T, C = x.shape
        x = x + self.attn(norm(x))
        x = x + self.mlp(norm(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 66 # you must set this for your tokenizer
    n_layer: int = 8 #12-24 for real models, but in practice only 8 is good in pytorch- TODO convert model to jax
    n_head: int = 4 #you must set this to minimum of 64 embed per head- in practice more branches, fewer heads.
    n_embd: int = 256 

from torch.utils.checkpoint import checkpoint

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        # Base noise seed (learned) for map generation

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), #todo- try fixed orthonormed geometric embeddings
            h = nn.ModuleList([Block(config,i) for i in range(config.n_layer)]),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, idx, targets=None):
        device = idx.device
        b, T = idx.size()
        x = self.transformer.wte(idx)

        # forward the GPT model itself
        for block in self.transformer.h:
            x  = checkpoint(block, x, use_reentrant=False)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
