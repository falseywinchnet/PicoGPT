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
    def forward(self, x):
        x = self.c_fc(x)
        x = x**2 + 0.7049*x**3
        x = x * torch.sigmoid(self.scale * x)
        x = self.c_proj(x)
        return x

class Attention(nn.Module):
    def __init__(self, d_model, n_head): 
        super().__init__()
        #todo- ablate larger k_retrieval(we know >4 is optimal) and add sparse attention with DSA scanner from deepseek and SSA full/sparse disciplining
        #from "sparse sparse attention". 
        self.d_model = d_model
        # System Hierarchy
        assert d_model % n_head == 0, "d_model must be divisible by n_heads"
        self.n_sub_heads = n_head
        self.n_branches = 4 #todo: validate higher branch counts in larger models. expensive to ablate
        self.head_dim = d_model//n_head

        # Unified Head Dimension
        self.n_total_heads = self.n_branches * n_head
        self.scale = math.pi / math.sqrt(3.0)

        self.attnscale = self.head_dim ** -0.5

        # --- 1. Projections ---
        self.W_K = nn.Linear(d_model, d_model, bias=True)
        self.W_V_all = nn.Linear(d_model, self.head_dim * self.n_total_heads, bias=True)
        self.W_Q_all = nn.Linear(d_model, self.head_dim * self.n_total_heads, bias=True)

        self.rope = RoPE(self.head_dim)

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
        # Q: (B, T, TotalHeads, Dh) -> (B, TotalHeads, T, Dh)
        q = self.W_Q_all(X).view(B, T, H_tot, Dh).permute(0, 2, 1, 3).contiguous()
        # V: Expand to TotalHeads
        v = self.W_V_all(X).view(B, T, H_tot, Dh).permute(0, 2, 1, 3).contiguous()
        
        k_base = self.W_K(X)
        k_base_u = k_base.view(B, T, N_sh, Dh).permute(0, 2, 1, 3).contiguous()
        k = k_base_u.repeat(1, N_br, 1, 1) # (B, H_tot, T, Dh)
        # 3. PosEMB on both
        q = self.rope(q)
        k = self.rope(k)

        scores = (q @ k.transpose(-2, -1)) * self.attnscale
        key_self = (k * k).sum(dim=-1).clamp_min(1e-6)      # (B,T) = ||k_j||^2
        denom = key_self.unsqueeze(-2).sqrt()               # (B,1,T) = ||k_j||
        w = scores / denom

        # kernel weights then row-normalize
        w = F.softplus(w)                             # >= 0, softplus(-inf)=0
        mask = torch.triu(torch.ones(T, T, device=X.device), diagonal=1).bool()

        w = w.masked_fill(mask, float("0.0"))

        w = w * torch.sigmoid(self.scale * w)
        sinks = self.sink_scalars.view(1, H_tot, 1, 1).expand(B, -1, T, -1)
        sinks = torch.tanh(sinks)+1e-6 #gate this behavior to reasonable regimes
        w = torch.cat([w, sinks], dim=-1)

        alpha = (w.sum(dim=-1, keepdim=True) + 1e-6).reciprocal()
        probs_full = w * alpha

        # Split tokens vs sink mass
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
