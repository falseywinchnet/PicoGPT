import math
import copy
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

class PhaseVectorImprovedPositional(nn.Module):
    def __init__(self, dim, max_deg=None):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.max_deg = max_deg  # max Chebyshev degree; if None, set from dim

    def forward(self, q, k):
        B, Heads, T, D = q.shape
        device = q.device
        dtype = q.dtype
        H = D // 2

        # Choose max_deg if not provided: a bit larger than number of pairs
        max_deg = self.max_deg
        if max_deg is None:
            max_deg = max(3, 2 * H)  # e.g. up to degree 2H

        # Normalized time u in [0,1], then x in [-1,1]
        if T > 1:
            u = torch.arange(T, device=device, dtype=dtype) / (T - 1)
            x = 2.0 * u - 1.0   # (T,)
        else:
            x = torch.zeros(1, device=device, dtype=dtype)  # degenerate case

        # Precompute Chebyshev polynomials T_n(x) for n = 0..max_deg
        # Shape: (T, max_deg+1)
        T_all = torch.empty(T, max_deg + 1, device=device, dtype=dtype)
        T_all[:, 0] = 1.0
        if max_deg >= 1:
            T_all[:, 1] = x
        for d in range(2, max_deg + 1):
            T_all[:, d] = 2.0 * x * T_all[:, d - 1] - T_all[:, d - 2]

        # Map feature pairs to degrees n_f in [1, max_deg-1]
        if H > 1:
            f_idx = torch.arange(H, device=device, dtype=dtype)
            frac = f_idx / (H - 1)  # 0..1 across pairs
        else:
            frac = torch.zeros(1, device=device, dtype=dtype)

        # Spread degrees roughly across the available range
        n_f = 1 + (frac * (max_deg - 2)).round().to(torch.long)  # (H,)
        n_f = torch.clamp(n_f, 1, max_deg - 1)
        n_f_plus = n_f + 1  # still <= max_deg

        # Gather raw1 = T_{n_f}(x), raw2 = T_{n_f+1}(x)
        # T_all: (T, max_deg+1), n_f: (H,)
        raw1 = T_all[:, n_f]       # (T, H)
        raw2 = T_all[:, n_f_plus]  # (T, H)

        # Normalize to get orientation vectors on the unit circle
        norm = torch.sqrt(raw1 * raw1 + raw2 * raw2 + 1e-8)  # (T,H)
        base1 = (raw1 / norm).unsqueeze(0).unsqueeze(0)  # (1,1,T,H)
        base2 = (raw2 / norm).unsqueeze(0).unsqueeze(0)  # (1,1,T,H)

        def apply_rot(x_in):
            x1, x2 = x_in[..., :H], x_in[..., H:]
            xr1 = x1 * base1 - x2 * base2
            xr2 = x1 * base2 + x2 * base1
            return torch.cat([xr1, xr2], dim=-1)

        q_def = apply_rot(q)
        k_def = apply_rot(k)
        return q_def, k_def

        
class BiasedWedge(nn.Module):
    def __init__(self, head_dim, total_heads):
        super().__init__()
        self.head_dim = head_dim
        self.total_heads = total_heads

        # 1. Shared Skew (Global Twist): (D, D)
        self.A = nn.Parameter(torch.zeros(head_dim, head_dim))

        # 2. Identity Bias (Local Preservation): (TotalHeads, D)
        self.id_bias = nn.Parameter(torch.zeros(total_heads, head_dim))

    def forward(self, x):
        # x: (B, TotalHeads, T, D)

        # Construct S
        skew = self.A - self.A.transpose(-1, -2) # (D, D)
        diag = torch.diag_embed(self.id_bias)      # (H, D, D)
        S = skew + diag # Broadcasts to (H, D, D)
        
        flow = torch.einsum('bhtd,hde->bhte', x, S)

        return x + flow

def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))

class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c_fc    = nn.Linear( dim,4* dim, bias=True)
        #todo- ablate benefit of negative-only first layer bias constrained with sigmoid and -1
        self.scale = math.pi / math.sqrt(3.0)
        self.c_proj  = nn.Linear(4 * dim, dim, bias=True)
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
        self.W_V = nn.Linear(d_model, d_model, bias=True)

        self.W_Q_all = nn.Linear(d_model, d_model * self.n_branches, bias=True)

        # --- 2. Geometry ---
        self.wedge = BiasedWedge(self.head_dim, self.n_total_heads)
        self.rope = PhaseVectorImprovedPositional(self.head_dim)

        # --- 3. Sink Parameters ---
        self.sink_scalars = nn.Parameter(torch.zeros(self.n_total_heads, 1, 1))
        self.v_nulls = nn.Parameter(torch.zeros(self.n_branches, d_model))

        # --- 4. V network: maps marker (Dh) -> value (Dh)
        # This takes the "marker" (hologram of K geometry) and locates the value

        # --- 5. Output ---
        self.W_O_params = nn.Parameter(torch.empty(self.n_branches, d_model, d_model))
        self.W_O_bias = nn.Parameter(torch.zeros(self.n_branches, d_model))

        nn.init.xavier_uniform_(self.W_O_params)
        self.softmax = nn.Softmax(dim=-1)

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
        v_base = self.W_V(X).view(B, T, N_sh, Dh).permute(0, 2, 1, 3)
        v = v_base.repeat(1, N_br, 1, 1)
        
        k_base = self.W_K(X)
        k_base_u = k_base.view(B, T, N_sh, Dh).permute(0, 2, 1, 3).contiguous()
        k = k_base_u.repeat(1, N_br, 1, 1) # (B, H_tot, T, Dh)
        k_vanilla = k.clone()

        # 2. Geometry on Q
        q = self.wedge(q)
        # 3. PosEMB on both
        q , k= self.rope(q,k)



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

        w = torch.cat([w, sinks], dim=-1)

        alpha = (w.sum(dim=-1, keepdim=True) + 1e-6).reciprocal()
        probs_full = w * alpha

        # Split tokens vs sink mass
        probs_tokens = probs_full[..., :T]   # (B, H_tot, T, T)
        probs_sink   = probs_full[..., T:]   # (B, H_tot, T, 1)
        out_tokens = probs_tokens @ v

        # Sinks (Reshape v_nulls to broadcast correctly)
        # v_nulls: (Br, D) -> (Br*Sh, Dh) -> (1, H_tot, 1, Dh)
        v_null_expanded = self.v_nulls.view(N_br * N_sh, Dh).view(1, H_tot, 1, Dh)
        out_sinks = probs_sink * v_null_expanded

        context = out_tokens + out_sinks # (B, H_tot, T, Dh)

        # 5. Output
        # Recover Branch dim
        context = context.view(B, N_br, N_sh, T, Dh)
        context = context.permute(0, 1, 3, 2, 4).contiguous().view(B, N_br, T, C)
        
        # Projection & Bias (Explicit broadcast fix for Bias)
        y_proj = torch.einsum('bntc,ncd->bntd', context, self.W_O_params)
        bias = self.W_O_bias.view(1, N_br, 1, C)
        
        y = y_proj + bias

        return y.mean(dim=1)

class Block(nn.Module):
    def __init__(self, config,var):
        super().__init__()
       
        self.attn = Attention(config.n_embd,config.n_head)
        self.mlp = MLP(config.n_embd)

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
            synth = MLP(config.n_embd),
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
