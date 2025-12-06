#copyright joshuah.rainstar@gmail.com 2025
#MIT with attribution

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

class ConvexSoftmax(nn.Module):
    """Convex LSE (Float32 Precision)."""
    def forward(self, scores):
        m, _ = scores.max(dim=-1, keepdim=True)
        y = scores - m
        ex = y.exp()
        lse = m + ex.sum(dim=-1, keepdim=True).log()
        return torch.exp(scores - lse)

class BiasedWedge(nn.Module):
    """
    Symplectic Geometry with an Escape Hatch.
    S = (A - A^T) + D
    """
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

        # Apply Flow: x @ S
        # USE EINSUM TO PREVENT BROADCASTING ERRORS
        # b: Batch
        # h: TotalHeads (Matches S dim 0)
        # t: Time (Ignored by S)
        # d: Input Dim (Contracted)
        # e: Output Dim
        flow = torch.einsum('bhtd,hde->bhte', x, S)

        return x + flow

class Attention(nn.Module):
    def __init__(self, d_model, n_head, k_retrieval: int = 4):
        super().__init__()
        self.d_model = d_model
        # System Hierarchy
        assert d_model % n_head == 0, "d_model must be divisible by n_heads"
        self.n_sub_heads = n_head
        self.n_branches = 4
        self.head_dim = d_model//n_head

        # Unified Head Dimension
        self.n_total_heads = self.n_branches * n_head

        self.scale = self.head_dim ** -0.5
        self.k_retrieval = k_retrieval

        # --- 1. Projections ---
        self.W_K = nn.Linear(d_model, d_model, bias=True)
        self.W_Q_all = nn.Linear(d_model, d_model * self.n_branches, bias=False)
  
        # --- 2. Geometry ---
        self.wedge = BiasedWedge(self.head_dim, self.n_total_heads)
        self.rope = RoPE(self.head_dim)

        # --- 3. Sink Parameters ---
        self.sink_scalars = nn.Parameter(torch.zeros(self.n_total_heads, 1, 1))
        self.v_nulls = nn.Parameter(torch.zeros(self.n_branches, d_model))

        # --- 4. V network: maps marker (Dh) -> value (Dh)
        # This takes the "marker" (superposition of K) and hallucinates the value
        self.V_net = nn.Sequential(
            nn.Linear(self.head_dim, 4 * self.head_dim),
            nn.GELU(),
            nn.Linear(4 * self.head_dim, self.head_dim),
        )

        # --- 5. Output ---
        self.W_O_params = nn.Parameter(torch.empty(self.n_branches, d_model, d_model))
        self.W_O_bias = nn.Parameter(torch.zeros(self.n_branches, d_model))

        nn.init.xavier_uniform_(self.W_O_params)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, A,X):
        B, T, C = A.shape
        H_tot = self.n_total_heads
        N_br = self.n_branches
        N_sh = self.n_sub_heads
        Dh = self.head_dim
        K_top = min(self.k_retrieval, T)
        # 1. Projections
        # Q: (B, T, TotalHeads, Dh) -> (B, TotalHeads, T, Dh)
        q = self.W_Q_all(A).view(B, T, H_tot, Dh).permute(0, 2, 1, 3)

        # K: content-based key templates
        k_base = self.W_K(X)
        k_base_u = k_base.view(B, T, N_sh, Dh).permute(0, 2, 1, 3)
        k = k_base_u.repeat(1, N_br, 1, 1) # (B, H_tot, T, Dh)

        # 2. Geometry on Q, K
        q = self.wedge(q)
        k = self.wedge(k)

        q = self.rope(q)
        k = self.rope(k)

        # 3. Attention scores & mask
        # attn_scores: (B, H_tot, T, T)
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=A.device), diagonal=1).bool()
        attn_scores.masked_fill_(mask, float('-inf'))

        # Sinks
        sinks = self.sink_scalars.view(1, H_tot, 1, 1).expand(B, -1, T, -1)
        attn_scores_full = torch.cat([attn_scores, sinks], dim=-1)
        probs_full = self.softmax(attn_scores_full)

        # Split tokens vs sink mass
        probs_tokens = probs_full[..., :T]   # (B, H_tot, T, T)
        probs_sink   = probs_full[..., T:]   # (B, H_tot, T, 1)

        # 4. Build markers from K via top-K retrieval
        # vals, idxs: (B, H_tot, T, K_top)
        vals, idxs = probs_tokens.topk(K_top, dim=-1)

        # Normalized weights for the retrieved keys
        weights = vals / (vals.sum(dim=-1, keepdim=True) + 1e-9)  # (B, H_tot, T, K_top)

        # --- FIX: Flatten batch dims to perform gather safely ---
        # k: (B, H_tot, T, Dh) -> (B*H_tot, T, Dh)
        k_flat = k.contiguous().view(B * H_tot, T, Dh)
        
        # idxs: (B, H_tot, T, K_top) -> (B*H_tot, T*K_top)
        # We flatten the query structure so we can just grab items from T
        idxs_flat = idxs.contiguous().view(B * H_tot, T * K_top)
        
        # Expand indices for the feature dimension
        # (B*H_tot, T*K_top, Dh)
        idxs_expanded = idxs_flat.unsqueeze(-1).expand(-1, -1, Dh)
        
        # Gather: (B*H_tot, T*K_top, Dh)
        k_gathered_flat = k_flat.gather(dim=1, index=idxs_expanded)
        
        # Reshape back to structure: (B, H_tot, T, K_top, Dh)
        k_selected = k_gathered_flat.view(B, H_tot, T, K_top, Dh)
        
        # Weighted sum: (B, H_tot, T, K_top, 1) * (B, H_tot, T, K_top, Dh) -> sum over K_top
        marker = (weights.unsqueeze(-1) * k_selected).sum(dim=3) # (B, H_tot, T, Dh)

        # 5. Pass markers through V_net to get value vectors
        # marker_flat: (B*H_tot*T, Dh)
        marker_flat = marker.view(B * H_tot * T, Dh)
        v_flat = self.V_net(marker_flat) # (B*H_tot*T, Dh)
        out_tokens = v_flat.view(B, H_tot, T, Dh)

        # 6. Sink contribution
        # v_nulls: (Br, D) -> (Br*Sh, Dh) -> (1, H_tot, 1, Dh)
        v_null_expanded = self.v_nulls.view(N_br * N_sh, Dh).view(1, H_tot, 1, Dh)
        out_sinks = probs_sink * v_null_expanded

        context = out_tokens + out_sinks

        # 7. Output projection & Bias
        # Recover Branch dim
        context = context.view(B, N_br, N_sh, T, Dh)
        context = context.permute(0, 1, 3, 2, 4).contiguous().view(B, N_br, T, C)

        y_proj = torch.einsum('bntc,ncd->bntd', context, self.W_O_params)
        bias = self.W_O_bias.view(1, N_br, 1, C)

        y = y_proj + bias

        return y.mean(dim=1)


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.use_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(ndim))
        else:
            self.register_parameter("bias", None)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.bias if self.use_bias else None
        return F.layer_norm(x, self.weight.shape, self.weight, b, 1e-5)



class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear( config.n_embd,4* config.n_embd, bias=config.bias)
        self.scale = math.pi / math.sqrt(3.0)
        self.ln = LayerNorm(config.n_embd*4, bias=config.bias)
        
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = x**2 + 0.75*x**3
        x = self.ln(x)
        x = x * torch.sigmoid(self.scale * x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_3 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_4 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_5 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_6 = LayerNorm(config.n_embd, bias=config.bias)

        self.left = Attention(config.n_embd,config.n_head)
        self.right = Attention(config.n_embd,config.n_head)

        self.mlpa = MLP(config)
        self.mlpb = MLP(config)

    def forward(self,x):
        B, T, C = x.shape
        half = C//2
        A = x[...,:half]
        B = x[...,half:]
        A = A + self.left(self.ln_1(B),self.ln_2(A))
        A = A + self.mlpa(self.ln_3(A))

        B = B + self.right(self.ln_4(A),self.ln_5(B))
        B = B + self.mlpb(self.ln_6(A))

        x = torch.cat([A,B],dim=-1)

        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        # Base noise seed (learned) for map generation

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_q = LayerNorm(config.n_embd, bias=config.bias),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.synth = MLP(config)


        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params


    def forward(self, idx, targets=None):
        device = idx.device
        b, T = idx.size()
        tok_emb = self.transformer.wte(idx)
        q = self.transformer.ln_q(self.synth(torch.ones_like(tok_emb)))
        x = torch.cat([tok_emb,q],dim=-1)

        # forward the GPT model itself
        for block in self.transformer.h:
            x  = block(x)

        A = x[...,:self.config.n_embd]


        x = self.transformer.ln_f(A)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)

            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
