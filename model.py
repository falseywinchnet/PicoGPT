#copyright joshuah.rainstar@gmail.com 2025
#MIT with attribution

import math
import copy
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
# ----------------------------
# Layers
# ----------------------------

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
        b =self.bias if self.use_bias else None
        return F.layer_norm(x, self.weight.shape, self.weight, b, 1e-5)

def l2_normalize(x, dim=-1, eps=1e-6):
    # Cast to float32 for the norm calculation to prevent overflow (x^2)
    # and underflow (precision loss).
    x_float = x.float()
    norm = x_float.norm(dim=dim, keepdim=True)
    
    # Result is cast back to original dtype automatically by division if x is fp16,
    # but explicit casting ensures control.
    return x / (norm.to(x.dtype) + eps)
    

class ParsevalRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, theta_base: float = 10000.0):
        """
        dim: embedding dimension (must be even).
        max_seq_len: maximum sequence length for which to precompute sines/cosines.
        theta_base: base for frequency schedule (as in RoPE).
        """
        super().__init__()
        assert dim % 2 == 0, "dim must be even for pairing"
        self.dim = dim
        self.max_seq_len = max_seq_len

        # compute frequency for each pair
        half = dim // 2
        inv_freq = 1.0 / (theta_base ** (torch.arange(0, half, 1, dtype=torch.float32) / half))

        # position indices
        pos = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)  # (max_seq_len,1)
        # angles (max_seq_len x half) = pos * inv_freq
        angles = pos * inv_freq.unsqueeze(0)  # broadcast
        # compute cos and sin matrices for each pos and each half-dim
        self.register_buffer("cos", angles.cos().unsqueeze(0).unsqueeze(0))  # (1,1,max_seq_len,half)
        self.register_buffer("sin", angles.sin().unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor, seq_pos: torch.Tensor):
        """
        x: shape (B, H, T, D) or (B, T, H, D)
        seq_pos: tensor of positions indices shape (T,) or (B,T)
        Returns: same shape x but positionally encoded via orthogonal rotations.
        """
        # assume shape (B, H, T, D)
        B, H, T, D = x.shape
        half = D // 2
        # get cos/sin for positions
        # pos angles shape (1,1,T,half)
        cos_t = self.cos[:, :, seq_pos, :]  # broadcast
        sin_t = self.sin[:, :, seq_pos, :]

        x1 = x[..., :half]
        x2 = x[..., half:]

        # apply rotation: [x1'; x2'] = [x1*cos - x2*sin, x1*sin + x2*cos]
        x1_rot = x1 * cos_t - x2 * sin_t
        x2_rot = x1 * sin_t + x2 * cos_t

        x_rot = torch.cat([x1_rot, x2_rot], dim=-1)
        return x_rot


def build_haar_wavelet_basis(T, levels, device=None, dtype=torch.float32):
    """
    Build a Haar‐wavelet basis matrix W of shape (T, Bcoef).
    T: sequence length (must be divisible by 2^levels for full structure, but we will allow slicing).
    levels: number of levels of decomposition.
    """
    W_list = []
    for j in range(levels):
        block_count = 2**j
        block_size = T // block_count
        half = block_size // 2
        for k in range(block_count):
            vec = torch.zeros(T, dtype=dtype, device=device)
            start = k * block_size
            mid   = start + half
            end   = start + block_size
            if half > 0:
                vec[start:mid] =  1.0 / math.sqrt(half)
                vec[mid:end]  = -1.0 / math.sqrt(half)
            W_list.append(vec)
    W = torch.stack(W_list, dim=1)  # shape (T, Bcoef)
    return W

def variance_scaled_softmax(scores, dim: int = -1, eps: float = 1e-6):
    # scores may contain -inf from masking
    # Always compute softmax stats in float32
    dtype_in = scores.dtype
    scores_f32 = scores.float()
    
    finite = torch.isfinite(scores_f32)
    m = finite.to(scores_f32.dtype)                     # 1 where valid, 0 where masked
    n = m.sum(dim=dim, keepdim=True).clamp_min(1)  # count of valid entries per row

    # mean/var over valid entries only (population var)
    safe_scores = torch.where(finite, scores_f32, torch.zeros_like(scores_f32))
    mean = (safe_scores * m).sum(dim=dim, keepdim=True) / n
    
    # Squaring difference is risky in fp16, safe in fp32
    var  = ((safe_scores - mean)**2 * m).sum(dim=dim, keepdim=True) / n
    std  = var.clamp_min(eps).sqrt()

    scaled = (safe_scores - mean) / std
    
    # Restore -inf mask for softmax
    # We use float('-inf') which is valid in float32
    scaled = torch.where(finite, scaled, float('-inf'))
    
    # Softmax in float32 is standard stability practice
    out = torch.softmax(scaled, dim=dim)
    out = torch.where(n == 0, torch.zeros_like(out), out)
    
    # Cast back to original dtype (fp16)
    return out.to(dtype_in)

class DirectionalWedgeBias(nn.Module):
    def __init__(self, dim, heads, max_seq_len=1024, gamma=1.0):
        super().__init__()
        self.n_head = heads
        self.head_dim = dim // heads
        self.gamma = gamma
        self.max_seq_len = max_seq_len
        
        self.A = nn.Parameter(torch.empty(heads, self.head_dim, self.head_dim))
        nn.init.orthogonal_(self.A, gain=0.1)
        
        self.log_tau = nn.Parameter(torch.zeros(heads)) 
        
        # Cache for distance matrix to avoid recomputing on every forward
        # We register it as a buffer so it's saved but not updated by optimizers
        idx = torch.arange(max_seq_len)
        dist = (idx[None, :] - idx[:, None]).abs().view(1, 1, max_seq_len, max_seq_len)
        self.register_buffer("dist_cache", dist, persistent=False)

    def forward(self, x):
        B, T, D = x.shape
        H, Dh = self.n_head, self.head_dim
        
        # 1. Compute Wedge Geometry
        # Transpose and view
        v = x.view(B, T, H, Dh).transpose(1, 2) # (B, H, T, Dh)
        
        # --- FLOAT32 SAFEGUARD START ---
        # We perform the normalization and geometric projections in float32
        # to avoid exploding norms and loss of precision in the 'wedge' product.
        v_f32 = v.float()
        
        # rsqrt is safer, but we must prevent the square sum from overflowing first
        sq_norm = (v_f32 ** 2).sum(dim=-1, keepdim=True)
        
        # Normalize
        v_norm = v_f32 * torch.rsqrt(sq_norm + 1e-6)
        
        # Projection matrix S needs to be float32 for this operation
        # (casting on the fly is cheap compared to the instability risk)
        A_f32 = self.A.float()
        S = A_f32 - A_f32.transpose(-1, -2) # (H, Dh, Dh)
        
        Sv = torch.matmul(v_norm, S) 
        wedge = torch.matmul(Sv, v_norm.transpose(-1, -2)) # (B, H, T, T)
        
        # Cast back to fp16 (or bf16) for the decay and output
        wedge = wedge.to(x.dtype)
        # --- FLOAT32 SAFEGUARD END ---
        
        # 2. Apply Decay using Cached Distance
        dist = self.dist_cache[:, :, :T, :T]
        
        tau = F.softplus(self.log_tau).view(1, H, 1, 1) + 1e-4
        decay = torch.exp(-dist * 0.01 / tau) 

        return self.gamma * wedge * decay        

class ParsevalWaveletAttention(nn.Module):
    def __init__(self, config, near_window=64):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        
        assert self.head_dim * self.n_head == self.n_embd, "n_embd must be divisible by n_head"

        # Null Vector Parameters (The Sink)
        # One unique sink per head to allow independent "voting"
        self.k_null = nn.Parameter(torch.randn(1, 1, self.n_head, self.head_dim) * 0.02)
        self.register_buffer("v_null", torch.zeros(1, 1, self.n_head, self.head_dim))
        self.W_Q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.W_K = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.W_V = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.W_O = nn.Linear(self.n_embd, self.n_embd, bias=False)
        
        self.ln = LayerNorm(config.n_embd, bias=config.bias)
        
        # Auto-tune levels
        target_block = near_window * 2
        min_blocks = config.block_size / target_block
        self.wavelet_levels = max(3, int(math.ceil(math.log2(min_blocks))) + 1)
        
        self.near_window = near_window
        self.block_size = config.block_size
        
        W_haar_full = build_haar_wavelet_basis(self.block_size,
                                               self.wavelet_levels,
                                               device='cpu')
        
        W_haar_full = l2_normalize(W_haar_full, dim=0)
        
        self.register_buffer("W_haar_full", W_haar_full)

        mask_bool = torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)
        self.register_buffer("causal_mask", mask_bool)
        
        self.pos_encoder = ParsevalRotaryEmbedding(dim=self.head_dim, max_seq_len=config.block_size)
        
        # Pass max_seq_len to optimize bias
        self.wedge_bias = DirectionalWedgeBias(self.n_embd, self.n_head, 
                                              max_seq_len=config.block_size, 
                                              gamma=0.5)
    
    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head
        D = self.head_dim
        
        # 1. Projections
        q = self.W_Q(x).view(B, T, H, D).transpose(1, 2) # (B, H, T, D)
        k = self.W_K(x).view(B, T, H, D).transpose(1, 2)
        v = self.W_V(self.ln(x)).view(B, T, H, D).transpose(1, 2)
        idx = torch.arange(T, device=x.device)

        q = self.pos_encoder(q, idx)
        k = self.pos_encoder(k, idx)

        # L2 Normalize
        q = l2_normalize(q, dim=-1)
        k = l2_normalize(k, dim=-1)
        
        # ---------------------------------------------------------
        # 2. Compute Far Field (Base Layer)
        # ---------------------------------------------------------
        # We use this as the 'canvas' to reduce memory allocation
        W_h = self.W_haar_full[:T, :].to(x.device)
        
        q_flat = q.reshape(B * H, T, D)
        k_flat = k.reshape(B * H, T, D)
        
        # Project to Wavelet Domain (Compressed)
        q_far = W_h.T @ q_flat 
        k_far = W_h.T @ k_flat
        att_far_comp = q_far @ k_far.transpose(-2,-1) 
        
        # Reconstruct to Dense (B*H, T, T)
        # This is our initial attention map 'att'
        # Reshape immediately to (B, H, T, T) to match others
        att = ((W_h @ att_far_comp) @ W_h.T).view(B, H, T, T)
        
        # ---------------------------------------------------------
        # 3. Add Wedge Bias (In-Place)
        # ---------------------------------------------------------
        # We add the bias directly to the Far field canvas
        geo_bias = self.wedge_bias(x)
        att.add_(geo_bias)

        # ---------------------------------------------------------
        # 4. Compute Near Field (Sparse Override)
        # ---------------------------------------------------------
        # Create indices for mask
        # Band mask (Boolean)
        near_mask = (idx.view(1,-1) - idx.view(-1,1)).abs() <= self.near_window
        near_mask = near_mask.view(1, 1, T, T)
        
        # Compute Near Attention
        # We only strictly need to compute this, but without a banded kernel 
        # we must compute the full matrix. However, we can optimize the combination.
        att_near = q @ k.transpose(-2, -1)
        
        # Fused combination:
        # Instead of torch.where (which allocates a 3rd tensor), we use masked write.
        # att[near_mask] = att_near[near_mask] + geo_bias[near_mask]
        # Since we already added geo_bias to 'att', we just need to replace 
        # the specific values where the near mask is active.
        
        # Note: att currently contains (Far + Bias). 
        # We want (Near + Bias) where mask is True.
        # So: att[mask] = Near + Bias
        # We can extract just the masked elements to save memory bandwidth
        
        # Efficient update:
        att = torch.where(near_mask, att_near + geo_bias, att)
        
        # Apply Causal Mask
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        
        # ---------------------------------------------------------
        # 5. Null Vector & Softmax
        # ---------------------------------------------------------
        k_null_norm = l2_normalize(self.k_null, dim=-1)
        k_null_ex = k_null_norm.expand(B, -1, -1, -1).transpose(1, 2)
        null_scores = q @ k_null_ex.transpose(-2, -1)
        
        att_full = torch.cat([att, null_scores], dim=-1)
        
        # Cleanup large tensors before Softmax to free graph memory if possible
        del att_near, geo_bias, q_far, k_far
        
        att_full = variance_scaled_softmax(att_full, dim=-1)
        
        # Weighted Sum
        attn_seq_probs = att_full[..., :T]
        y = attn_seq_probs @ v
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_embd)
        
        return self.W_O(y)

# ----------------------------
# Transformer Block
# ----------------------------

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear( config.n_embd,4* config.n_embd, bias=config.bias)
        self.scale = math.pi / math.sqrt(3.0)

        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = x * torch.sigmoid(self.scale * x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = ParsevalWaveletAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
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

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

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

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, T = idx.size()

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
           
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
