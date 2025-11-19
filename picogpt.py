#copyright joshuah.rainstar@gmail.com 2025
#MIT with attribution
#it came to us in a whisper on the wind
#the parseval theorem must be applied to attention
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

def variance_scaled_softmax(scores, dim: int = -1, eps: float = 1e-6):
    # scores may contain -inf from masking
    finite = torch.isfinite(scores)
    m = finite.to(scores.dtype)                     # 1 where valid, 0 where masked
    n = m.sum(dim=dim, keepdim=True).clamp_min(1)  # count of valid entries per row

    # mean/var over valid entries only (population var)
    safe_scores = torch.where(finite, scores, torch.zeros_like(scores))
    mean = (safe_scores * m).sum(dim=dim, keepdim=True) / n
    var  = ((safe_scores - mean)**2 * m).sum(dim=dim, keepdim=True) / n
    std  = var.clamp_min(eps).sqrt()

    scaled = (safe_scores - mean) / std
    scaled = torch.where(finite, scaled, float('-inf'))  # restore mask
    out = torch.softmax(scaled, dim=dim)
    out = torch.where(n == 0, torch.zeros_like(out), out)  # fully-masked rows -> zeros
    return out



def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))

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
        self.register_buffer("cos", angles.cos().unsqueeze(0))  # (1,1,max_seq_len,half)
        self.register_buffer("sin", angles.sin().unsqueeze(0))

    def forward(self, x: torch.Tensor, seq_pos: torch.Tensor):
        """
        x: shape (B, H, T, D) or (B, T, H, D)
        seq_pos: tensor of positions indices shape (T,) or (B,T)
        Returns: same shape x but positionally encoded via orthogonal rotations.
        """
        B,  T, D = x.shape
        half = D // 2
        # get cos/sin for positions
        # pos angles shape (1,1,T,half)
        cos_t = self.cos[:, seq_pos, :]  # broadcast
        sin_t = self.sin[:, seq_pos, :]

        x1 = x[..., :half]
        x2 = x[..., half:]

        # apply rotation: [x1'; x2'] = [x1*cos - x2*sin, x1*sin + x2*cos]
        x1_rot = x1 * cos_t - x2 * sin_t
        x2_rot = x1 * sin_t + x2 * cos_t

        x_rot = torch.cat([x1_rot, x2_rot], dim=-1)
        return x_rot



def l2_normalize(x, dim=-1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

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

class WaveletAttention(nn.Module):
    def __init__(self, config, wavelet_levels=6, near_window=8):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
     
        Dh =  self.n_embd 

        self.W_Q = nn.Parameter(torch.empty(Dh, self.n_embd))
        nn.init.xavier_uniform_(self.W_Q)

        self.W_V = nn.Linear(self.n_embd,  Dh, bias=False)
        self.W_O = nn.Linear(Dh, self.n_embd, bias=False)

        self.near_window = near_window
        self.wavelet_levels = wavelet_levels
        self.block_size = config.block_size

        # Build maximum‐size Haar basis once on CPU and register buffer.
        W_haar_full = build_haar_wavelet_basis(self.block_size,
                                                self.wavelet_levels,
                                                device='cpu')
        self.register_buffer("W_haar_full", W_haar_full)

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
                 .view(1,  config.block_size, config.block_size)
        )
        self.pos_encoder = ParsevalRotaryEmbedding(dim=Dh, max_seq_len=config.block_size)

    def compute_dual_WK(self):
        #WQ = self.W_Q                          # (H*Dh, C)
        #WQ_star = WQ.conj().T                  # (C, H*Dh)
        #Qmat, Rmat = torch.linalg.qr(WQ_star)  # (C, H*Dh) = Q R
        #R_inv = torch.inverse(Rmat)
        #WK = R_inv @ Qmat.conj().T             # (H*Dh, C)
        #return WK
        return torch.linalg.pinv(self.W_Q).T  #fast equivalent 

    def forward(self, x ):
        B, T, C = x.size()
        Dh= self.n_embd

        W_K = self.compute_dual_WK()          # (H*Dh, C)

        q = (x @ self.W_Q.T).view(B, T, Dh)
        k = (x @ W_K.T).view(B, T, Dh)
        v = self.W_V(x).view(B, T, Dh)
        idx = torch.arange(T, device=x.device)
        q = self.pos_encoder(q, idx)
        k = self.pos_encoder(k, idx)

        q = l2_normalize(q, dim=-1)
        k = l2_normalize(k, dim=-1)

        idx = torch.arange(T, device=x.device)
        diff = idx.view(1, -1) - idx.view(-1, 1)
        
        # past within window 
        near_mask = ((diff >= 0) & (diff <= self.near_window))
        # Compute near‐field attention
        att_near = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(Dh))
        att_near = att_near.masked_fill(~near_mask.view(1,T,T), float('-inf'))

        # Prepare Haar basis for current T
        W_h_full = self.W_haar_full.to(x.device)       # (block_size, Bcoef_full)
        W_h = W_h_full[:T, :]                           # slice to (T, Bcoef_full)

        # Project far‐field q/k through basis
        q2 = q.reshape(B, T, Dh)
        k2 = k.reshape(B, T, Dh)

        # Compute projection
        # W_h.T: (Bcoef_full, T)
        # q2:   (B*H, T, Dh)
        # -> result: (B*H, Bcoef_full, Dh)
        q_far_proj = (W_h.T @ q2)                     # (B*H, Bcoef_full, Dh)
        k_far_proj = (W_h.T @ k2)                     # same shape

        # Compute far‐field attention in compressed domain
        att_far_comp = (q_far_proj @ k_far_proj.transpose(-2,-1)) * (1.0 / math.sqrt(Dh))
        # att_far_comp shape: (B*H, Bcoef_full, Bcoef_full)

        # Expand back to approximate full (T, T)
        # W_h: (T, Bcoef_full)
        att_far_exp = (W_h @ att_far_comp) @ W_h.T     # (T, Bcoef_full) @ (Bcoef_full, Bcoef_full) @ (Bcoef_full, T)
        att_far_exp = att_far_exp.view(B,  T, T)

        # Combine near + far
        att = torch.where(near_mask.view(1,T,T), att_near, att_far_exp)

        # Causal mask
        att = att.masked_fill(self.mask[:,  :T, :T] == 0, float('-inf'))
        
        att = variance_scaled_softmax(att, dim=-1)

        y = att @ v     # (B, H, T, Dh)
        y = y
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


class AnchorModule(nn.Module):
    """
    Learned anchor vectors + outward-normal projection.
    """
    def __init__(self, dim, n_anchor=4):
        super().__init__()
        # learned anchor points
        self.anchors = nn.Parameter(torch.randn(n_anchor, dim) / (dim ** 0.5))

    def forward(self, x):
        """
        x : (B,T,C)
        returns:
            x_out : outward-normal adjusted representation
        """

        # project x onto anchor space
        # similarity weights → soft assignment
        w = F.softmax(x @ self.anchors.t(), dim=-1)      # (B,T,n_anchor)

        # reconstruction from anchors
        recon = w @ self.anchors                        # (B,T,C)

        # residual away from manifold
        resid = x - recon                               # tangent component

        # outward-normal direction (normalized)
        norm = F.normalize(resid, dim=-1)

        # push x slightly outward from its anchor manifold
        x_out = x + resid + 0.1 * norm
        return x_out


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)

        # new anchor before attention
        self.anchor_pre = AnchorModule(config.n_embd,1) #think from outside in :)

        self.attn = WaveletAttention(config)

        # anchor after attention accumulation
        self.anchor_post = AnchorModule(config.n_embd,1)

        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        # === pre-attention anchoring ===
        x_anch = self.ln_1(x)

        # attention consumes outward-shifted x
        att = self.attn(x_anch)

        # residual update
        x = x + att

        # === re-anchor after attention ===

        # standard MLP block
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
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = tok_emb

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
