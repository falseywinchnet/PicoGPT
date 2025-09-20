#copyright joshuah.rainstar@gmail.com 2025
from __future__ import annotations
import math
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Tuple


def phase_transport_between(curr: torch.Tensor, prev: torch.Tensor, tau: float = 1e-6) -> torch.Tensor:
    B, T, C = curr.shape
    eps = tau

    u = _unit(curr)
    v = _unit(prev)
    w = curr - prev

    c = (u * v).sum(dim=-1, keepdim=True)  # (B,T,1)

    # masks (all as (B,T))
    near_pos = (c > 1.0 - tau).squeeze(-1)           # (B,T)
    near_neg = (c < -1.0 + tau).squeeze(-1)          # (B,T)
    small_u  = (_norm(curr) < tau).squeeze(-1)       # (B,T)  <-- FIX
    small_v  = (_norm(prev) < tau).squeeze(-1)       # (B,T)  <-- FIX
    trivial  = near_pos | small_u | small_v          # (B,T)

    # general branch
    denom  = (1.0 + c).clamp_min(eps)                # (B,T,1)
    a = (v * w).sum(dim=-1, keepdim=True)
    b = (u * w).sum(dim=-1, keepdim=True)
    Kw  = u * a - v * b
    K2w = u * (a * c - b) + v * (b * c - a)
    y_gen = w - Kw + (K2w / denom)                   # (B,T,C)

    # antipodal branch
    if C == 1:
        y_neg = -w
    else:
        v_flat = v.reshape(-1, C)
        p_flat = _orthonormal_perp(v_flat)
        p = p_flat.view(B, T, C)
        proj_v = (v * w).sum(dim=-1, keepdim=True) * v
        proj_p = (p * w).sum(dim=-1, keepdim=True) * p
        y_neg = w - 2.0 * proj_v - 2.0 * proj_p

    # blend (no in-place masked writes)
    y = torch.where(trivial.unsqueeze(-1), w, y_gen)
    y = torch.where(near_neg.unsqueeze(-1), y_neg, y)
    return y

def _norm(v, eps: float = 1e-12):
    return torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(eps)

def _unit(v, eps: float = 1e-12):
    return v / _norm(v, eps)

def _orthonormal_perp(v: torch.Tensor):
    # v: (..., C), returns p ⟂ v, ||p||=1
    *batch, C = v.shape
    flat = v.reshape(-1, C)
    idx = torch.argmin(torch.abs(flat), dim=-1)
    e = torch.zeros_like(flat)
    e.scatter_(1, idx.unsqueeze(1), 1.0)
    proj = (e * flat).sum(dim=-1, keepdim=True) * flat
    p = e - proj
    p = p / _norm(p)
    return p.view(*batch, C)


# ---------- materialize left-aligned centroids at every t for each scale from two trees ----------
def materialize_centroids_from_trees(x: torch.Tensor, levels0, levels1, K: int):
    """
    For each scale W=2^f (f>=1), build mu^{(W)}(t) as the centroid of the left-aligned
    block that contains t. Uses offset-0 tree if block starts at 0 mod W; otherwise offset-1 tree.
    Returns: mu_all: (B, T, K-1, C)   (no scale-1 here; K-1 levels for f=1..K-1)
    """
    B, T, C = x.shape
    device = x.device
    t_idx = torch.arange(T, device=device)  # (T,)
    mus = []
    for f in range(1, K):
        W = 1 << f
        # block start for t: s = floor(t/W)*W
        s = (t_idx // W) * W           # (T,)
        use_offset1 = (s % 2 == 1)     # whether the block start is odd (needs tree-1)
        if f-1 < len(levels0):
            L0 = levels0[f-1]          # (B, N0, C)
            N0 = L0.shape[1]
        else:
            N0 = 0
        if f-1 < len(levels1):
            L1 = levels1[f-1]          # (B, N1, C)
            N1 = L1.shape[1]
        else:
            N1 = 0

        # index within chosen tree:
        # for offset-0 (blocks [0..W-1], [W..2W-1], ...): idx0 = floor(t/W)
        # for offset-1 (blocks [1..W], [W+1..2W], ...):   idx1 = floor((t-1)/W)
        idx0 = (t_idx // W).clamp_max(max(N0-1, 0))
        idx1 = ((t_idx - 1).clamp_min(0) // W).clamp_max(max(N1-1, 0))

        # gather from the two trees
        mu0 = L0.index_select(1, idx0) if N0 > 0 else x.new_zeros(B, T, C)
        mu1 = L1.index_select(1, idx1) if N1 > 0 else x.new_zeros(B, T, C)

        mu = torch.where(use_offset1.view(1, T, 1), mu1, mu0)  # (B,T,C)

        # early region safety: if t < W-1 there is no full left-aligned block yet → zero
        mu = torch.where((t_idx < (W-1)).view(1, T, 1), torch.zeros_like(mu), mu)
        mus.append(mu)
    if len(mus) == 0:
        return x.new_zeros(B, T, 0, C)
    return torch.stack(mus, dim=2)  # (B, T, K-1, C)

class CausalCentroidPyramid(nn.Module):
    """
    Vectorized causal pyramid:
      inputs x: (B, T, C)
      returns deltas: (B, T, K, C)
        where K = 1 (token PT) + (num_scales-1) (cluster PTs)
    """
    def __init__(self, num_scales: int, tau: float = 1e-6):
        super().__init__()
        assert num_scales >= 1
        self.K = num_scales
        self.tau = float(tau)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        device = x.device

        # token-level PT (scale-1)
        prev_tok = torch.zeros_like(x)
        if T > 1:
            prev_tok[:, 1:, :] = x[:, :-1, :].contiguous()
        d1 = phase_transport_between(x, prev_tok, tau=self.tau)  # (B,T,C)
        d1[:, :1, :].zero_()  # mask early region (no previous token)

        if self.K == 1:
            return d1.unsqueeze(2)  # (B,T,1,C)

        # Build centroids causally with recursive halves
        mus = []
        mu_prev = x  # μ_0 = x (width=1)
        for s in range(1, self.K):              # s=1..K-1 (width W=2^s)
            W1 = 1 << (s - 1)
            shifted = torch.zeros_like(mu_prev)
            if T > W1:
                shifted[:, W1:, :] = mu_prev[:, :-W1, :].contiguous()  # μ_{s-1}(t - 2^{s-1})

            mu_s = 0.5 * (mu_prev + shifted)     # μ_s(t)
            # zero early region t < W-1
            W = 1 << s
            if W > 1:
                mu_s[:, :W-1, :].zero_()
            mus.append(mu_s)
            mu_prev = mu_s

        mu_all = torch.stack(mus, dim=2) if mus else x.new_zeros(B, T, 0, C)  # (B,T,K-1,C)

        # PT deltas between adjacent causal chunks
        d_list = []
        for j in range(self.K - 1):
            W = 1 << (j + 1)
            prev_mu = torch.zeros_like(mu_all[:, :, j, :])
            if T > W:
                prev_mu[:, W:, :] = mu_all[:, :-W, j, :].contiguous()
            d = phase_transport_between(mu_all[:, :, j, :], prev_mu, tau=self.tau)
            d[:, :W, :].zero_()  # mask early region
            d_list.append(d)

        d_clusters = torch.stack(d_list, dim=2) if d_list else x.new_zeros(B, T, 0, C)
        return torch.cat([d1.unsqueeze(2), d_clusters], dim=2)  # (B,T,K,C)


# ----- STREAMING STATE FOR INFERENCE -----
class CausalPyramidState:
    """
    O(K) step-time updates, no recompute.
    For level ℓ we keep a ring buffer of length 2^ℓ storing μ_ℓ (with μ_0=x).
    That suffices both to:
      - build μ_{ℓ+1}(t) from μ_ℓ(t) and μ_ℓ(t-2^ℓ)
      - compute deltas at scale s=ℓ via μ_s(t-2^s)
    """
    def __init__(self, num_scales: int, C: int, device, batch_size: int = 1, tau: float = 1e-6):
        self.K = num_scales
        self.C = C
        self.B = batch_size
        self.device = device
        self.tau = float(tau)
        self.t = 0  # number of tokens processed so far

        # ring buffers: list over levels ℓ = 0..K-1, each [B, L=2^ℓ, C]
        self.buffers = []
        self.ptrs = []
        for l in range(self.K):
            L = 1 << l
            self.buffers.append(torch.zeros(self.B, L, C, device=device))
            self.ptrs.append(0)

    def _read_lookback(self, level: int, r: int):
        """return μ_level(t - r); zeros if not enough history yet"""
        if self.t < r:
            return torch.zeros(self.B, self.C, device=self.device)
        L = self.buffers[level].size(1)
        idx = (self.ptrs[level] - r) % L
        return self.buffers[level][:, idx, :]

    def _push(self, level: int, value: torch.Tensor):
        """write current μ_level(t) and advance ptr"""
        L = self.buffers[level].size(1)
        self.buffers[level][:, self.ptrs[level], :] = value
        self.ptrs[level] = (self.ptrs[level] + 1) % L

    @torch.no_grad()
    def step(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        x_t: (B, C)
        returns d(t): (B, K, C)  [token PT + (K-1) cluster PTs]
        """
        B, C = x_t.shape
        feats = []

        # ------- token PT (read BEFORE any push) -------
        prev_x = self._read_lookback(level=0, r=1)  # μ0(t-1)
        d1 = phase_transport_between(x_t[:, None, :], prev_x[:, None, :], tau=self.tau).squeeze(1)
        if self.t == 0:
            d1.zero_()
        feats.append(d1)

        # ------- (A) compute all μ_s(t) with pre-push lookbacks -------
        mu_curr = [None] * self.K
        mu_curr[0] = x_t                      # μ0(t)
        mu_prev = x_t
        for s in range(1, self.K):
            W1 = 1 << (s - 1)
            W  = 1 << s
            mu_back = self._read_lookback(level=s-1, r=W1)   # μ_{s-1}(t - 2^{s-1})  (pre-push!)
            mu_s_t  = 0.5 * (mu_prev + mu_back)              # μ_s(t)
            if self.t < (W - 1):                             # early mask (global t)
                mu_s_t.zero_()
            mu_curr[s] = mu_s_t
            mu_prev = mu_s_t

        # ------- (B) compute all deltas d_s using μ_s(t−W) (pre-push) -------
        for s in range(1, self.K):
            W = 1 << s
            mu_prevW = self._read_lookback(level=s, r=W)     # μ_s(t - 2^s)  (pre-push!)
            d_s = phase_transport_between(mu_curr[s][:, None, :], mu_prevW[:, None, :], tau=self.tau).squeeze(1)
            if self.t + 1 <= W:
                d_s.zero_()
            feats.append(d_s)

        # ------- (C) push μ_ℓ(t) for all levels, exactly once -------
        self._push(level=0, value=mu_curr[0])
        for s in range(1, self.K):
            self._push(level=s, value=mu_curr[s])

        self.t += 1
        return torch.stack(feats, dim=1)  # (B, K, C)


class SemanticClusterFeaturesCausal(nn.Module):
    """
    Unified wrapper:
      - forward(x): vectorized for training
      - step(x_t, state): single-step for inference with cache
    """
    def __init__(self, num_scales: int, tau: float = 1e-6):
        super().__init__()
        self.pyramid = CausalCentroidPyramid(num_scales=num_scales, tau=tau)
        self.K = num_scales
        self.tau = float(tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pyramid(x)  # (B,T,K,C)

    @torch.no_grad()
    def step(self, x_t: torch.Tensor, state: CausalPyramidState) -> torch.Tensor:
        return state.step(x_t)  # (B,K,C)



# ---------------------------
# Small utilities
# ---------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUContextNav(nn.Module):
    def __init__(self, C: int, K: int, dropout: float = 0.0):
        super().__init__()
        self.C, self.K = C, K
        self.W_h = nn.Linear(C, C)
        self.W_r = nn.Linear(C, C)           # computed once (kept as in your spec)
        self.W_z = nn.Linear(C, C)
        self.U_h = nn.ModuleList([nn.Linear(C, C, bias=False) for _ in range(K)])
        self.W_init = nn.Linear(C, C)
        self.U_Z = nn.Linear(C, C, bias=False)
        self.drop = nn.Dropout(dropout)
        self.out_ln = nn.LayerNorm(C)

    def _ensure_seq(self, base, feats):
        """
        Accept:
          base  : (B,C) or (B,T,C)
          feats : (B,K,C) or (B,T,K,C)
        Normalize to:
          base  -> (B,T,C)
          feats -> (B,T,K,C)  (broadcast T if missing)
        """
        squeeze_time = False
        if base.dim() == 2:                    # (B,C) -> (B,1,C)
            base = base[:, None, :]
            squeeze_time = True
        elif base.dim() != 3:
            raise ValueError("base must be (B,C) or (B,T,C)")

        if feats.dim() == 3:                   # (B,K,C) -> (B,1,K,C), will broadcast over T
            feats = feats[:, None, :, :]
        elif feats.dim() != 4:
            raise ValueError("feats must be (B,K,C) or (B,T,K,C)")

        return base, feats, squeeze_time

    def forward(self, base: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        base, feats, squeeze_time = self._ensure_seq(base, feats)  # (B,T,C), (B,T,K,C)
        B, T, K, C = feats.size()
        assert K == self.K and C == self.C

        ground   = self.W_h(base)        # (B,T,C)
        criteria = self.W_r(base)        # (B,T,C)  (kept for parity with your sketch)
        state    = self.W_z(base)        # (B,T,C)

        # h_hat stack
        hhat_sum = torch.zeros_like(base)
        for i in range(K):
            ui = self.U_h[i](feats[:, :, i, :])       # (B,T,C)
            hhat_i = torch.tanh(ground + ui)          # (B,T,C)
            hhat_sum = hhat_sum + hhat_i

        initial = self.W_init(self.drop(hhat_sum))    # (B,T,C)

        # reset gate — FIX 1: no extra dimension
        reset = torch.sigmoid(state + initial)        # (B,T,C)

        # h_final with reset — FIX 2: multiply in (B,T,C) space
        h_final = torch.zeros_like(base)
        for i in range(K):
            ui = self.U_h[i](reset * feats[:, :, i, :])   # (B,T,C)
            h_final = h_final + torch.tanh(ground + ui)   # (B,T,C)

        update = torch.sigmoid(state + self.U_Z(initial)) # (B,T,C)
        result = (1.0 - update) * base + update * h_final
        result = self.out_ln(result)

        if squeeze_time:
            result = result[:, 0, :]
        return result

    @torch.no_grad()
    def step(self, base_t: torch.Tensor, feats_t: torch.Tensor) -> torch.Tensor:
        # base_t: (B,C), feats_t: (B,K,C)
        return self.forward(base_t, feats_t)

class GPTSemanticBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        C = config.n_embd
        self.C = C
        self.K = config.n_scales
        self.L = 1 + self.K                    # [K features | base]
        self.features = SemanticClusterFeaturesCausal(num_scales=self.K, tau=1e-6)
        self.drop = nn.Dropout(config.dropout)
        self.ln = nn.LayerNorm(self.C)
        self.nav = GRUContextNav(C=C, K=self.K, dropout=config.dropout)
    
    # vectorized
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,C)
        feats = self.features(x)                      # (B,T,K,C)
        return self.nav(x, feats)             # returns (B,T,C)

    # single-step incremental
    @torch.no_grad()
    def step(self, x_t: torch.Tensor, feat_state: CausalPyramidState) -> torch.Tensor:
        feats_t = self.features.step(x_t, feat_state)  # (B,K,C)
        return self.nav.step(x_t, feats_t)           # (B,C)


class FixedEmbedding(nn.Module):
    def __init__(self, config, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        W = torch.randn(config.vocab_size, config.n_embd, generator=g)
        # row-center and row-normalize so rows are zero-mean, unit-norm
        W = W - W.mean(dim=1, keepdim=True)
        W = W / (W.norm(dim=1, keepdim=True) + 1e-8)
        self.weight = nn.Parameter(W, requires_grad=False)

    def forward(self, idx):
        return self.weight[idx]

# ---- BlockFast wired for list-in/list-out mixer ----

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 66 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 6
    n_head:int = 6
    n_embd: int = 128
    n_scales:int = 9
    dropout: float = 0.1

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.n_embd = config.n_embd

        self.transformer = nn.ModuleDict(dict(
            wte = FixedEmbedding(config),
            h = nn.ModuleList([GPTSemanticBlock(config) for _ in range(config.n_layer)]),
        ))
       
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)


    # ---------- forward ----------
    def forward(self, idx, targets=None, eprint=False):
        device = idx.device
        b, t = idx.size()
        x = self.transformer.wte(idx) 
        x = x.detach()                 # sever any stale history just in case
        x.requires_grad_(True)         # make x a grad leaf for τ at layer 0

        for block in self.transformer.h:
                x= block(x)
        
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss


    @torch.no_grad()
    def generate_greedy(model: nn.Module, idx: torch.LongTensor, max_new_tokens: int, block_size: int):
        """
        model: your GPT with:
           - transformer.wte (embedding)
           - transformer.h : list[GPTSemanticBlock]
           - lm_head
        idx: (B, T0) prompt token ids
        """
        device = next(model.parameters()).device
        B = idx.size(0)
        # per-block feature caches
        feat_states = [CausalPyramidState(model.config.n_scales, model.config.n_embd, device, batch_size=B)
                       for _ in model.transformer.h]
    
        # 1) prime caches with the prompt (causal, one step at a time)
        x_all = model.transformer.wte(idx)  # (B,T0,C); fixed embeddings in your code
        for t in range(idx.size(1)):
            x_t = x_all[:, t, :]
            for blk, st in zip(model.transformer.h, feat_states):
                x_t = blk.step(x_t, st)      # per-block step
            # we discard logits during priming
    
        # 2) roll out new tokens
        out = [idx]
        cur = idx
        for _ in range(max_new_tokens):
            # last token embedding
            last_idx = cur[:, -1]                      # (B,)
            x_t = model.transformer.wte(last_idx)      # (B,C)
            for blk, st in zip(model.transformer.h, feat_states):
                x_t = blk.step(x_t, st)                # (B,C)
            logits = model.lm_head(x_t)                # (B,V)
            next_idx = torch.argmax(logits, dim=-1, keepdim=True)  # greedy; swap to sampling if you like
            out.append(next_idx)
            cur = torch.cat([cur, next_idx], dim=1)
            # keep only last block_size tokens in cur (typical AR convenience)
            if cur.size(1) > block_size:
                cur = cur[:, -block_size:]
        return torch.cat(out, dim=1)
