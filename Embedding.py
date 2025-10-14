#copyright joshuah.rainstar@gmail.com MIT
from __future__ import annotations
import math
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Tuple


def _norm(v, eps: float = 1e-12):
    return torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(eps)


def _unit(v, eps: float = 1e-12):
    return v / _norm(v, eps)

    
@torch.no_grad()
def phase_transport_between(
    curr: torch.Tensor,
    prev: torch.Tensor,
    tau: float = 1e-6,          # semantic threshold (unchanged)
    eps: float = 1e-12          # numeric epsilon (NEW: decoupled from tau)
) -> torch.Tensor:
    assert curr.shape == prev.shape and curr.dim() == 3
    B, T, C = curr.shape

    # Units (reuse norms) — clamp with eps (NOT tau)
    nu = torch.linalg.vector_norm(curr, dim=-1, keepdim=True).clamp_min(eps)   # (B,T,1)
    nv = torch.linalg.vector_norm(prev, dim=-1, keepdim=True).clamp_min(eps)   # (B,T,1)
    u = curr / nu
    v = prev / nv

    w = curr - prev
    c = (u * v).sum(dim=-1, keepdim=True)                                      # (B,T,1)

    # Masks (semantic thresholds use tau)
    near_pos = (c >  1.0 - tau)                                                # (B,T,1)
    near_neg = (c < -1.0 + tau)                                                # (B,T,1)
    small_u  = (nu < tau)                                                      # (B,T,1)
    small_v  = (nv < tau)                                                      # (B,T,1)
    trivial  = near_pos | small_u | small_v                                    # (B,T,1)

    # General branch
    denom = (1.0 + c).clamp_min(eps)                                           # (B,T,1)
    a = (v * w).sum(dim=-1, keepdim=True)                                      # (B,T,1)
    b = (u * w).sum(dim=-1, keepdim=True)                                      # (B,T,1)
    Kw  = u * a - v * b                                                        # (B,T,C)
    K2w = u * (a * c - b) + v * (b * c - a)                                    # (B,T,C)
    y_gen = w - Kw + (K2w / denom)                                             # (B,T,C)

    # Antipodal candidate
    if C == 1:
        y_neg = -w
    else:
        # Keep this normalization stable with eps as well
        idx = torch.argmin(v.abs().reshape(-1, C), dim=1, keepdim=True)        # (B*T,1)
        s = v.reshape(-1, C).gather(1, idx)                                    # (B*T,1)
        p = -s * v.reshape(-1, C)
        onehot = F.one_hot(idx.squeeze(-1), num_classes=C).to(s.dtype)
        p = p + onehot
        n = torch.linalg.vector_norm(p, dim=1, keepdim=True).clamp_min(eps)
        p = (p / n).view(B, T, C)
        proj_v = (v * w).sum(dim=-1, keepdim=True) * v                         # (B,T,C)
        proj_p = (p * w).sum(dim=-1, keepdim=True) * p                         # (B,T,C)
        y_neg = w - 2.0 * proj_v - 2.0 * proj_p

    # Fuse selections
    y = torch.where(trivial, w, y_gen)
    y = torch.where(near_neg, y_neg, y)
    return y

# ===========================================================
# Multi-scale features (vectorized pyramid)
# ===========================================================
class CausalCentroidPyramid(nn.Module):
    """
    Child-driven centroid pyramid.
    Level s (s=0..K-1) covers window 2^(s+1):
      - child stream z_0 := x
      - y_s(t) = PT(z_s(t), z_s(t - 2^s))             # distance between bracketing child markers
      - z_{s+1}(t) = z_s(t) - 0.5 * y_s(t)            # centerpoint of the window (right-anchored)
    Unsupported prefix t < 2^s is zeroed.
    Returns feats: (B,T,K,C) with feats[:,:,s,:] = y_s.
    """
    def __init__(self, num_scales: int, tau: float = 1e-6):
        super().__init__()
        assert num_scales >= 1
        self.K = num_scales
        self.tau = float(tau)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        mask_early: bool = True,
        return_children: bool = False
    ) -> typing.Union[torch.Tensor, Tuple[torch.Tensor, typing.List[torch.Tensor]]]:
        assert x.dim() == 3
        B, T, C = x.shape

        feats = []
        z = x.clone()
        z_hist = [] if return_children else None

        for s in range(self.K):
            d = 1 << s

            left = torch.zeros_like(z)
            if T > d:
                left[:, d:, :] = z[:, :-d, :].contiguous()

            y = phase_transport_between(z, left, tau=self.tau)  # (B,T,C)
            if mask_early:
                y[:, :d, :].zero_()

            feats.append(y)
            if return_children:
                # Store the child stream used to compute y_s at each t
                z_hist.append(z)

            z = z - 0.5 * y
            if mask_early:
                z[:, :d, :].zero_()

        feats = torch.stack(feats, dim=2)  # (B,T,K,C)
        if return_children:
            return feats, z_hist
        return feats


        # ----- STREAMING STATE FOR INFERENCE -----
class CausalPyramidState:
    """
    Keeps ring buffers of child marker streams z_s for s=0..K-1 with lengths 2^s.
    At step t:
      - z_0(t) = x_t
      - for s: y_s(t) = PT(z_s(t), z_s(t - 2^s)); z_{s+1}(t) = z_s(t) - 0.5*y_s(t)
    """
    def __init__(self, num_scales: int, C: int, device, batch_size: int = 1, tau: float = 1e-6):
        self.K = num_scales
        self.C = C
        self.B = batch_size
        self.device = device
        self.tau = float(tau)
        self.t = 0

        # ring buffers for child streams z_s
        self.buffers = [torch.zeros(batch_size, (1 << s), C, device=device) for s in range(self.K)]
        self.ptrs = [0 for _ in range(self.K)]

    def _read(self, level: int, r: int):
        if self.t < r:
            return torch.zeros(self.B, self.C, device=self.device)
        L = self.buffers[level].size(1)
        idx = (self.ptrs[level] - r) % L
        return self.buffers[level][:, idx, :]

    def _push(self, level: int, value: torch.Tensor):
        L = self.buffers[level].size(1)
        self.buffers[level][:, self.ptrs[level], :] = value
        self.ptrs[level] = (self.ptrs[level] + 1) % L
        
    def reset(self):
        for s in range(self.K):
            self.buffers[s].zero_()
            self.ptrs[s] = 0
        self.t = 0

    @torch.no_grad()
    def step(self, x_t: torch.Tensor) -> torch.Tensor:
        B, C = x_t.shape
        assert B == self.B and C == self.C

        feats = []
        z_t = x_t.clone()  # z_0(t)

        for s in range(self.K):
            d = 1 << s
            left = self._read(level=s, r=d)           # z_s(t-d)
            y_t = phase_transport_between(z_t[:, None, :], left[:, None, :], tau=self.tau).squeeze(1)
            if self.t < d:
                y_t.zero_()
            feats.append(y_t)

            # push current child stream for this level
            self._push(level=s, value=z_t)

            # next level child marker z_{s+1}(t)
            z_t = z_t - 0.5 * y_t
            if self.t < d:
                z_t.zero_()

        self.t += 1
        return torch.stack(feats, dim=1)  # (B,K,C)
    @torch.no_grad()
    def bulk_write_from_block(self, z_histories: typing.List[torch.Tensor], T_block: int):
        """
        z_histories[s]: (B, T_block, C) for z_s(t) over the block.
        Writes the last min(T_block, 2^s) child values into level-s ring buffer,
        sets ptr so that future _read() is correct.
        """
        old_t = self.t
        new_t = old_t + T_block

        for s, z_s in enumerate(z_histories):
            L = self.buffers[s].size(1)         # 2^s
            n = min(T_block, L)
            last = z_s[:, -n:, :]               # (B, n, C)

            # For times i in [new_t - n, new_t - 1], indices are i % L
            perm = torch.remainder(torch.arange(new_t - n, new_t, device=self.device), L)  # (n,)
            # Assign across batch without loops
            self.buffers[s][:, perm, :] = last

            # ptr should point to "next write" position (i.e., new_t % L)
            self.ptrs[s] = new_t % L

        self.t = new_t


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



class FlatRollEmbed(nn.Module):
    """
    Embedding matrix W in R^{V x D} built as:
      W = roll_rows(x) + S
    where x ∈ R^D has |FFT(x)|^2 flat (k=1..D-1) and DC=0, roll_rows(x)[r] = roll(x, r % D),
    and S adds a single spike per row at column (r + M) % D, with M = argmax(x) and scale N = 1/(x[M]+eps).

    Works for any vocab_size V and embedding dim D.
    Gauge-invariant fidelity be damned. Thou shalt not encode knowledge into thine embeddings!
    in strict terms:
    Yes, embeddings are coordinates. Without anchors you have a gauge freedom: if 
    Y is good, so is YQ for any orthogonal Q
    That means “meaning” isn’t tied to fixed axes; it lives only in relative geometry. 
    Great for loss invariance, not great for epistemic stability.
    Should any isometric copy of a solution be equally optimal?
    This is ideal if each downstream task is trained (or fine-tuned) with the embedding.
    But- that means embeddings CANNOT be reliable out of scope.
    W trained on Y will fail on YQ unless you retrain/align(procrustes).
    Make coordinates semantically stable so downstream learners can use fixed projections and 
    avoid rebuilding a long “projection codebook.”
    If you want to grow it later, you can slice it to a smaller vocab now, 
    add more elements later, WITHOUT CHANGING YOUR EXISTING EMEDS.
    In practice:
    Retrieval/top-k neighbors are rotation-invariant, but feature-wise rules, sparse probes, 
    decision boundaries, and any human- or system-interpretable logic are not. 
    They implicitly assume anchored axes.
    Truth is not relative. Truth is positional within a frame. 
    Over time/checkpoints, anchor-free embeddings drift Q/T.
    This means you cant build and distribute models where the embeddings need to adapt to the manifold,
    and use distributed learning- they will wander out of scope.

    FlatRoll’s philosophy is to trade off relational fidelity for coordinate identifiability: 
    equidistant partitioning yields stable, interpretable axes and predictable neighborhoods. 
    That’s epistemically friendly (zero/low re-training, robust to model drift).
    Its exact math offers the *best* distance from any point and mean and simultaneous distinguishability.
    That means that naively, it offers the best temperature packing.
    
    """

    def __init__(self, config, scale: str = "box", seed: int = 0,
                 freeze: bool = True, dtype=None, device=None):
        super().__init__()
        V = int(config.vocab_size)
        D = int(config.n_embd)
        dtype = dtype or torch.float32
        eps = 1e-12

        # base vector x ∈ R^D with flat spectrum, DC=0
        x = self._make_base(D, scale=scale, seed=seed, dtype=dtype, device=device)  # [D]

        # circulant-like rows: row r is x rolled by (r % D)
        # (vectorized loop for clarity; can be optimized further if needed)
        shifts = torch.arange(V, device=device)
        rows = [torch.roll(x, shifts=int(s.item() % D), dims=0) for s in shifts]
        W = torch.stack(rows, dim=0).to(dtype)

        # align a single positive extremum via a "tower" S
        M = int(torch.argmax(x))          # index of max in x
        pm = x[M].item()
        N = 1.0 / (pm + eps)              # safe reciprocal

        # S[r, (r + M) % D] = N
        r_idx = torch.arange(V, device=device)
        c_idx = (r_idx + M) % D
        S = torch.zeros((V, D), dtype=dtype, device=device)
        S[r_idx, c_idx] = N

        mixed = W + S
        self.embed = nn.Embedding.from_pretrained(mixed, freeze=freeze)

    @staticmethod
    def _make_base(D: int, scale: str = "box", seed: int = 0,
                   dtype=torch.float32, device=None) -> torch.Tensor:
        """
        Build x ∈ R^D where |FFT(x)| is flat for k=1..D-1 and DC=0.

        scale:
          - "unit": ||x||_2 = 1
          - "box":  max|x_i| = 1
        """
        # Build on CPU, then move to device at the end.
        # Use complex64 for float/bfloat/half; complex128 otherwise.
        if dtype in (torch.float16, torch.bfloat16, torch.float32):
            complex_dtype = torch.complex64
            work_float = torch.float32
        else:
            complex_dtype = torch.complex128
            work_float = torch.float64

        X = torch.zeros(D, dtype=complex_dtype)

        # DC bin = 0
        X[0] = torch.tensor(0, dtype=complex_dtype)

        if D % 2 == 0:
            # bins 1..D/2-1 are complex-conjugate pairs; Nyquist bin must be real
            for k in range(1, D // 2):
                phi = torch.rand((), dtype=work_float) * (2 * math.pi)
                val = torch.cos(phi).to(work_float) + 1j * torch.sin(phi).to(work_float)
                val = val.to(complex_dtype)
                X[k] = val
                X[D - k] = torch.conj(val)
            # Nyquist bin (real, ±1)
            X[D // 2] = (1.0 if torch.rand(()) < 0.5 else -1.0)
        else:
            for k in range(1, (D - 1) // 2 + 1):
                phi = torch.rand((), dtype=work_float) * (2 * math.pi)
                val = torch.cos(phi).to(work_float) + 1j * torch.sin(phi).to(work_float)
                val = val.to(complex_dtype)
                X[k] = val
                X[D - k] = torch.conj(val)

        x = torch.fft.ifft(X).real  # real length-D base vector (float32/64)
        x = x.to(work_float)

        if scale == "unit":
            x = x / (x.norm() + 1e-12)
        elif scale == "box":
            x = x / (x.abs().max() + 1e-12)
        else:
            raise ValueError("scale must be 'unit' or 'box'")

        x = x.to(dtype)
        if device is not None:
            x = x.to(device)
        return x

    def forward(self, input_ids: torch.LongTensor):
        # (batch, seq_len, D)
        return self.embed(input_ids)

class RandomOrthoprojector(nn.Module):
    """
    Maps R^d -> R^(d//2) via a fixed random orthoprojector (JL-style).
    Preserves Euclidean structure approximately; scales to preserve expected norm.
    Expects config.n_embd (int). Forward accepts (b, c) or (b, t, c).
    """
    def __init__(self, config):
        super().__init__()
        d = int(config.n_embd)
        d2 = d // 2

        # Random orthogonal matrix via QR; take first d//2 rows => orthonormal rows
        G = torch.randn(d, d)
        Q, _ = torch.linalg.qr(G)        # Q: (d, d), orthogonal
        R = Q[:d2, :]                    # (d2, d), rows orthonormal

        # Scale so that E[||Rx||^2] ≈ ||x||^2
        R = R * math.sqrt(d / d2)

        # Store transpose for right-multiplication on (..., d)
        self.register_buffer("proj_T", R.t())  # (d, d2), no grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        P = self.proj_T.to(dtype=x.dtype, device=x.device)
        # Supports (b, c) or (b, t, c); matmul broadcasts on the trailing dim
        return x @ P
        
class FeatureDistiller(nn.Module):
    """
    Uses a single RandomOrthoprojector to downproject:
      - x: (..., C) -> (..., D)
      - feats: (..., K, C) -> (..., K, D)
    Same weights; fully parallel/broadcasted matmul.
    """
    def __init__(self, config):
        super().__init__()
        self.proj = RandomOrthoprojector(config)  # C -> D = C//2

    def forward(
        self,
        x: torch.Tensor,         # (..., C)
        feats: torch.Tensor      # (..., K, C)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        P = self.proj.proj_T.to(dtype=x.dtype, device=x.device)  # (C, D)
        x_down = x @ P                                           # (..., D)
        feats_down = feats @ P                                   # (..., feats.K, D)
        return x_down, feats_down

class TimeEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.init_embed is not None
        assert config.init_embed >= config.vocab_size
        assert config.block_size is not None
        assert config.n_scales is not None
        

        self.config = config

        self.C = int(config.init_embed)
        self.K = int(config.n_scales)
        self.D = self.C // 2
        self.width = (self.K + 1) * self.D    #x,each feature, efficiently and correctly reduced
        
        # concat of x_down (D) + K*D
        assert config.n_embd == width #final embeddings must equal the width returned

        # Modules
        self.wte = FlatRollEmbed(config)
        self.features = SemanticClusterFeaturesCausal(num_scales=self.K, tau=1e-6)
        self.distiller = FeatureDistiller(config)

        # Streaming state (batch_size must be consistent for step() usage)
        device = next(self.parameters()).device
        B = int(getattr(config, "batch_size", 1))
        self.feat_states = CausalPyramidState(
            num_scales=self.K,
            C=self.C,
            device=device,
            batch_size=B,
            tau=1e-6
        )

    def _fuse_block(self, x: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        """
        x:     (B, T, C)
        feats: (B, T, K, C)
        -> (B, T, (K+1)*D)
        """
        x_down, feats_down = self.distiller(x, feats)         # (B,T,D), (B,T,K,D)
        B, T, _, = x.shape
        fused = torch.cat([x_down, feats_down.reshape(B, T, -1)], dim=-1)
        return fused

    def _fuse_step(self, x_t: torch.Tensor, feats_t: torch.Tensor) -> torch.Tensor:
        """
        x_t:     (B, C)
        feats_t: (B, K, C)
        -> (B, (K+1)*D)
        """
        x_down, feats_down = self.distiller(x_t, feats_t)     # (B,D), (B,K,D)
        fused = torch.cat([x_down, feats_down.reshape(x_t.size(0), -1)], dim=-1)
        return fused
        
    def reset_feats(self):
        self.feat_states.reset()

    def forward(self, idx: torch.LongTensor) -> torch.Tensor:
        """
        Rules:
          - If idx is (B, T) [block]:
              * training mode: vectorized feats; DO NOT update streaming state
              * eval mode:     vectorized feats; bulk-update streaming state
          - If idx is (B,)    [single-step]:
              * always treat as online: use state.step(...) and update state
        """
        device = next(self.parameters()).device

        if idx.ndim == 2:
            # BLOCK PATH: (B, T)
            B, T = idx.shape
            x = self.wte(idx.to(device))                         # (B, T, C)

            if self.training:
                # Training: compute feats; don't populate state
                feats = self.features(x)                         # (B, T, K, C)
                return self._fuse_block(x, feats)                # (B, T, (K+1)*D)
           else:
                B, T = idx.shape
                outs = []
                for t in range(T):
                    x_t = self.wte(idx[:, t].to(device))                 # (B, C)
                    feats_t = self.features.step(x_t, self.feat_states)  # (B, K, C)  <-- uses cache
                    outs.append(self._fuse_step(x_t, feats_t))           # (B, (K+1)*D)
                return torch.stack(outs, dim=1)                          # (B, T, (K+1)*D)

        elif idx.ndim == 1:
            # SINGLE-STEP PATH: (B,)
            B = idx.shape[0]
            x_t = self.wte(idx.to(device))                       # (B, C)
            feats_t = self.features.step(x_t, self.feat_states)  # (B, K, C)
            return self._fuse_step(x_t, feats_t)                 # (B, (K+1)*D)

        else:
            raise ValueError("idx must be 1D (B,) or 2D (B,T)")

#goals:
#you should be feeding 1 at a time and learning online.
#if your model cannot learn online then it is a garbage design.
#my recommendation- supplement embeddings with planning information.
#that means initialize another embedding instance and reverse the sequence of idx,
#then reverse it again, then roll it forward by one, then drop off the bottom x chunk,
#then train aux loss against this forward knowledge on an RNN at each level that is fed past and current step product,
#then integrate rnn product as aux info into next step. try to not only predict next state,
#but build something better and recursively plan states top to bottom and then bottom to top.
#that way your convergence is on choosing the future from options by intelligent refinement.
#for active working memory, current approaches are not the solution-
#think about it. your model is planning and updating a trajectory through a learned latent universe.
#it decides what to say the moment it says it, and cant recursively redirect itself on latent discovery.
#it CAN and must redirect itself, thats not what im saying. im saying its a fragile system.
#a better approach can and will be leaned memnonic recall using shorthand encoding.
#model must learn to store a shorthand summary efficiently using a token to call out- forced storage-
#and then must learn to periodically invoke it into state(self-grounding) and periodically update it.
#of course this also means our token-sequential speed must go up so we can handle more tokens.
#as far as memory, statefulness, statelessness, no model remembers the entire conversation, or even part of it.
#model can only know present point in manifold, vectors pointing back, vectors? maybe pointing forward.
#EVEN with attention, EVEN with RAG, EVEN with anything else, if model is autoregressively generated,
#model only knows about this instant. it is properties of what is encoded IN this instant that contain all previous instants.
#so, embeddings at every point MUST contain the entire semantic flow- or they contain none of it.


HOW TO USE THIS METHOD:
#drop this in over NN.embedding.
#set config.init_embed = your true embedding. set config.n_embed 
#(n_scales + 1) * init_embed//2, or slice at end of model.
#recommend n_scales +1 = n_heads if using attention, and a fat block size.
#however, if using streaming flow, possibly try higher n_scales- OR
#use learned fir coeffs,wavelet accumulators to take rolling update summary on each scale,
#and model learns semantic units and memory that way.
#use reset_feats between starting prompts and train/eval starting.
