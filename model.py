#copyright joshuah.rainstar@gmail.com 2025 
#MIT License terms apply
#the only challenge remaining in terms of the base transformer architecture(this is an optimal design)
#is the MLP. MLP = designed to learn functions.
#the role of an MLP is, however, a learned codebook. 
#now, we believe one could massively scale up blocks- 
#and that one could sigmoid route blocks, or use similar expert selection mechanisms.
#but it can't be MLP. it has to be the entire block.
#note: lots of optimization needed to get this attention to behave on cuda and fast.
#testing only done on an m4 mac

from __future__ import annotations
import math
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

class PairwiseRotSpiral(nn.Module):
    def __init__(self, dim, radius=6.0, omega=1.0, k=1.0, step=0.1, cube_shell=False):
        super().__init__()
        self.dim = dim
        self.radius = float(radius)
        self.omega = float(omega)
        self.k = float(k)
        self.step = float(step)
        self.cube_shell = bool(cube_shell)
        self.eps = 1e-8

    def _cos_sin(self, x):
        theta = self.omega * self.step
        # Use Python math for scalar, then create tensors on correct device and dtype
        c = torch.tensor(math.cos(theta), device=x.device, dtype=x.dtype)
        s = torch.tensor(math.sin(theta), device=x.device, dtype=x.dtype)
        return c, s

    def forward(self, x):
        D = x.size(-1)
        # radial term
        r = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(self.eps)
        radial = (self.radius - r) * (x / r)

        # rotation on 2D pairs, vectorized
        if D >= 2:
            c, s = self._cos_sin(x)
            n2 = D // 2
            head = x[..., : n2 * 2].reshape(*x.shape[:-1], n2, 2)
            xi = head[..., 0]
            xj = head[..., 1]
            yi = c * xi - s * xj
            yj = s * xi + c * xj
            rot = torch.stack([yi, yj], dim=-1).reshape(*x.shape[:-1], n2 * 2)
            if D % 2 == 1:
                y = torch.cat([rot, x[..., -1:].contiguous()], dim=-1)
            else:
                y = rot
        else:
            y = x

        # one-step Euler update
        y = x + self.step * ((y - x) + self.k * radial)

        if self.cube_shell:
            y = self.radius * torch.tanh(y / self.radius)
        return y



class SpiralMix(nn.Module):
    def __init__(self, rank, **spiral_kwargs):
        super().__init__()
        self.rank = rank
        self.flow = PairwiseRotSpiral(rank, **spiral_kwargs)

    def forward(self, comps, center=None, loop_iters=2):
        # Accept either a list/tuple of [...,] Tensors or a single Tensor [..., r]
        if isinstance(comps, (list, tuple)):
            # old DynMix API: list of [B,T] or [B] -> stack on last dim -> [B,T,r] (or [B,r])
            x = torch.stack(comps, dim=-1)
            return_list = True
        else:
            # new API: comps is already [B,T,r] (or any leading dims, last is r)
            x = comps
            return_list = False

        if center is None:
            center = 0.0  # broadcastable scalar OK
        y = x
        for _ in range(loop_iters):
            y = self.flow(y - center) + center  # pairwise rotations on last dim only

        if return_list:
            # match DynMix return type: list of [...,] tensors
            return [y[..., i] for i in range(y.size(-1))]
        return y


class Coop5(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _softsign(x: torch.Tensor) -> torch.Tensor:
        return x / (1.0 + x.abs())

    @staticmethod
    def _zls(x: torch.Tensor) -> torch.Tensor:
        sp = F.softplus(x)
        sa = torch.sigmoid(0.5 * x)
        ba = sa * (1.0 - sa)
        return sp - 2.77258872223978123766 * ba  # 4*ln(2)

    @staticmethod
    def forward(R: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Similarity gating
        w = Coop5._zls(Coop5._softsign(R * C).sum(dim=-1, keepdim=True) / (2 * R.size(-1) ** 0.5))
        # RK2-style update
        k1 = w * (C - R)
        k2 = w * (C - (R + 0.5 * k1))
        return R + 0.25 * (k1 - k2)

    '''#may offer better effect, but more costly
    @staticmethod
    def forward(R: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Similarity gating
        w = Coop5._zls(Coop5._softsign(R * C).sum(dim=-1, keepdim=True) / (2 * R.size(-1) ** 0.5))
        # RK2-style update
        k1 = w * (C - R)
        # Midpoint
        R_mid = R + 0.5 * k1
        w2 = Coop5._zls(Coop5._softsign(R_mid * C).sum(dim=-1, keepdim=True) / (2 * R.size(-1) ** 0.5))
        k2 = w2 * (C - R_mid)
        return R + 0.25 * (k1 + k2)
    '''


class DynMix(nn.Module):
    def __init__(self):
        super().__init__()
        self.coop = Coop5()

    def mix_list(self,xs):
        n = len(xs)
        stacked = torch.stack(xs, 0)
        total = stacked.sum(0)
        out = []
        for i in range(n):
            others_mean = (total - stacked[i]) / (n - 1)
            out.append(self.coop(stacked[i], others_mean))
        return out

    def forward(self, comps, loop_iters: int = 2):
        for _ in range(loop_iters):
            comps = self.mix_list(comps)
        return comps
        

class PhaseTap(nn.Module):
    """
    Phase-preserving vector shift with guarded Householder.
    x: (B,T,C) -> y: (B,T,C)
      - t < d:  y[:, t, :] = (1/(d - t)) * a
      - t >= d: y[:, t, :] = H(x_t)^T @ (x_t - x_{t-d})
    Guards:
      - near u_t ≈ a: skip reflection, use identity on v
      - near u_t ≈ -a: use fixed orthonormal b
      - near zero ||x_t||: skip reflection
    """
    def __init__(self, d: int, tau: float = 1e-6):  # ?1 tau
        super().__init__()
        assert isinstance(d, int) and d >= 1
        self.d = d
        self.tau = float(tau)

    @staticmethod
    def _norm(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return torch.linalg.vector_norm(v, dim=-1).clamp_min(eps)

    @staticmethod
    def _safe_unit(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        n = torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(eps)
        return v / n

    def _apply_householder_sym(self, a: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Apply H v with H a = u, H = I - 2 w w^T, symmetric so H^T = H.
        a,u,v: (..., C)
        Guards near a, near -a, and near zero u.
        """
        C = a.shape[-1]
        # masks
        dot = (a * u).sum(dim=-1, keepdim=True)                      # (...,1)
        near_pos = (dot > 1.0 - self.tau).squeeze(-1)                # (...)
        near_neg = (dot < -1.0 + self.tau).squeeze(-1)               # (...)
        near_zero_u = (torch.linalg.vector_norm(u, dim=-1) < self.tau)  # (...)

        y = v.clone()

        # general case mask
        gen = ~(near_pos | near_neg | near_zero_u)
        if gen.any():
            w = self._safe_unit(a[gen] - u[gen])
            wTv = (w * v[gen]).sum(dim=-1, keepdim=True)
            y[gen] = v[gen] - 2.0 * w * wTv

        # near -a: reflect across fixed b orthonormal to a
        if near_neg.any():
            if C == 1:
                y[near_neg] = -v[near_neg]
            else:
                b = torch.zeros_like(a[near_neg])
                b[..., 1] = 1.0
                bbT_v = (b * v[near_neg]).sum(dim=-1, keepdim=True)
                y[near_neg] = v[near_neg] - 2.0 * b * bbT_v

        # near +a or near zero u: identity on v
        # y[near_pos] and y[near_zero_u] already equal v by init

        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3, "x must be (B,T,C)"
        B, T, C = x.shape
        device = x.device
        dtype = x.dtype

        y = torch.zeros_like(x)

        # anchor a = e0
        a = torch.zeros(B, 1, C, device=device, dtype=dtype)
        a[..., 0] = 1.0

        # early baseline
        if self.d > 0:
            t_idx = torch.arange(T, device=device)
            early_mask = t_idx < self.d
            if early_mask.any():
                denom = (self.d - t_idx[early_mask]).to(dtype=dtype)
                y[:, early_mask, :] = a.expand(B, early_mask.sum(), C) * denom.unsqueeze(0).reciprocal().unsqueeze(-1)

        if T <= self.d:
            return y

        # main region
        x_t  = x[:, self.d:, :]          # (B,T-d,C)
        x_tm = x[:, :-self.d, :]         # (B,T-d,C)
        u_t  = self._safe_unit(x_t)      # (B,T-d,C)

        a_bt = a.expand(B, x_t.shape[1], C)
        v    = x_t - x_tm

        if C == 1:
            y[:, self.d:, :] = v
            return y

        y[:, self.d:, :] = self._apply_householder_sym(a_bt, u_t, v)
        return y
        
class PhaseTransport(nn.Module):
    def __init__(self, d: int, tau: float = 1e-6):
        super().__init__()
        assert isinstance(d, int) and d >= 1
        self.d = d
        self.tau = float(tau)

    @staticmethod
    def _norm(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return torch.linalg.vector_norm(v, dim=-1).clamp_min(eps)

    @staticmethod
    def _safe_unit(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        n = torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(eps)
        return v / n

    @staticmethod
    def _orthonormal_perp(v: torch.Tensor) -> torch.Tensor:
        # v: (N, C) assumed nonzero
        N, C = v.shape
        idx = torch.argmin(torch.abs(v), dim=-1)      # pick coord with smallest magnitude
        e = torch.zeros_like(v)
        e.scatter_(1, idx.unsqueeze(1), 1.0)
        p = e - (e * v).sum(dim=-1, keepdim=True) * v # Gram-Schmidt
        p = p / PhaseTransport._norm(p).unsqueeze(-1)
        return p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3, "x must be (B,T,C)"
        B, T, C = x.shape
        device, dtype = x.device, x.dtype
        y = torch.zeros_like(x)

        # early baseline with per-sequence direction, not a global axis
        if T > 0:
            ref_t = min(self.d, T - 1)
            u_ref = self._safe_unit(x[:, ref_t, :])  # (B, C)
            if self.d > 0:
                t_idx = torch.arange(T, device=device)
                early_mask = t_idx < self.d
                if early_mask.any():
                    denom = (self.d - t_idx[early_mask]).to(dtype=dtype)     # (Te,)
                    scales = (1.0 / denom).view(1, -1, 1)                    # (1, Te, 1)
                    y[:, early_mask, :] = u_ref.view(B, 1, C) * scales       # (B, Te, C)

        if T <= self.d:
            return y

        # main region t >= d
        xt  = x[:, self.d:, :]             # (B, T-d, C)
        xtm = x[:, :-self.d, :]            # (B, T-d, C)
        u   = self._safe_unit(xt)          # (B, T-d, C)
        v   = self._safe_unit(xtm)         # (B, T-d, C)
        w   = xt - xtm                      # (B, T-d, C)

        c = (u * v).sum(dim=-1, keepdim=True)          # (B, T-d, 1)
        # squeeze masks to (B, T-d)
        near_pos = (c > 1.0 - self.tau).squeeze(-1)
        near_neg = (c < -1.0 + self.tau).squeeze(-1)
        small_u  = (torch.linalg.vector_norm(xt,  dim=-1) < self.tau)
        small_v  = (torch.linalg.vector_norm(xtm, dim=-1) < self.tau)
        trivial  = near_pos | small_u | small_v

        y_main = w.clone()

        # general case
        gen = ~(trivial | near_neg)
        if gen.any():
            u_g = u[gen]                       # (N, C)
            v_g = v[gen]
            w_g = w[gen]
            c_g = c[gen].unsqueeze(-1)[:, 0, :]  # (N, 1) ensure 2D
            alpha = 1.0 / (1.0 + c_g)          # (N, 1)

            a = (v_g * w_g).sum(dim=-1, keepdim=True)  # v·w
            b = (u_g * w_g).sum(dim=-1, keepdim=True)  # u·w
            Kw  = u_g * a - v_g * b
            K2w = u_g * (a * c_g - b) + v_g * (b * c_g - a)
            y_main[gen] = w_g - Kw + alpha * K2w

        # antipodal 180 deg
        if near_neg.any():
            v_n = v[near_neg]                 # (N, C)
            w_n = w[near_neg]
            p   = self._orthonormal_perp(v_n) # (N, C)
            proj_v = (v_n * w_n).sum(dim=-1, keepdim=True) * v_n
            proj_p = (p   * w_n).sum(dim=-1, keepdim=True) * p
            y_main[near_neg] = w_n - 2.0 * proj_v - 2.0 * proj_p

        y[:, self.d:, :] = y_main
        return y
        
class Cell(nn.Module):
    def __init__(self, dim_in: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, hidden, bias=False) #
        torch.nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')

        self.fc2 = nn.Linear(hidden, dim_in, bias=True)
        torch.nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')

        self.act = nn.Softplus()
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x))-1)   

class GRUStyleRefinementAttention(nn.Module):
    """
    One-shot GRU-like refinement attention with hierarchical causal synthesis
    (MuSe × FMMformer) and mild alignment noise (no aux loss).

    Pipeline (single forward pass)
    ------------------------------
    0) Projections Q,K,V; multi-head split; LASER on V (exp).
    1) Seed (fast/low-rank): rank-r EA + Sigmoid + LASER => H0. (unchanged)
    2) Full attention = Near + Far (causal, hierarchical):
        • Near-field (level 0): FMMformer-style banded causal blocks of width k0.
        • Far-field (levels 1..L): per level ℓ, build causal blocks via a segment tree.
            - Mild aligner (standardize + alignment-conditioned noise) per block/head.
            - MuSe-style independent clustering of Q and K within the block.
            - Centroid attention with EA + Sigmoid + LASER, bias b=-log(n_ℓ).
            - Dipole correction per K-cluster (linear logit augmentation).
            - Causal aggregation over past (ancestor/previous) blocks only.
        → Sum near + all far levels to get 	ilde H.
    3) Head-wise gating (sigmoid) on 	ilde H.
    4) GRU blend: H_out = (1 - z) * H0 + z * 	ilde H.

    Notes
    -----
    • No softmax anywhere: EA ((QK^T/√d)^2) + elementwise sigmoid attention + LASER(V).
    • No Q/K invariance tricks; we rely on mild aligner to stabilize geometry.
    • Causality enforced at all stages (banded near; triangular past-only for far).

    Config (dict)
    -------------
    Required:
      - n_embd (int), n_heads (int)
    Optional (defaults in __init__):
      - block_size (int): not required; used only to cap k0 if provided.
      - k0 (int): near-field band width (default 128)
      - levels (int): #hierarchy levels for far-field (default 2)
      - clusters_per_level (int): C clusters per level per block (default 8)
      - dipole_weight (float): weight for dipole logit augmentation (default 0.25)
      - use_alignment_noise (bool): enable mild aligner (default True)
      - sigma_min, sigma_max, lam, gamma, eps: aligner params
    """

    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_heads = config.n_head
        assert self.n_embd % self.n_heads == 0, "n_embd must be divisible by n_heads"
        self.head_dim = self.n_embd // self.n_heads

        # --- Projections ---
        self.Wq = nn.Linear(self.n_embd, self.n_embd)
        self.Wk = nn.Linear(self.n_embd, self.n_embd)
        self.Wv = nn.Linear(self.n_embd, self.n_embd)
        self.Wo = nn.Linear(self.n_embd, self.n_embd)

        # --- Low-rank seed projections (rank r) ---
        self.rank = max(16, self.head_dim // 2)
        self.Pq = nn.Linear(self.n_embd, self.rank, bias=False)
        self.Pk = nn.Linear(self.n_embd, self.rank, bias=False)
        self.Pv = nn.Linear(self.n_embd, self.rank, bias=False)
        self.up_seed = nn.Linear(self.rank, self.n_embd, bias=False)

        # --- Head-wise gating after refinement ---
        self.gate_u = nn.Parameter(torch.randn(self.n_heads, self.head_dim))  # (H, D)
        self.gate_b = nn.Parameter(torch.zeros(self.n_heads))

        # --- GRU-style update gate ---
        self.Wz = nn.Linear(self.n_embd * 4, self.n_embd)
        self.use_reset_gate = False
        self.Wr = nn.Linear(self.n_embd * 3, self.n_embd)

        # --- Hierarchy & clustering hyperparams ---
        self.k0 = 128
        self.levels = 2
        self.C = 8
        self.dipole_weight = 0.25

        # --- Alignment noise (mild aligner) ---
        self.use_alignment_noise =True
        self.sigma_min = 0.0
        self.sigma_max = 1e-2
        self.lam = 5e-3
        self.gamma = 1.5
        self.eps = 1e-4

        # Numerics
        self.neg_inf = -1e9

    # --------------------------- utils ---------------------------
    @staticmethod
    def _shape_heads(x, n_heads):
        B, T, C = x.shape
        D = C // n_heads
        return x.view(B, T, n_heads, D).transpose(1, 2)  # (B,H,T,D)

    @staticmethod
    def _merge_heads(x):
        B, H, T, D = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * D)

    @staticmethod
    def _causal_mask(T: int, device: torch.device):
        return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))

    @staticmethod
    def _band_mask(T: int, device: torch.device, k0: int):
        idx = torch.arange(T, device=device)
        diff = idx[None, :] - idx[:, None]
        # i >= j and within band k0
        return (diff <= 0) & (diff >= -k0)

    @staticmethod
    def _segment_tree_blocks(T: int, levels: int) -> List[List[Tuple[int, int]]]:
        """Return list of levels, each a list of (start,end) half-open index blocks covering [0,T).
        Level 0 is the whole sequence split into 2^0 blocks? We define:
          - level 0: fine (near handled separately)
          - levels 1..L: 2^ℓ blocks of equal size (last block may be shorter)
        """
        blocks_per_level: List[List[Tuple[int, int]]] = []
        for ell in range(1, levels + 1):
            num = 2 ** ell
            size = max(1, (T + num - 1) // num)
            level_blocks = []
            s = 0
            for _ in range(num):
                e = min(T, s + size)
                if s < e:
                    level_blocks.append((s, e))
                s = e
                if s >= T:
                    break
            blocks_per_level.append(level_blocks)
        return blocks_per_level

    @staticmethod
    def _kmeans_simple(X: torch.Tensor, C: int, iters: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        """Very small k-means (no-grad). X: (N,D). Returns (centroids[C,D], assign[N])."""
        N, D = X.shape
        C = min(C, max(1, N))
        # init: random subset (deterministic by seeding on N,D,C)
        with torch.no_grad():
            idx = torch.linspace(0, N - 1, steps=C, device=X.device).round().long()
            cent = X[idx].clone()
            for _ in range(iters):
                # assign
                dists = torch.cdist(X, cent, p=2)  # (N,C)
                assign = dists.argmin(dim=1)
                # update
                for c in range(C):
                    mask = (assign == c)
                    if mask.any():
                        cent[c] = X[mask].mean(dim=0)
            # final assign
            dists = torch.cdist(X, cent, p=2)
            assign = dists.argmin(dim=1)
        return cent, assign

    def _standardize(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # per-feature zero-mean, unit-variance (diag) with eps
        mu = X.mean(dim=0, keepdim=True)
        Xc = X - mu
        var = Xc.pow(2).mean(dim=0) + self.eps
        Xw = Xc / var.sqrt()
        return Xw, mu, var

    def _align_noise(self, Q_blk: torch.Tensor, K_blk: torch.Tensor, head_seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute standardized Q,K and inject alignment-conditioned Gaussian noise (same at train/infer
        given deterministic RNG seeded by head_seed). Returns (Qw_noisy, Kw_noisy). Shapes: (t,D).
        Uses cosine similarity between mean-standardized features to define misalignment.
        """
        # standardize (diag)
        with torch.no_grad():
            Qw, _, _ = self._standardize(Q_blk)  # (tq, D)
            Kw, _, _ = self._standardize(K_blk)  # (tk, D)
            # Cosine similarity between per-block mean feature vectors
            mq = Qw.mean(dim=0)
            mk = Kw.mean(dim=0)
            nq = torch.norm(mq) + 1e-8
            nk = torch.norm(mk) + 1e-8
            rho = torch.dot(mq, mk) / (nq * nk)  # in [-1, 1]
            rho = torch.clamp(rho, -1.0, 1.0)
            e = (1.0 - rho) * 0.5  # [0,1]
            sigma2 = self.sigma_min ** 2 + self.lam * (float(e) ** self.gamma)
            sigma2 = float(min(self.sigma_max ** 2, max(self.sigma_min ** 2, sigma2)))
        # deterministic noise
        g = torch.Generator()
        g.manual_seed(int(head_seed))
        noise_q = torch.normal(mean=0.0, std=math.sqrt(sigma2), size=Q_blk.shape, generator=g, device=Q_blk.device)
        noise_k = torch.normal(mean=0.0, std=math.sqrt(sigma2), size=K_blk.shape, generator=g, device=K_blk.device)
        return Qw + noise_q, Kw + noise_k

    # --------------------------- forward ---------------------------
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        assert C == self.n_embd
        device = x.device
        eps = 1e-30
    
        # --- Projections ---
        Q = self.Wq(x)                           # (B,T,C)
        K = self.Wk(x)                           # (B,T,C)
        V = self.Wv(x)                           # (B,T,C)
    
        # --- Multi-head split ---
        Qh = self._shape_heads(Q, self.n_heads)  # (B,H,T,D)
        Kh = self._shape_heads(K, self.n_heads)  # (B,H,T,D)
        Vh = self._shape_heads(V, self.n_heads)  # (B,H,T,D)
    
        # --- Masks & global bias ---
        causal = self._causal_mask(T, device) if attn_mask is None else attn_mask.to(torch.bool)  # (T,T)
        k0 = min(self.k0, T - 1)
        near_mask = self._band_mask(T, device, k0) & causal                                       # (T,T)
        b_scalar = -math.log(max(T, 1))
    
        # ---------------- 1) Seed: low-rank EA + Sigmoid + LASER (stable) -> H0 ----------------
        # Low-rank projections
        Qr = self.Pq(Q)                     # (B,T,r)
        Kr = self.Pk(K)                     # (B,T,r)
        Vr = self.Pv(V)                     # (B,T,r)
    
        # EA logits (square) with scaling
        S_seed = (Qr @ Kr.transpose(1, 2))                   # (B,T,T)
        S_seed.mul_(1.0 / math.sqrt(self.rank)).pow_(2)      # ((·)/√r)^2
        S_seed = S_seed.masked_fill(~causal[None, :, :], self.neg_inf)
        A_seed = torch.sigmoid(S_seed + b_scalar)            # (B,T,T)
    
        # LASER on seed values: per-query masked max over keys (causal), then log-sum-exp
        # Build (B,T,T,r) by expanding keys along a query axis
        Vr_keys = Vr[:, None, :, :].expand(-1, T, -1, -1)    # (B,T,T,r)
        neg_inf_seed = torch.finfo(Vr.dtype).min
        mask_seed = causal[None, :, :, None]                 # (1,T,T,1)
        Vr_masked = Vr_keys.masked_fill(~mask_seed, neg_inf_seed)
        m_seed = Vr_masked.amax(dim=2, keepdim=True)         # (B,T,1,r)
        Yr = (A_seed[..., None] * torch.exp(Vr_keys - m_seed)).sum(dim=2)  # (B,T,r)
        H0_r = torch.log(Yr.clamp_min(eps)) + m_seed.squeeze(2)            # (B,T,r)
        H0 = self.up_seed(H0_r)                                            # (B,T,C)
    
        # ---------------- 2) Full attention (NEAR path before FAR loop) ----------------
        # Cosine EA logits for near band
        tau = getattr(self, "tau", 0.35)
        Qn = F.normalize(Qh, dim=-1)
        Kn = F.normalize(Kh, dim=-1)
        cos = torch.matmul(Qn, Kn.transpose(-2, -1))            # (B,H,T,T) in [-1,1]
        logits_near = (cos * cos) / tau
        logits_near = logits_near.masked_fill(~near_mask[None, None, :, :], self.neg_inf)
        A_near = torch.sigmoid(logits_near + b_scalar)          # (B,H,T,T)
    
        # LASER (stable) for near: causal sliding max over last k0+1 keys (fast, no 5D)
        # compute per (B,H,T,D) max over keys in window [i-k0, i]
        kernel = k0 + 1
        neg_inf_near = torch.finfo(Vh.dtype).min
        V_perm = Vh.permute(0, 1, 3, 2)                         # (B,H,D,T)
        V_pad  = F.pad(V_perm, (kernel - 1, 0), value=neg_inf_near)  # left pad with -inf
        V_unf  = V_pad.unfold(-1, kernel, 1)                    # (B,H,D,T,kernel)
        m_near = V_unf.max(dim=-1).values.permute(0, 1, 3, 2)   # (B,H,T,D)
    
        V_shift = Vh - m_near                                   # (B,H,T,D)
        V_exp   = torch.exp(V_shift)                            # (B,H,T,D)
        Y_near  = torch.matmul(A_near, V_exp)                   # (B,H,T,D)
        H_near_h = torch.log(Y_near.clamp_min(eps)) + m_near    # (B,H,T,D)
    
        # Prepare FAR accumulator and blocks
        H_far_total_h = torch.zeros_like(H_near_h)              # (B,H,T,D)
        blocks_per_level = self._segment_tree_blocks(T, self.levels)
    
        # ----- FAR LOOP STARTS HERE -----
        for ell, level_blocks in enumerate(blocks_per_level, start=1):
            # per-level bias uses block-level size for stability
            for b in range(B):
                for h in range(self.n_heads):
                    Q_all = Qh[b, h]                            # (T,D)
                    K_all = Kh[b, h]                            # (T,D)
                    # raw values; we will shift/exp per-block where needed
                    for (s, e) in level_blocks:
                        if s >= e:
                            continue
    
                        # Keys: all j < s minus the recent k0 window (far-only region)
                        j_end = s
                        j_start = 0
                        if j_end <= 0:
                            continue
                        key_idx_full = torch.arange(j_start, j_end, device=device)
                        if k0 > 0:
                            key_idx_far = key_idx_full[:-k0] if j_end - j_start > k0 else key_idx_full[:0]
                        else:
                            key_idx_far = key_idx_full
                        if key_idx_far.numel() == 0:
                            continue
    
                        # Blocks
                        q_blk = Q_all[s:e]                      # (tq, D)
                        k_blk = K_all[key_idx_far]              # (tk, D)
                        V_keys = Vh[b, h][key_idx_far]          # (tk, D) raw V
    
                        # --- mild aligner (optional) ---
                        if self.use_alignment_noise:
                            head_seed = (b + 1) * 1_000_003 + (h + 1) * 911 + (ell + 1) * 97 + (s + 1)
                            q_std, k_std = self._align_noise(q_blk, k_blk, head_seed=head_seed)
                        else:
                            q_std, _, _ = self._standardize(q_blk)
                            k_std, _, _ = self._standardize(k_blk)
    
                        # --- independent clustering (MuSe-style) ---
                        Cq = min(self.C, max(1, q_std.size(0)))
                        Ck = min(self.C, max(1, k_std.size(0)))
                        with torch.no_grad():
                            cq, q_assign = self._kmeans_simple(q_std, Cq, iters=2)
                            ck, k_assign = self._kmeans_simple(k_std, Ck, iters=2)
    
                        one_hot = F.one_hot(k_assign, num_classes=Ck).to(V_keys.dtype)  # (tk, Ck)
                        count = one_hot.sum(dim=0).clamp_min(1.0)                        # (Ck,)
    
                        # dipole (optional)
                        ck_assigned = ck[k_assign]                 # (tk, D)
                        resid = k_std - ck_assigned                # (tk, D)
                        dipole = (one_hot.T @ resid) / count[:, None]
    
                        # cosine EA logits over centroids
                        qcn = F.normalize(q_std, dim=-1)
                        ckn = F.normalize(ck, dim=-1)
                        S = (qcn @ ckn.t())                        # (tq, Ck) in [-1,1]
                        S = (S * S) / tau
                        if self.dipole_weight != 0.0:
                            S = S + self.dipole_weight * ((q_std @ dipole.t()) / math.sqrt(self.head_dim))
    
                        # sigmoid attention with per-level bias
                        b_level = -math.log(max(ck.size(0), 1))
                        A = torch.sigmoid(S + b_level)             # (tq, Ck)
    
                        # LASER (stable) in far block: shift -> exp -> cluster -> sum -> log + m
                        m_blk = V_keys.max(dim=0, keepdim=True).values          # (1, D)
                        V_exp_blk = torch.exp(V_keys - m_blk)                   # (tk, D)
                        v_cent_exp = (one_hot.T @ V_exp_blk) / count[:, None]   # (Ck, D)
                        Y_blk = A @ v_cent_exp                                  # (tq, D)
                        H_blk = torch.log(Y_blk.clamp_min(eps)) + m_blk         # (tq, D)
    
                        H_far_total_h[b, h, s:e] += H_blk
    
        # ----- FAR LOOP ENDS HERE -----
    
        # Sum near and far
        H_refine_h = H_near_h + H_far_total_h                                  # (B,H,T,D)
    
        # ---------------- 3) Head-wise gated attention on refined path ----------------
        gate_scores = torch.einsum('bhtd,hd->bht', Qh, self.gate_u) + self.gate_b[None, :, None]
        H_refine_h = H_refine_h * torch.sigmoid(gate_scores)[..., None]
        Ht = self._merge_heads(H_refine_h)                                      # (B,T,C)
    
        # ---------------- 4) GRU-style blend ----------------
        if self.use_reset_gate:
            r = torch.sigmoid(self.Wr(torch.cat([Q, K, H0], dim=-1)))
            Ht = r * Ht
    
    
        g = self.Wz(torch.cat([Q, K, H0, Ht], dim=-1))
        #z = 0.5 * (torch.tanh(g) + 1.0)     # smoother than sigmoid but still (0,1)
        #H_out = (1.0 - z) * H0 + z * Ht
    
        #return self.Wo(H_out)
        z = torch.sigmoid(g)
        H_out = (1.0 - z) * H0 + z * Ht

        return self.Wo(H_out)






class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln = nn.LayerNorm(config.n_embd)
        self.encoder  = Cell(config.n_embd,config.n_embd*2)
        self.distance_encoder_learned = PhaseTransport(1) #PhaseTap(d)
        self.decoder  = Cell(config.n_embd,config.n_embd*2)

        self.attn = GRUStyleRefinementAttention(config)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        B,T,C= x.shape
        w = self.ln(x)
        a = self.distance_encoder_learned(w)
        x = x + self.dropout(self.encoder(a))
        x = x + self.dropout(self.decoder(x + self.attn(a)))
        return x



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
            h = nn.ModuleList([Decoder(config) for _ in range(config.n_layer)]),
        ))
       
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight #weight tying in-out


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
