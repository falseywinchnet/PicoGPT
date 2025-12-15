#copyright joshuah.rainstar@gmail.com 2025
#MIT with attribution

import math
import copy
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        x = x**2 + 0.75*x**3
        x = x * torch.sigmoid(self.scale * x)
        x = self.c_proj(x)
        return x

import math
import torch
import torch.nn as nn


class CausalSegmentationSetHash(nn.Module):
    """
    Prefix fingerprint that ignores order and unknown irregular boundaries by
    averaging a set-of-segments hash over all segmentations under a length prior.

    Input:
      x: (B,T,C)

    Output:
      y: (B,T,C) with y[:,t] summarizing x[:,:t+1] without encoding order.
    """
    def __init__(
        self,
        c: int,
        d_rff: int = 256,          # D
        lmax: int = 64,            # maximum segment length considered
        n_scales: int = 4,         # number of different length priors (granularities)
        alpha: float = 8.0,        # within-segment shrinkage
        beta: float = 16.0,        # anchor strength (backoff)
        gamma: float = 16.0,       # output confidence shrinkage
        seed: int = 0,
        trainable_priors: bool = False,
    ):
        super().__init__()
        assert d_rff % 2 == 0
        self.c = c
        self.d_rff = d_rff
        self.lmax = lmax
        self.n_scales = n_scales
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)

        g = torch.Generator()
        g.manual_seed(seed)
        d_half = d_rff // 2

        # psi via random Fourier features (token -> D)
        self.register_buffer("W", torch.randn(c, d_half, generator=g) / math.sqrt(c))
        self.register_buffer("b", 2 * math.pi * torch.rand(d_half, generator=g))

        # phi (segment -> C), lightweight and trainable
        self.phi = nn.Sequential(
            nn.Linear(d_rff, c, bias=True),
            nn.Tanh(),
        )

        # Per-scale anchors
        self.anchor = nn.Parameter(torch.zeros(n_scales, c))

        # Length priors: logits over lengths 1..Lmax for each scale
        # Default initialization: different geometric-ish biases, but still supports arbitrary lengths.
        init = torch.zeros(n_scales, lmax)
        for s in range(n_scales):
            # bias toward different typical lengths, no periodicity
            tau = 2.0 ** s
            lengths = torch.arange(1, lmax + 1).float()
            init[s] = -lengths / tau
        if trainable_priors:
            self.log_w = nn.Parameter(init)
        else:
            self.register_buffer("log_w", init)

        # Mix scales back to C
        self.mix = nn.Linear(n_scales * c, c, bias=True)

    def _psi_rff(self, x_bt_c: torch.Tensor) -> torch.Tensor:
        # x: (..., C) -> (..., D)
        z = x_bt_c @ self.W + self.b
        feats = torch.cat([torch.cos(z), torch.sin(z)], dim=-1)
        feats = feats * (1.0 / math.sqrt(self.d_rff // 2))
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        assert C == self.c
        device, dtype = x.device, x.dtype

        # Token features u_t = psi(x_t) in R^D
        u = self._psi_rff(x.reshape(B * T, C)).reshape(B, T, self.d_rff)

        # Prefix sums of u for fast segment sums
        pref = torch.zeros(B, T + 1, self.d_rff, device=device, dtype=dtype)
        pref[:, 1:] = torch.cumsum(u, dim=1)

        # Outputs
        y = torch.zeros(B, T, C, device=device, dtype=dtype)

        # Run DP per scale
        reps_per_scale = []

        log_w = self.log_w.to(dtype=dtype)  # (S, Lmax)

        for s in range(self.n_scales):
            # DP state at each prefix length t:
            # logZ[t] is log normalizer for segmentations of prefix length t
            # q[t] is expected sum of segment embeddings under that distribution
            # k[t] is expected segment count under that distribution
            logZ = torch.full((B, T + 1), -float("inf"), device=device, dtype=dtype)
            logZ[:, 0] = 0.0

            q = torch.zeros(B, T + 1, C, device=device, dtype=dtype)
            k = torch.zeros(B, T + 1, 1, device=device, dtype=dtype)

            for t in range(1, T + 1):
                L = min(t, self.lmax)

                # Build candidate last-segment lengths 1..L
                lengths = torch.arange(1, L + 1, device=device)
                starts = t - lengths  # start index in prefix-sum space

                # Segment sums s(i,t) for each length
                seg_sum = pref[:, t:t+1, :] - pref[:, starts, :]  # (B,L,D)
                seg_mean = seg_sum / (lengths.view(1, L, 1).to(dtype) + self.alpha)

                # Segment embeddings e(i,t)
                seg_emb = self.phi(seg_mean.reshape(B * L, self.d_rff)).reshape(B, L, C)

                # Mixture weights over last length using log-space:
                # loga_l = logw_l + logZ[t-l]
                loga = log_w[s, :L].view(1, L).to(dtype) + logZ[:, starts]  # (B,L)
                pi = F.softmax(loga, dim=1)  # (B,L)

                # Update expected sum and expected count
                prev_q = q[:, starts, :]      # (B,L,C)
                prev_k = k[:, starts, :]      # (B,L,1)

                q[:, t, :] = torch.sum(pi.unsqueeze(-1) * (prev_q + seg_emb), dim=1)
                k[:, t, :] = torch.sum(pi.unsqueeze(-1) * (prev_k + 1.0), dim=1)

                # Update logZ for completeness
                logZ[:, t] = torch.logsumexp(loga, dim=1)

            # Anchored backoff average: (q + beta*a) / (k + beta)
            a = self.anchor[s].to(dtype).view(1, 1, C)
            rep = (q[:, 1:, :] + self.beta * a) / (k[:, 1:, :] + self.beta)

            # Confidence shrinkage
            conf = k[:, 1:, :] / (k[:, 1:, :] + self.gamma)
            rep = rep * conf

            reps_per_scale.append(rep)

        h = torch.cat(reps_per_scale, dim=-1)  # (B,T,S*C)
        y = self.mix(h)                        # (B,T,C)
        return y



class PhaseVectorImprovedPositional(nn.Module):
    """
    Chebyshev rotary-style positional embedding that matches RoPE tensor layout.

    Expects q, k shaped (B, H, T, D) with even D, rotates pairs over the last dim.
    If headwise=True, degree assignment is spread across (H * (D/2)) pairs, not just (D/2).
    """
    def __init__(self, dim: int, max_deg: Optional[int] = None, headwise: bool = True):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.max_deg = max_deg
        self.headwise = headwise

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert q.shape == k.shape
        B, Hh, T, D = q.shape
        assert D % 2 == 0
        device, dtype = q.device, q.dtype
        P = D // 2

        max_deg = self.max_deg
        if max_deg is None:
            max_deg = max(3, 2 * P)

        if T > 1:
            u = torch.arange(T, device=device, dtype=dtype) / (T - 1)
            x = 2.0 * u - 1.0
        else:
            x = torch.zeros(1, device=device, dtype=dtype)

        T_all = torch.empty(T, max_deg + 1, device=device, dtype=dtype)
        T_all[:, 0] = 1.0
        if max_deg >= 1:
            T_all[:, 1] = x
        for d in range(2, max_deg + 1):
            T_all[:, d] = 2.0 * x * T_all[:, d - 1] - T_all[:, d - 2]

        if self.headwise and (Hh * P) > 1:
            total_pairs = Hh * P
            g = torch.arange(total_pairs, device=device, dtype=dtype)
            frac = g / (total_pairs - 1)
            n = 1 + (frac * (max_deg - 2)).round().to(torch.long)
            n = torch.clamp(n, 1, max_deg - 1).view(Hh, P)
        else:
            if P > 1:
                p = torch.arange(P, device=device, dtype=dtype)
                frac = p / (P - 1)
            else:
                frac = torch.zeros(1, device=device, dtype=dtype)
            n = 1 + (frac * (max_deg - 2)).round().to(torch.long)
            n = torch.clamp(n, 1, max_deg - 1).view(1, P).expand(Hh, P)

        n_plus = n + 1

        raw1 = T_all[:, n]       # (T, H, P)
        raw2 = T_all[:, n_plus]  # (T, H, P)

        raw1 = raw1.permute(1, 0, 2).contiguous()  # (H, T, P)
        raw2 = raw2.permute(1, 0, 2).contiguous()  # (H, T, P)

        denom = torch.sqrt(raw1 * raw1 + raw2 * raw2 + 1e-8)
        base1 = (raw1 / denom).unsqueeze(0)  # (1, H, T, P)
        base2 = (raw2 / denom).unsqueeze(0)  # (1, H, T, P)

        def apply_rot(x_in: torch.Tensor) -> torch.Tensor:
            x1 = x_in[..., :P]
            x2 = x_in[..., P:]
            xr1 = x1 * base1 - x2 * base2
            xr2 = x1 * base2 + x2 * base1
            return torch.cat([xr1, xr2], dim=-1)

        return apply_rot(q), apply_rot(k)



class Attention(nn.Module):
    def __init__(self, d_model, n_head): 
        super().__init__()
        k_retrieval = 12
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
        self.k_retrieval = k_retrieval

        # --- 1. Projections ---
        self.W_K = nn.Linear(d_model, d_model, bias=True)
        self.W_Q_all = nn.Linear(d_model, d_model * self.n_branches, bias=True)
        self.posemb = PhaseVectorImprovedPositional(self.head_dim, headwise=True)

        self.phash = CausalSegmentationSetHash(d_model)
        # --- 4. V network: maps marker (Dh) -> value (Dh)
        self.v_net = MLP(self.head_dim)
        # This takes the "marker" (hologram of K geometry) and locates the value
        # --- 5. Output ---
        self.W_O_params = nn.Parameter(torch.empty(self.n_branches, d_model, d_model))
        self.W_O_bias = nn.Parameter(torch.zeros(self.n_branches, d_model))
        nn.init.xavier_uniform_(self.W_O_params)
        
    def forward(self, A, X):
        B, T, C = A.shape
        H_tot = self.n_total_heads
        N_br = self.n_branches
        N_sh = self.n_sub_heads
        Dh = self.head_dim
        K_top = min(self.k_retrieval, T)
        # 1. Projections
        # Q: (B, T, TotalHeads, Dh) -> (B, TotalHeads, T, Dh)
        q = self.W_Q_all(A).view(B, T, H_tot, Dh).permute(0, 2, 1, 3).contiguous()
        q = norm(q)
        # K: content-based key templates
        k_base = self.W_K(X)
        k_base_u = k_base.view(B, T, N_sh, Dh).permute(0, 2, 1, 3).contiguous()
        k = k_base_u.repeat(1, N_br, 1, 1) # (B, H_tot, T, Dh)
        q,k = self.posemb(q,k)

        a = self.phash(X)
        a = a.view(B, T, N_sh, Dh).permute(0, 2, 1, 3).contiguous()
        anchor = a.repeat(1, N_br, 1, 1).unsqueeze(3)   # (B, H_tot, T, Dh)

        # 3. Calibrated absolute scores (no softmax, no sink)
        # raw scores s_{t j} = <q_t, k_j>/sqrt(Dh)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B,H_tot,T,T)

        # key self-scores c_j = <k_j, k_j>/sqrt(Dh) as per-key reference
        key_self = (k * k).sum(dim=-1) * self.scale
        denom = key_self.unsqueeze(-2).clamp_min(1e-6)    # (B,H_tot,1,T)

        # normalized scores (ratio wrt each key's self-score)
        weights_full = attn_scores / denom               # (B,H_tot,T,T)

        weights_full = weights_full * F.sigmoid(self.scale*weights_full) #logistic scaling
        weights_full = F.softplus(weights_full) #soft clamp the floor to discourage neg results
        mask = torch.triu(torch.ones(T, T, device=X.device), diagonal=1).bool()
        weights_full = weights_full.masked_fill(mask, 0.0)
        weights, idxs = weights_full.topk(K_top, dim=-1)     # (B,H_tot,T,K_top)

        # Gather keys for these indices from k (unsorted)
        k_flat        = k.contiguous().view(B * H_tot, T, Dh)   # (B*H_tot, T, Dh)
       
        # Chronological ordering of neighbors (by token index in T)
        idxs_sorted, sort_order = idxs.sort(dim=-1)                     # (B,H_tot,T,K_top)
        # sort weights to match time order
        weights_sorted = torch.gather(weights, -1, sort_order)          # (B,H_tot,T,K_top)

        # gather keys according to chrono-sorted indices
        idxs_sorted_flat     = idxs_sorted.contiguous().view(B * H_tot, T * K_top)
        idxs_sorted_expanded = idxs_sorted_flat.unsqueeze(-1).expand(-1, -1, Dh)
        k_gathered_sorted_flat = k_flat.gather(dim=1, index=idxs_sorted_expanded)
        k_sorted = k_gathered_sorted_flat.view(B, H_tot, T, K_top, Dh)  # (B,H_tot,T,K_top,Dh)

        # apply weights to k_sorted: weighted K sequence (the "hologram")
        #note- we have an intrinsic sink in the anchor.
        #if the k_sorted is small, the dist will shift-> anchor.
        #this allows the model to instantly realize "i am unsure".
        k_sorted_weighted = weights_sorted.unsqueeze(-1) * k_sorted     # (B,H_tot,T,K_top,Dh)
        
        # self-token key (anchor) from K: unscaled
        # k_vanilla is already (B, H_tot, T, Dh), each [b,h,t] is self row in K
        # k_vanilla is already (B, H_tot, T, Dh), each [b,h,t] is self row in K
        out = torch.cat([k_sorted_weighted, anchor], dim=-2) #(B, H_tot, T, K_top+1, Dh)
        context = out.mean(dim=3)                                  # (B,H_tot,T,Dh)
        context = self.v_net(context)#digest in context

        # 7. Output projection & Bias
        # Recover Branch dim
        context = context.view(B, N_br, N_sh, T, Dh)
        context = context.permute(0, 1, 3, 2, 4).contiguous().view(B, N_br, T, C)

        y_proj = torch.einsum('bntc,ncd->bntd', context, self.W_O_params)
        bias = self.W_O_bias.view(1, N_br, 1, C)

        y = y_proj + bias

        return y.mean(dim=1)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.architect = Attention(config.n_embd,config.n_head)
        self.engineer = Attention(config.n_embd,config.n_head)
        self.mlpa = MLP(config.n_embd)
        self.mlpb = MLP(config.n_embd)

    def forward(self,x):
        B, T, C = x.shape
        half = C//2
        A = x[...,:half]
        B = x[...,half:]
        B = B + self.architect(norm(A),norm(B))
        B = B + self.mlpa(norm(B))

        A = A + self.engineer(norm(B),norm(A))
        A = A + self.mlpb(norm(A))

        x = torch.cat([A,B],dim=-1)
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
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
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
        tok_emb = self.transformer.wte(idx)
        q = norm(self.transformer.synth(torch.ones_like(tok_emb)))
        x = torch.cat([tok_emb,q],dim=-1)

        # forward the GPT model itself
        for block in self.transformer.h:
            x  = checkpoint(block, x, use_reentrant=False)

        #extract the origin shift
        x = (x[...,:self.config.n_embd])

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
