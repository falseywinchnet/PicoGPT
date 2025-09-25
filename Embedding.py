from __future__ import annotations
import math
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Tuple

def _is_prime(n: int) -> bool:
    if n < 2: return False
    if n % 2 == 0: return n == 2
    r = int(n**0.5)
    for f in range(3, r+1, 2):
        if n % f == 0: return False
    return True

def _factorize(n: int):
    f, cnt = [], {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            cnt[d] = cnt.get(d, 0) + 1
            n //= d
        d += 1 if d == 2 else 2
    if n > 1: cnt[n] = cnt.get(n, 0) + 1
    return list(cnt.keys())

def _primitive_root(p: int) -> int:
    # p must be prime
    phi = p - 1
    factors = _factorize(phi)
    for g in range(2, p):
        ok = True
        for q in factors:
            if pow(g, phi // q, p) == 1:
                ok = False
                break
        if ok:
            return g
    raise RuntimeError("no primitive root found")

def _welch_costas_perm(V: int, device=None):
    """
    Welch Costas permutation σ on {0..V-1}, where V = p-1 for prime p.
    σ[i] = g^(i+1) mod p, mapped to 0..V-1 by subtracting 1.
    """
    p = V + 1
    if not _is_prime(p):
        return None
    g = _primitive_root(p)
    sigma = torch.empty(V, dtype=torch.long, device=device)
    for i in range(V):
        sigma[i] = pow(g, i + 1, p) - 1
    return sigma  # permutation of 0..V-1

def _coprime_mul_perm(V: int, device=None):
    """
    Fallback: σ[i] = (a*i + b) % V with gcd(a, V)=1 and a not ≡ ±1 mod V.
    Not Costas, but non-monotone and well-distributed.
    """
    # pick a
    a = None
    for cand in range(2, V):
        if math.gcd(cand, V) == 1 and cand % V not in (1, V-1):
            a = cand
            break
    if a is None:
        a = 1  # degenerate small V
    b = V // 3
    i = torch.arange(V, device=device)
    return ((a * i + b) % V).long()

def _perm_inverse(sigma: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(sigma)
    inv[sigma] = torch.arange(sigma.numel(), device=sigma.device)
    return inv

class FlatRollEmbed(nn.Module):
    """
    Replacement for nn.Embedding that maps token id i -> cyclic roll^i of a base
    length-V vector whose non-DC spectrum is flat (DC=0). Requires V == n_embd.
    Weights are frozen by default.
    this yields an optimal embedding that is considered perfect.
    The 'eye' is mixed at 0.5 and then rows are permuted by a Costas-like order
    to maximize uniqueness while keeping even collapse.
    but wait, you're asking, my embeds/vocab is not orthagonal!
    the solution is simple, clever, efficient- 
    use  Smooth full-space rotation matrix via Lie algebra exponential map.
        A = exp(t·G), where G ∈ so(D) is skew-symmetric and full-rank.
    partition vocab idx space by modulo over chosen block size, use different rotation
    range from 0 to pi(evenly divided) for all partitions, use ONE embed matrix,
    embed->shift. Minimizes necessary parameter count. up-project to desired embed dim.
    for decoder, you're operating over a larger dimensional space as-is. that's fine.
    if you like, you can try down-project and repeat-decode invert on all blocks,
    and use stiefel inverting by transpose but use two sets of slices of rotation ranges
    so that you have blue noise coverage with a partition going from original bound to bound
    but also overlap going from mid to mid, try decode on all, hard route to one,
    take logits from that one- > bam, no learned decode either.
    down-projection tied to up-projection and you have a learned high efficiency mapping.
    
    """
    def __init__(self, config, scale: str = "box", seed: int = 0,
                 freeze: bool = True, dtype=None, device=None):
        super().__init__()
        assert config.n_embd == config.vocab_size, (
            f"Expected n_embd == vocab_size, got {config.n_embd} != {config.vocab_size}"
        )
        V = int(config.vocab_size)
        dtype = dtype or torch.float32

        eye = torch.eye(V, dtype=dtype, device=device)
        weight = self._make_weight(V, scale=scale, seed=seed,
                                   dtype=dtype, device=device)  # [V, V]
        M = int(torch.argmax(weight[0]))        # index of max in base x (row 0)
        pm = weight[0, M]                       # scalar
        N = 1.0 / pm
        
        eye = torch.roll(eye, shifts=M, dims=1) # shift spike position within each row
        eye = eye * N
        mixed =  weight + eye  # add identity towers

        # --- compute a strong-scramble row order (Costas if possible) ---
        sigma = _welch_costas_perm(V, device=device)
        if sigma is None:
            sigma = _coprime_mul_perm(V, device=device)
        # We want ones at (row = σ[i], col = i). For row-permutation via index_select,
        # use r_idx = σ^{-1} so that new_row j pulls old_row r_idx[j] with 1 at column j=σ[i].
        r_idx = _perm_inverse(sigma)

        # keep for reference / decoding
        self.register_buffer("row_perm", r_idx, persistent=False)
        self.register_buffer("sigma", sigma, persistent=False)

        mixed = mixed.index_select(0, r_idx)
        self.embed = nn.Embedding.from_pretrained(mixed, freeze=freeze)


    @staticmethod
    def _row_perm_max_equidistant(V: int, device=None) -> torch.Tensor:
        """
        Row permutation that evenly offsets the identity's '1' away from the diagonal.
        Uses a single cyclic shift by k = floor(V/2).
        """
        if V <= 1:
            return torch.arange(V, device=device, dtype=torch.long)
        k = V // 2
        if k == 0:  # only happens when V == 1, handled above; keep for safety
            k = 1
        return ((torch.arange(V, device=device) + k) % V).long()

    @staticmethod
    def _make_weight(V: int, scale: str = "box", seed: int = 0,
                     dtype=torch.float32, device=None) -> torch.Tensor:
        """
        Returns a (V, V) tensor whose rows are cyclic rolls of a base vector x in R^V
        with |FFT(x)|^2 flat for k=1..V-1 and DC=0.
        scale:
          - "unit": ||x||_2 = 1
          - "box":  max|x_i| = 1
        """
        # build on CPU, move at end
        complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        g = torch.Generator().manual_seed(seed)

        X = torch.zeros(V, dtype=complex_dtype)
        # DC bin
        X[0] = torch.tensor(0, dtype=complex_dtype)

        if V % 2 == 0:
            # bins 1..V/2-1 are complex-conjugate pairs; Nyquist bin must be real
            for k in range(1, V // 2):
                phi = torch.rand((), generator=g) * (2 * math.pi)
                val = torch.cos(phi) + 1j * torch.sin(phi)
                X[k] = val
                X[V - k] = torch.conj(val)
            X[V // 2] = 1.0 if torch.rand((), generator=g) < 0.5 else -1.0
        else:
            for k in range(1, (V - 1) // 2 + 1):
                phi = torch.rand((), generator=g) * (2 * math.pi)
                val = torch.cos(phi) + 1j * torch.sin(phi)
                X[k] = val
                X[V - k] = torch.conj(val)

        x = torch.fft.ifft(X).real  # real length-V base vector

        if scale == "unit":
            x = x / (x.norm() + 1e-12)
        elif scale == "box":
            x = x / (x.abs().max() + 1e-12)
        else:
            raise ValueError("scale must be 'unit' or 'box'")

        rows = [torch.roll(x, shifts=r, dims=0) for r in range(V)]
        W = torch.stack(rows, dim=0).to(dtype=dtype)
        if device is not None:
            W = W.to(device)
        return W

    def forward(self, input_ids: torch.LongTensor):
        # (batch, seq_len, V)
        return self.embed(input_ids)
