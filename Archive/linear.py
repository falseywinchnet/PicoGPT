import math, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

# ---------- utilities ----------
def fixed_orthonormal_2xd(d, seed=0):
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(d, d, generator=g)
    Q, _ = torch.linalg.qr(M)
    U = Q[:, :2].T  # (2, d), orthonormal rows
    return U

def random_orthonormal_2x2(seed=0):
    torch.manual_seed(seed)
    M = torch.randn(2,2)
    Q, _ = torch.linalg.qr(M)
    return Q

class LinearWithLeak(nn.Module):
    """
    Linear layer with a fixed orthonormal 'leak' term ensuring R^2>0 by construction.
    Behaves like nn.Linear: exposes .weight and .bias for compatibility.
    If negative_bias=True, bias = -softplus(u) (so it's always <= 0).
    notes: use negative_bias=true before gelu, softplus, etc but not after.
    #possibly only use this modified linear BEFORE gelu in 2 layer MLP.
    #promotes invertibility, high rank behavior.
    """
    def __init__(self, in_features, out_features, eps=1e-3, leak_seed=0, negative_bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.negative_bias = negative_bias

        # learnable free weight
        self.W_free = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.W_free, a=math.sqrt(5))

        # fixed leak directions
        U = fixed_orthonormal_2xd(in_features, seed=leak_seed)  # (2, in_features)
        if out_features > 2:
            U_full = torch.zeros(out_features, in_features)
            U_full[:2, :] = U
            U = U_full
        elif out_features < 2:
            U = U[:out_features, :]
        self.register_buffer("U", U)

        # bias parameters
        if negative_bias:
            self.u = nn.Parameter(torch.zeros(out_features))
        else:
            self.b = nn.Parameter(torch.zeros(out_features))

    @property
    def weight(self):
        """Effective weight = W_free + eps * U, matches nn.Linear.weight semantics."""
        return self.W_free + self.eps * self.U

    @property
    def bias(self):
        """Effective bias, negative-only if requested."""
        if self.negative_bias:
            return -F.softplus(self.u)
        return self.b

    def forward(self, x):
        return x @ self.weight.T + self.bias
