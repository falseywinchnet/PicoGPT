#MiT Licensed joshuah.rainstar@gmail.com 2025
#Gelu? Relu? Swish? sigmoid? norming? Layernorm? 
#forget all of that. Put one of these bad boys at the end of any linear.
#embed->PNG->block/>attention->QK->PNG->V->PNG->W_O_>PNG/>MLP->linear->PNG->unembed
import math
import torch
import torch.nn as nn

class PolyNormGate(nn.Module):
    """
    Unified "norm + activation" operator:
      v = u^2 + 0.75 u^3
      z = v / rms(v) over chosen dim
      h = z * sigmoid(a z),  a = pi/sqrt(3)

    dim:
      -1 for last-dim features (MLP/Transformer)
       1 for channels-first conv tensors (N,C,H,W), normalized over C per spatial location
    """
    def __init__(self, dim: int = -1, eps: float = 1e-6, affine: bool = False, num_features: int = None):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.a = math.pi / math.sqrt(3.0)

        self.affine = bool(affine)
        if self.affine:
            if num_features is None:
                raise ValueError("num_features required when affine=True")
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def _apply_affine(self, x: torch.Tensor) -> torch.Tensor:
        if not self.affine:
            return x
        if x.dim() == 2 and self.dim == -1:
            # (B, C)
            w = self.weight.view(1, -1)
            b = self.bias.view(1, -1)
            return x * w + b
        if x.dim() == 3 and self.dim == -1:
            # (B, T, C)
            w = self.weight.view(1, 1, -1)
            b = self.bias.view(1, 1, -1)
            return x * w + b
        if x.dim() == 4 and self.dim == 1:
            # (B, C, H, W)
            w = self.weight.view(1, -1, 1, 1)
            b = self.bias.view(1, -1, 1, 1)
            return x * w + b
        # Generic broadcast along chosen dim
        shape = [1] * x.dim()
        shape[self.dim] = -1
        w = self.weight.view(*shape)
        b = self.bias.view(*shape)
        return x * w + b

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        v = u * u + 0.75 * u * u * u
        rms = torch.sqrt(v.pow(2).mean(dim=self.dim, keepdim=True) + self.eps)
        z = v / rms
        h = z * torch.sigmoid(self.a * z)
        return self._apply_affine(h)

