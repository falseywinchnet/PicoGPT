#dramatically improves learning on a variety of tasks.
#copyright 2025 joshuah.rainstar@gmail.com mit licensed
import torch
import torch.nn as nn

class TanhLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.use_bias     = bias

        # raw weight params
        self.W_raw = nn.Parameter(torch.randn(out_features, in_features) * 0.05)
        # optional bias (standard, unconstrained)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("b", None)

    @property
    def weight(self) -> torch.Tensor:
        # fused effective weight
        return torch.tanh( 0.5 * self.W_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.weight
        if self.use_bias:
            return x @ W.t() + self.bias
        else:
            return x @ W.t()

    @torch.no_grad()
    def fused_params(self):
        """Return (W_eff, b_eff or None) detached copies for freezing/export."""
        W = self.weight.detach().clone()
        b = self.bias.detach().clone() if self.use_bias else None
        return W, b

    @torch.no_grad()
    def distill(self) -> nn.Linear:
        """Create a standard nn.Linear with the current fused params."""
        lin = nn.Linear(self.in_features, self.out_features, bias=self.use_bias)
        W, b = self.fused_params()
        lin.weight.copy_(W)
        if self.use_bias:
            lin.bias.copy_(b)
        return lin
