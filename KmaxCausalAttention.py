class KmaxCausalAttention(nn.Module):
    def __init__(self, head_dim: int, eps: float = 1e-8, use_attnscale: bool = False):
        super().__init__()
        self.head_dim = head_dim
        self.eps = eps
        self.use_attnscale = use_attnscale
        self.scale = math.pi / math.sqrt(3.0)
        self.inv_scale = 1.0 / self.scale
        self.attnscale = head_dim ** -0.5

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self.use_attnscale:
            q = q * self.attnscale

        k_norm = k.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        u = k / k_norm
        z = q @ u.transpose(-2, -1)
        u_sp = F.softplus(z)
        w = F.silu(self.scale * u_sp) * self.inv_scale
        w = torch.tril(w)
        denom = w.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        attn = w / denom
        return attn @ v
