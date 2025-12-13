import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, d_in,h_dim,d_out):
        super().__init__()
        self.c_fc    = nn.Linear( d_in, h_dim, bias=True)
        self.scale = math.pi / math.sqrt(3.0)
        self.c_proj  = nn.Linear(h_dim, d_out, bias=True)

    def forward(self, x):
        x = self.c_fc(x)
        x = x **2 + 0.7049* x**3 #todo- determine if must be adjusted for hidden_dim size changes
        x = x * torch.sigmoid(self.scale * x)
        x = self.c_proj(x)
        return x
