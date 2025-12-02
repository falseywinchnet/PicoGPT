
import torch
import torch.nn as nn
import torch.nn.functional as F


#optimal MLP structure.
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        HIDDEN_DIM_CONSTANT = 2 #I have code for a system that converges this dynamically to find the optimal count for asic/FPGA HW.
        #however, in terms of times the input, 2-4x is sufficient for 99% of models and generalizes well.
        self.c_fc    = nn.Linear( config.n_embd, HIDDEN_DIM_CONSTANT * config.n_embd, bias=config.bias)
        self.scale = math.pi / math.sqrt(3.0)
        self.ln = LayerNorm(config.n_embd*HIDDEN_DIM_CONSTANT, bias=config.bias)

        self.c_proj  = nn.Linear(HIDDEN_DIM_CONSTANT * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = x **2 + 0.5* x**3
        x = self.ln(x)
        x = x * torch.sigmoid(self.scale * x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
