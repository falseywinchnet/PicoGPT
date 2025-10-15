import torch
from torch import nn

class HadamardMeanDistance(nn.Module):
    """
    Computes your distance between:
      • the average of all Hadamard products over all rearrangements (which collapses to mean(x)*mean(y) per element),
      • and the elementwise mean (x + y) / 2,
    taking the Euclidean norm along the LAST dimension.

    Given x, y with shape (..., C), returns a tensor with shape (...,).
    Assumes x and y have the same shape and compares along the last dim.

    Formula per sample (last-dim = C):
        D = || (x + y)/2  -  mean(x)*mean(y) * 1_C ||_2
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.shape != y.shape:
            raise ValueError(f"x and y must have the same shape, got {x.shape} vs {y.shape}")
        # Ensure floating type for means
        if not torch.is_floating_point(x):
            x = x.float()
        if not torch.is_floating_point(y):
            y = y.float()

        # Means along the last dimension, keepdim for broadcasting
        mx = x.mean(dim=-1, keepdim=True)  # (..., 1)
        my = y.mean(dim=-1, keepdim=True)  # (..., 1)

        # Average of Hadamard products over all rearrangements -> constant vector mx*my
        h_avg = (mx * my).expand_as(x)     # (..., C)

        # Elementwise average of the two original arrays
        m_vec = 0.5 * (x + y)              # (..., C)

        # Euclidean distance along last dimension
        diff = m_vec - h_avg               # (..., C)
        D = diff.norm(p=2, dim=-1)         # (...,)

        return D


# --- Example usage ---
if __name__ == "__main__":
    x = torch.tensor([[1.0, 2.0, 3.0],
                      [0.5, 0.5, 0.5]])           # shape (2, 3)
    y = torch.tensor([[3.0, 2.0, 1.0],
                      [0.0, 1.0, 2.0]])           # shape (2, 3)

    metric = HadamardMeanDistance()
    d = metric(x, y)                               # shape (2,)
    print(d)                                       # tensor([something, something])
