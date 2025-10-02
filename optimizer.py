#MIT License Copyright 2024 joshuah rainstar
import torch
from torch.optim.optimizer import Optimizer
class Wolf(Optimizer):
  """Implements Wolf algorithm."""
  #Wolf, also called Rainstar Optimizer, is fast. it is resistant to valleys and some other things where adam hangs.
  #on some problems, it is faster than adam. on most things, proper annealing and adam is still a good idea.
  #wolf is initially smoother than adam over difficult plateaus and at high LR, except on quadratic problems.
  #for a proven guarantee on all problems set etcerta to  2/(lr*λ_max_hessian_eigenvalue).
  #however, this is more easily said than done- economically anyway. 
  #ongoing work to optimise this is in progress.
  def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8):
        # Define default parameters
        defaults = dict(lr=lr, betas=betas, eps=eps)
        self.lr = lr
        # Initialize the parent Optimizer class first
        super().__init__(params, defaults)
        # Constants specific to Wolf
        # Initialize state for each parameter
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['p'] = torch.zeros_like(p)  # Second moment estimate

  @torch.no_grad()
  def step(self, closure=None):
    """Performs a single optimization step.

    Args:
      closure (callable, optional): A closure that reevaluates the model
        and returns the loss.

    Returns:
      the loss.
    """
    etcerta = 0.367879441
    et = 1 - etcerta

    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue
        state = self.state[p]
        
        # Perform stepweight decay
        grad = p.grad
        state = self.state[p]
        # State initialization

        exp_avg = state['p'] 
        # Weight update
        update = exp_avg * et + grad * etcerta
        state['p']  = exp_avg * et + update * etcerta
        sign_agreement = torch.sign(update) * torch.sign(grad)

        update = update + (torch.rand_like(update)*2 - 1) * etcerta * update
        # Where signs agree (positive), apply normal update
        mask = (sign_agreement > 0)
        p.data = torch.where(mask, 
                            p.data - self.lr * update,
                            p.data - p.data*self.lr)
    return loss
