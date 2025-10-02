#MIT License Copyright 2024 joshuah rainstar
import torch
from torch.optim.optimizer import Optimizer

class IsotropicrWolf(Optimizer):
    """
    Wolf (polar-core):
      - operates in rotation-equivariant space via polar(G) = U @ V^T
      - EMA of the *direction* (Wolf's state['p']), but projected back to the orthogonal manifold
      - 'agreement' gating keeps the Wolf behavior: agree -> step; disagree -> global shrink
      - isotropic jitter added in the same space (no elementwise noise)

    Args:
      lr: step size (used both for descent and shrink)
      beta: EMA coefficient in [0,1] for direction averaging (Wolf's mixing)
      eps: numerical epsilon
      noise: scale of isotropic jitter (0 disables)
    """
    def __init__(self, params, lr=2e-3, beta=0.367879441, eps=1e-8, noise=0.367879441):
        defaults = dict(lr=lr, beta=beta, eps=eps, noise=noise)
        super().__init__(params, defaults)
        # state['p'] retains Wolf's name: it's the EMA direction in polar space
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['p'] = None  # will hold an orthogonal-like tensor

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr   = group['lr']
            beta = group['beta']
            eps  = group['eps']
            nz   = group['noise']

            for p in group['params']:
                g = p.grad
                if g is None:
                    continue

                # ---------- scalar/vector fallback ----------
                if g.ndim < 2:
                    # normalize gradient direction (no whitening; just direction)
                    g_norm = g.norm() + eps
                    d = g / g_norm
                    # init/ema in direction space
                    m = self.state[p]['p']
                    if m is None:
                        m = d.clone()
                    else:
                        m.mul_(1 - beta).add_(d, alpha=beta)
                        # reproject to unit direction
                        m /= (m.norm() + eps)
                    # agreement via cosine
                    agree = (m * d).sum() > 0
                    if agree:
                        # isotropic jitter in direction space
                        if nz > 0:
                            j = torch.randn_like(m)
                            j /= (j.norm() + eps)
                            d_step = (m + nz * j)
                            d_step /= (d_step.norm() + eps)
                        else:
                            d_step = m
                        p.add_(d_step, alpha=-lr)
                    else:
                        # Wolf-style global shrink on disagreement
                        p.mul_(1.0 - lr)
                    self.state[p]['p'] = m
                    continue

                # ---------- matrix case (core polar update) ----------
                g32 = g.float()
                # polar(G) = U @ V^T (equal energy across singular directions)
                U, _, Vh = torch.linalg.svd(g32, full_matrices=False)
                d = (U @ Vh).to(g.dtype)

                # init / EMA of direction in polar space (Wolf's exp_avg idea)
                m = self.state[p]['p']
                if m is None:
                    m = d.clone()
                else:
                    m = (1 - beta) * m + beta * d
                    # reproject EMA back to the orthogonal manifold: polar(m)
                    Um, _, Vmh = torch.linalg.svd(m.float(), full_matrices=False)
                    m = (Um @ Vmh).to(g.dtype)

                # rotation-invariant agreement: Frobenius cosine between m and d
                # (both near-orthogonal; norms ~ sqrt(min(m,n)), but we normalize anyway)
                num = (m * d).sum()
                den = (m.norm() * d.norm() + eps)
                agree = (num / den) > 0

                if agree:
                    # isotropic Wolf-style jitter, but in polar space:
                    # sample Gaussian, normalize Frobenius, blend then re-polarize
                    if nz > 0:
                        J = torch.randn_like(m, dtype=torch.float32)
                        J /= (J.norm() + 1e-9)
                        # blend in tangent-ish way then reproject
                        blended = (m.float() + nz * J)
                        Ub, _, Vbh = torch.linalg.svd(blended, full_matrices=False)
                        step_dir = (Ub @ Vbh).to(g.dtype)
                    else:
                        step_dir = m
                    p.add_(step_dir, alpha=-lr)
                else:
                    # Wolf's shrink-on-disagree, but global & rotation-equivariant
                    p.mul_(1.0 - lr)

                # stash EMA direction
                self.state[p]['p'] = m

        return loss


class AnsitropicWolf(Optimizer):
  """Implements Wolf algorithm."""
  #Wolf, also called Rainstar Optimizer, is fast. it is resistant to valleys and some other things where adam hangs.
  #it is a highly ansitropic single-direction annealer- it works best to replace SGD, not a real optimizer.
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
