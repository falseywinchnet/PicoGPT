# --- End-only Coordinate-Aware Backprop for logits (NanoGPT-ready) ---
import torch
import torch.nn.functional as F

@torch.no_grad()
def _endcab_hook_factory(logits, lam=1e-4, gauge_fix=True):
    """
    Returns a hook that maps grad wrt logits g -> v = (F + lam*I)^(-1) g,
    where F = Diag(p) - p p^T for softmax p = softmax(logits).
    Supports logits of shape (..., V). No EMA; pure per-step metric.
    """
    # Cache the probabilities at attach-time; grads flow via the transformed g.
    logits_det = logits.detach()
    last_dim = logits_det.shape[-1]
    lam_t = torch.tensor(lam, dtype=logits_det.dtype, device=logits_det.device)

    # Precompute softmax in the flattened space for numerical stability
    p_flat = torch.softmax(logits_det.reshape(-1, last_dim), dim=-1)

    def hook(grad):
        # grad has same shape as logits; flatten to (N, V)
        g = grad.reshape(-1, last_dim)

        # Sherman–Morrison for A = (Diag(p) - p p^T + lam I)
        # Let M = Diag(p) + lam I  (note the sign; A = M - p p^T)
        # A^{-1} = M^{-1} + M^{-1} p p^T M^{-1} / (1 - p^T M^{-1} p)
        ainv = 1.0 / (p_flat + lam_t)             # (N, V)
        Ainv_g = ainv * g                         # M^{-1} g
        Ainv_p = ainv * p_flat                    # M^{-1} p
        num   = (p_flat * Ainv_g).sum(dim=1, keepdim=True)          # (N,1)
        denom = 1.0 - (p_flat * Ainv_p).sum(dim=1, keepdim=True)    # (N,1)
        v = Ainv_g + (num / (denom + 1e-12)) * Ainv_p               # (N,V)

        if gauge_fix:
            v = v - v.mean(dim=1, keepdim=True)   # remove softmax gauge component

        return v.reshape_as(grad)

    return hook

def endcab_attach(logits, lam=1e-4, gauge_fix=True):
    """
    Attach end-only CAB to a logits tensor. Call this *before* loss.backward().
    """
    logits.register_hook(_endcab_hook_factory(logits, lam=lam, gauge_fix=gauge_fix))


# ------------------------ Usage patterns ------------------------
# Case A: NanoGPT-style forward returns (logits, loss) when targets are given
# logits, loss = model(idx, targets)          # typical NanoGPT API
# endcab_attach(logits, lam=1e-4)             # swap ∂L/∂logits for NG step
# loss.backward()
# optimizer.step()

# Case B: Model returns logits; you compute CE yourself
# logits = model(idx)                          # shape (B,T,V) or (N,V)
# loss   = F.cross_entropy(logits.view(-1, V), targets.view(-1), ignore_index=-100)
# endcab_attach(logits, lam=1e-4)
# loss.backward()
# optimizer.step()

#discovered on idea that embeddings are coordinates
