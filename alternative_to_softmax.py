self.scale = math.pi / math.sqrt(3.0)
scores = (q @ k.transpose(-2, -1))
key_self = (k * k).sum(dim=-1)     # (B,T) = ||k_j||^2
denom = key_self.unsqueeze(-2).sqrt() + 1e-8    # (B,1,T) = ||k_j||
w = scores / denom #scaling by k 
w = F.softplus(w)  # >= 0, softplus(-inf)=0
mask = torch.triu(torch.ones(T, T, device=X.device), diagonal=1).bool()
w = w * torch.sigmoid(self.scale * w)
w = w.masked_fill(mask, float("0.0"))
alpha = (w.sum(dim=-1, keepdim=True) + 1e-8).reciprocal()
attn_weights = w * alpha
out_tokens = attn_weights @ v
'''
Let a row use a_j = f(z_j) / sum_k f(z_k), with f(z) = softplus(z) sigmoid(c softplus(z)).

For large positive z, f is approximately linear, f(z) ≈ z. In that regime:

Global upward drift flattens
If every z_j increases by about the same amount c, 
then weights become proportional to z_j + c, 
and the added c increases the denominator by about T c. 
Relative differences shrink, so attention becomes less peaky. 
That is negative feedback against “inflate everything.”

Mild growth avoids runaway peaking
Because f grows roughly linearly rather than exponentially, 
increasing the spread of z does not produce the same winner take all 
collapse as softmax. The q scaling tests show exactly that, 
10x to 100x barely changes top1 for custom, while softmax collapses.

Key norm cancellation removes the incentive to “cheat” by blowing up key norms
z uses k direction only. So one common failure mode in softmax, 
inflate key norms to sharpen, is not available. 
The key scaling invariance tests confirm this.

So yes, there is a stabilizing loop: scale inflation pushes into a regime 
where additional inflation yields diminishing selectivity gains, 
and baseline inflation can even reduce selectivity.


Softmax attention, for a single head and one query position i:
Scores: s_ij = (q_i dot k_j) / sqrt(d)
Weights: a_ij = exp(s_ij) / sum_j exp(s_ij)

The provided replacement (rewritten with explicit intermediate z):

Scores: s_ij = (q_i dot k_j) / sqrt(d)
Key norm: n_j = ||k_j|| + eps
Normalized scores: z_ij = s_ij / n_j
Nonnegative preweights: u_ij = softplus(z_ij)
Gated preweights: w_ij = u_ij * sigmoid(c * u_ij), c = pi / sqrt(3)
Weights: a_ij = w_ij / sum_j w_ij
Output: out_i = sum_j a_ij v_j


Exact invariance to scaling keys
If k_j is replaced by a * k_j, then s_ij scales by a and n_j scales by a, 
so z_ij stays the same, so a_ij stays the same (up to tiny floating error). 
Softmax attention is not invariant to this.

Growth is roughly linear for large positive scores
Softmax uses exponentials, so large score gaps can collapse attention 
onto a single key quickly. Here, softplus grows roughly like identity on large positives,
sigmoid(c*u) saturates to 1, so w grows roughly linearly with z for large z. 
That tends to be less aggressively peaky.

Shift invariance differs
Softmax is invariant to adding the same constant to all s_ij for fixed i.
The softplus based mechanism is not, since softplus(z + const) changes absolute magnitudes 
before normalization. This can be either a bug or a feature, depending on whether absolute score level
is intended to control sharpness.


import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.set_printoptions(precision=4, sci_mode=False)

def custom_attn(q, k, v, eps=1e-8, causal=True):
    """
    q,k,v: (B,T,D)
    Returns: out (B,T,D), attn (B,T,T), z (B,T,T) where z = ((qk^T)/sqrt(D)) / (||k||+eps)
    """
    B, T, D = q.shape
    attnscale = D ** -0.5
    c = math.pi / math.sqrt(3.0)

    scores = (q @ k.transpose(-2, -1)) * attnscale  
# (B,T,T)
    key_norm = (k * k).sum(dim=-1).sqrt() + eps                  # (B,T)
    z = scores / key_norm.unsqueeze(-2)                          # (B,T,T)
    u = F.softplus(z)                                            # >= 0
    w = u * torch.sigmoid(c * u)                                 # >= 0, ~0.5u for small u, ~u for large u

    if causal:
        mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
        w = w.masked_fill(mask, 0.0)

    attn = w / (w.sum(dim=-1, keepdim=True) + eps)               # row-normalized, sum_j attn_ij = 1
    out = attn @ v
    return out, attn, z

def softmax_attn(q, k, v, causal=True):
    """
    q,k,v: (B,T,D)
    Returns: out (B,T,D), attn (B,T,T), scores (B,T,T)
    """
    B, T, D = q.shape
    attnscale = D ** -0.5

    scores = (q @ k.transpose(-2, -1)) * attnscale
    if causal:
        mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, -1e9)

    attn = torch.softmax(scores, dim=-1)
    out = attn @ v
    return out, attn, scores

def entropy(p, eps=1e-12):
    p = p.clamp_min(eps)
    return -(p * p.log()).sum(dim=-1)

def topk_mass(p, k):
    vals, _ = p.topk(k, dim=-1)
    return vals.sum(dim=-1)

def gini(p):
    ps, _ = p.sort(dim=-1)                       # ascending
    T = ps.shape[-1]
    idx = torch.arange(1, T + 1, device=p.device, dtype=p.dtype)
    weights = (T - idx + 0.5) / T
    return 1.0 - 2.0 * (ps * weights).sum(dim=-1)

def summarize(name, attn):
    ent = entropy(attn).mean().item()
    t1 = topk_mass(attn, 1).mean().item()
    t4 = topk_mass(attn, 4).mean().item()
    gn = gini(attn).mean().item()
    print(f"{name:>10}  mean_entropy={ent:.4f}  mean_top1={t1:.4f}  mean_top4={t4:.4f}  mean_gini={gn:.4f}")

def plot_rows(attn_a, attn_b, rows, title_a="custom", title_b="softmax"):
    # attn_*: (T,T) for a single batch
    T = attn_a.shape[-1]
    x = torch.arange(T).cpu()

    for r in rows:
        plt.figure()
        plt.plot(x, attn_a[r].detach().cpu())
        plt.plot(x, attn_b[r].detach().cpu())
        plt.title(f"attention row i={r}   {title_a} vs {title_b}")
        plt.xlabel("key index j")
        plt.ylabel("weight a_ij")
        plt.ylim(bottom=0.0)
        plt.show()

def max_abs_diff(a, b):
    return (a - b).abs().max().item()

# ---------- setup ----------
torch.manual_seed(0)
device = "cpu"

B, T, D = 1, 64, 32
q = torch.randn(B, T, D, device=device)
k = torch.randn(B, T, D, device=device)
v = torch.randn(B, T, D, device=device)

out_c, attn_c, z_c = custom_attn(q, k, v, causal=True)
out_s, attn_s, s_s = softmax_attn(q, k, v, causal=True)

print("Base distribution summary:")
summarize("custom", attn_c)
summarize("softmax", attn_s)

# Plot a few rows (early, middle, late)
plot_rows(attn_c[0], attn_s[0], rows=[0, T//2, T-1])

# ---------- property 1: invariance to scaling keys ----------
scale_k = 3.0
k_scaled = k * scale_k

_, attn_c2, _ = custom_attn(q, k_scaled, v, causal=True)
_, attn_s2, _ = softmax_attn(q, k_scaled, v, causal=True)

print("\nKey scaling test (k <- 3*k):")
print(f"custom max|Δattn| = {max_abs_diff(attn_c, attn_c2):.6g}")
print(f"softmax max|Δattn| = {max_abs_diff(attn_s, attn_s2):.6g}")

# ---------- property 2: shift invariance (softmax yes, custom no) ----------
# For softmax, adding a constant per-row to scores should not change the distribution.
# For custom, adding a constant to z changes softplus inputs and changes the distribution.
shift = 5.0

def softmax_from_scores(scores, causal=True):
    T = scores.shape[-1]
    if causal:
        mask = torch.triu(torch.ones(T, T, device=scores.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, -1e9)
    return torch.softmax(scores, dim=-1)

def custom_from_z(z, eps=1e-8, causal=True):
    c = math.pi / math.sqrt(3.0)
    u = F.softplus(z)
    w = u * torch.sigmoid(c * u)
    T = z.shape[-1]
    if causal:
        mask = torch.triu(torch.ones(T, T, device=z.device, dtype=torch.bool), diagonal=1)
        w = w.masked_fill(mask, 0.0)
    return w / (w.sum(dim=-1, keepdim=True) + eps)

attn_s_shift = softmax_from_scores(s_s + shift, causal=True)
attn_c_shift = custom_from_z(z_c + shift, causal=True)

print("\nShift test (+5 to all logits in a row):")
print(f"softmax max|Δattn| = {max_abs_diff(attn_s, attn_s_shift):.6g}")
print(f"custom  max|Δattn| = {max_abs_diff(attn_c, attn_c_shift):.6g}")

# ---------- property 3: peaking under large q scaling ----------
q_big = q * 10.0
_, attn_c_big, _ = custom_attn(q_big, k, v, causal=True)
_, attn_s_big, _ = softmax_attn(q_big, k, v, causal=True)

print("\nLarge q scaling test (q <- 10*q):")
summarize("custom", attn_c_big)
summarize("softmax", attn_s_big)

# ---------- gradient behavior snapshot ----------
def grad_snapshot(method, q_in, k_in, v_in):
    qg = q_in.clone().detach().requires_grad_(True)
    kg = k_in.clone().detach().requires_grad_(True)
    vg = v_in.clone().detach().requires_grad_(True)

    if method == "custom":
        out, attn, _ = custom_attn(qg, kg, vg, causal=True)
    else:
        out, attn, _ = softmax_attn(qg, kg, vg, causal=True)

    loss = out.pow(2).mean()
    loss.backward()

    return {
        "loss": loss.item(),
        "q_grad_norm": qg.grad.norm().item(),
        "k_grad_norm": kg.grad.norm().item(),
        "v_grad_norm": vg.grad.norm().item(),
        "mean_entropy": entropy(attn.detach()).mean().item(),
        "mean_top1": topk_mass(attn.detach(), 1).mean().item(),
    }

base_c = grad_snapshot("custom", q, k, v)
base_s = grad_snapshot("softmax", q, k, v)
big_c  = grad_snapshot("custom", q_big, k, v)
big_s  = grad_snapshot("softmax", q_big, k, v)

print("\nGradient snapshot (loss = mean(out^2)):")
print("Base:")
print(" custom", base_c)
print(" softmx", base_s)
print("\nWith q <- 10*q:")
print(" custom", big_c)
print(" softmx", big_s)


'''
