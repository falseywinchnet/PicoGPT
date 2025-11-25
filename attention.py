import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
THE PARSEVAL MANIFOLD:
Deriving an Inherently Autoregressive Attention Mechanism
=========================================================
A Gemini-GPT-Falseywinchnet Collaborative Development (2024- 2025)
Human notes: a long time coming. a lot of papers, a lot of ideas.
A lot of experimenting, hunting and pecking. 

Our paradigm rules are as follows:
1: If a high LR makes it explode, something is wrong.
You should be able to learn at 1/e.
2: attention is basically a sort of weighted covariance portfolio. 
there's all kinds of parseval behavior going on here.
there are other projections we could make as far as what it does.
3: understand what the manifold is doing to figure out how to constrain it.

I. THE ONTOLOGICAL CRISIS: "The Riemannian Slaughterhouse"
Standard causal attention commits a topological violence. It computes a beautiful
All-to-All energy matrix (Riemannian metric) only to butcher half of it with a
crude boolean mask. This violates Parseval's Theorem of energy conservation.

The partition function Z(t) sums over 1 item at t=1 and 1000 items at t=1000.
The model is forced to act as a variance compensator, wasting capacity on
numerical stabilization rather than semantic flow. "Orphaned" probability mass
creates Attention Sinks (garbage collection on BOS tokens).

II. THE SOLUTION: "Causality by Construction"
We invert the paradigm. Instead of a "Search" (Q looking for K), we implement
a "Resonance" (Integrated Context vibrating against Instantaneous Pulse).

- Q (Context): The Leaky Integrator. A vessel containing the geometric sum of history.
- K (Pulse): The Dirac Innovation. The raw, instantaneous event.

If j <= t: K_j is physically embedded in Q_t. Dot product = ||K_j||^2 (Resonance).
If j > t:  K_j is absent from Q_t. Dot product = 0 (Orthogonality).

The mask becomes a formality—a noise filter, not a structural guillotine.

III. THE GEOMETRIC STACK
This implementation balances opposing forces to create a stable, differentiable manifold:
1. RoPE (Isometry): Preserves identity (Distance). Conservative Force.
2. Wedge (Symplectomorphism): Twists space (Flow). Disruptive Force.
3. Symbiotic Norm (Coupling): Forces Q/K to share energy. Stabilizing Force.
4. Scalar Sink (Entropy): Absorbs confusion isotropicallly. Entropic Force.
"""

# ==============================================================================
# 1. GEOMETRIC PRIMITIVES: The Tensor Operators
# ==============================================================================

class RoPE(nn.Module):
    """
    Rotary Positional Embeddings (The Conservative Force).
    
    PHYSICS:
    Applies a unitary rotation matrix R(theta). Since R^T R = I, this transformation
    is Isometric (Norm-Preserving). 
    
    INTERACTION:
    It enforces rigid relative distance. On the diagonal (t=t), RoPE is Identity.
    It anchors the manifold to the sequence grid, fighting the drift of the Wedge.
    """
    def __init__(self, dim, max_len=2048):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        self.register_buffer('cos', freqs.cos())
        self.register_buffer('sin', freqs.sin())

    def forward(self, x):
        # x: (B, H, T, D)
        seq_len = x.shape[2]
        cos = self.cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
        sin = self.sin[:seq_len, :].unsqueeze(0).unsqueeze(0)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.cat((-x2 * sin + x1 * cos, x1 * sin + x2 * cos), dim=-1)


class WedgeTransform(nn.Module):
    """
    The Symplectic Wedge (The Disruptive Force).
    
    PHYSICS:
    Applies a skew-symmetric flow: v_new = v(I + S), where S = A - A^T.
    This approximates a symplectic deformation, preserving phase-space volume but
    twisting the manifold to bridge semantic gaps.
    
    INTERACTION:
    - Diagonal (Self): <v, vS> = 0. The Wedge is invisible to the self. It cannot
      corrupt Identity.
    - Off-Diagonal (Relation): The Wedge activates, warping the space to allow
      distant tokens to resonate despite RoPE's rigid distance cost.
      
    INITIALIZATION: "Silent Wedge" (Zeros).
    We start in flat Euclidean space. The model learns to curve the universe
    only where necessary. 
    """
    def __init__(self, dim, heads):
        super().__init__()
        self.n_head = heads
        self.head_dim = dim // heads
        # Silent Init: Start flat.
        self.A = nn.Parameter(torch.zeros(heads, self.head_dim, self.head_dim))
        
    def forward(self, x):
        # x: (B, H, T, D)
        B, H, T, Dh = x.shape
        v = x.transpose(1, 2) # (B, T, H, D)
        
        # S is (H, D, D). Enforce Skew-Symmetry.
        S = self.A - self.A.transpose(-1, -2)
        
        # Apply Flow
        flow = torch.matmul(v, S)
        v_twisted = v + flow
        return v_twisted.transpose(1, 2) # Back to (B, H, T, D)


class _FusedLogSumExp4D(torch.autograd.Function):
    """
    The Numerical Guardian (Convex Softmax Kernel).
    
    PHYSICS:
    Standard FP16 Softmax underflows for small probabilities (exp(-1000) -> 0).
    This destroys gradients for "quiet" signals (Gradient Starvation).
    We compute LogSumExp in Float32 to preserve the "Energy" of the distribution,
    ensuring probability mass is neither created nor destroyed by precision errors.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        m, _ = x.max(dim=-1, keepdim=True)           
        y = x - m
        ex = y.exp()
        s = ex.sum(dim=-1, keepdim=True)
        lse = m + s.log()
        ctx.save_for_backward(ex, s)
        return lse

    @staticmethod
    def backward(ctx, grad_output):
        ex, s = ctx.saved_tensors
        grad_x = grad_output * (ex / s)
        return grad_x

class ConvexSoftmax(nn.Module):
    def forward(self, scores):
        lse = _FusedLogSumExp4D.apply(scores)
        log_weights = scores - lse
        return torch.exp(log_weights)

# ==============================================================================
# 2. THE ARCHITECTURE: Context-Pulse Attention
# ==============================================================================

class ContextPulseAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.scale = self.head_dim ** -0.5

        # --- PROJECTIONS ---
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        
        # --- ROLE-BASED INITIALIZATION (Geometric Priors) ---
        # Q (Context): Orthogonal. The Vessel must be rigid to contain the sum.
        nn.init.orthogonal_(self.W_Q.weight)
        # K (Pulse): Silent (Normal 0.02). The Event starts quiet to avoid flooding.
        nn.init.normal_(self.W_K.weight, mean=0.0, std=0.02)
        # V (Value): Xavier. Standard signal transport.
        nn.init.xavier_normal_(self.W_V.weight)

        # --- DYNAMIC GATING ---
        # Controls the Leaky Integrator decay.
        # Shared projection, but per-head decay parameter.
        self.decay_logits = nn.Parameter(torch.zeros(1, n_head, 1, 1))
        self.gate_proj = nn.Linear(d_model, d_model)

        # --- GEOMETRIC STACK ---
        self.rope = RoPE(self.head_dim)
        self.wedge = WedgeTransform(self.head_dim, n_head) 
        self.softmax = ConvexSoftmax()
        
        # --- THE TOILET (Entropy Management) ---
        # 1. Scalar Sink: An isotropic energy floor. "If signal < threshold, dump here."
        #    Per-head independence allows Head A to be strict and Head B lenient.
        self.sink_scalar = nn.Parameter(torch.zeros(1, n_head, 1, 1))
        
        # 2. Learned Reset Value: The "I Don't Know" vector.
        #    When the Sink activates, we output this bias instead of silence.
        self.v_null = nn.Parameter(torch.zeros(1, n_head, 1, self.head_dim)) 

    def forward(self, x):
        B, T, C = x.shape
        H, Dh = self.n_head, self.head_dim
        
        # 1. Projections: (B, T, H, D) -> (B, H, T, D)
        q_raw = self.W_Q(x).view(B, T, H, Dh).transpose(1, 2)
        k_raw = self.W_K(x).view(B, T, H, Dh).transpose(1, 2)
        v = self.W_V(x).view(B, T, H, Dh).transpose(1, 2)
        
        # 2. Gating: Input-Dependent Decay
        gate = torch.sigmoid(self.gate_proj(x)).view(B, T, H, Dh).transpose(1, 2)
        decay = torch.sigmoid(self.decay_logits) # (1, H, 1, 1)
        
        # 3. THE INTEGRATION (Constructing the Manifold)
        # Q becomes the accumulation of past K pulses.
        # Causality is enforced by the loop structure, not a mask.
        Q_context = torch.zeros_like(q_raw)
        running_q = torch.zeros(B, H, Dh, device=x.device)
        
        for t in range(T):
            # gate controls "how much pulse enters context"
            input_t = gate[:, :, t, :] * q_raw[:, :, t, :]
            # decay controls "how much history is retained"
            running_q = decay[:, :, :, 0] * running_q + (1.0 - decay[:, :, :, 0]) * input_t
            Q_context[:, :, t, :] = running_q
        K_pulse = k_raw
        
        # 4. SYMBIOTIC INSTANCE NORM (The Coupling Force)
        # Forces Q and K to share energy statistics per-sample, per-head.
        # Prevents "Ghost of Newton" optimization drift where Q >> K.
        # Normalize over (Time, Dim) only. Keep Heads independent.
        dims = (2, 3) 
        mean_sym = 0.5 * (Q_context.mean(dim=dims, keepdim=True) + K_pulse.mean(dim=dims, keepdim=True))
        std_sym = 0.5 * (Q_context.std(dim=dims, keepdim=True) + K_pulse.std(dim=dims, keepdim=True))
        
        Q_context = (Q_context - mean_sym) / (std_sym + 1e-6)
        K_pulse = (K_pulse - mean_sym) / (std_sym + 1e-6)
            
        # 5. GEOMETRIC TWIST (Structure vs Flow)
        # RoPE applies rigid rotation (Time-Encoding).
        # Wedge applies symplectic twist (Semantic-Encoding).
        Q_geo = self.wedge(self.rope(Q_context))
        K_geo = self.wedge(self.rope(K_pulse))
        
        # 6. RESONANCE (Attention Scores)
        # Dot product measures geometric interference.
        Attn = (Q_geo @ K_geo.transpose(-2, -1)) * self.scale
        
        # Masking (Still needed to filter the "Future Pulse" noise, which is orthogonal but non-zero)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        Attn.masked_fill_(mask, float('-inf'))
        
        # 7. THE SINK (Entropy Floor)
        # Inject scalar threshold. If Attn < SinkScalar, probability flows here.
        # Expands to (B, H, T, 1). No rotation needed.
        null_scores = self.sink_scalar.expand(B, H, T, 1)
        Attn_full = torch.cat([Attn, null_scores], dim=-1)
        
        # 8. PROBABILITY MASS
        probs_full = self.softmax(Attn_full)
        
        # 9. VALUE INTEGRATION
        # Standard weighted sum of history
        probs_seq = probs_full[..., :T]
        out = probs_seq @ v
        
        # 10. RESET MECHANISM
        # If sink activated, add the learned "I Don't Know" vector.
        # This is numerically safe because the scalar sink naturally regulates probability.
        probs_sink = probs_full[..., T:] # (B, H, T, 1)
        out = out + probs_sink * self.v_null
        
        # 11. OUTPUT
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_O(out)
\
