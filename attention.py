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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import math

# --- 1. LOCKED PRIMITIVES ---
class RoPE(nn.Module):
    def __init__(self, dim, max_len=2048):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        self.register_buffer('cos', freqs.cos())
        self.register_buffer('sin', freqs.sin())
    def forward(self, x):
        seq_len = x.shape[2]
        cos = self.cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
        sin = self.sin[:seq_len, :].unsqueeze(0).unsqueeze(0)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.cat((-x2 * sin + x1 * cos, x1 * sin + x2 * cos), dim=-1)

class WedgeTransform(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.n_head = heads
        self.head_dim = dim // heads
        self.A = nn.Parameter(torch.zeros(heads, self.head_dim, self.head_dim))
    def forward(self, x):
        B, H, T, Dh = x.shape
        v = x.transpose(1, 2)
        S = self.A - self.A.transpose(-1, -2)
        v_twisted = v + torch.matmul(v, S)
        return v_twisted.transpose(1, 2)

class ConvexSoftmax(nn.Module):
    def forward(self, scores):
        m, _ = scores.max(dim=-1, keepdim=True)
        y = scores - m
        ex = y.exp()
        lse = m + ex.sum(dim=-1, keepdim=True).log()
        return torch.exp(scores - lse)

# --- 2. THE CAUSAL CHUNKED MODEL ---
class ContextPulseAttention(nn.Module):
    def __init__(self, d_model, head_dim, chunk_size=16):
        super().__init__()
        self.d_model = d_model
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.scale = head_dim ** -0.5

        self.W_Q = nn.Linear(d_model, head_dim, bias=False)
        self.W_K = nn.Linear(d_model, head_dim, bias=False)
        self.W_V = nn.Linear(d_model, head_dim, bias=False)
        
        nn.init.orthogonal_(self.W_Q.weight)
        nn.init.normal_(self.W_K.weight, mean=0.0, std=0.02)
        nn.init.xavier_normal_(self.W_V.weight)

        self.gate_proj = nn.Linear(d_model, head_dim)
        self.rope = RoPE(head_dim)
        self.wedge = WedgeTransform(head_dim, 1)
        self.softmax = ConvexSoftmax()
        
        self.sink_scalar = nn.Parameter(torch.tensor(0.0))
        self.v_null = nn.Parameter(torch.zeros(1, 1, 1, head_dim)) 

    def forward(self, x):
        B, T, C = x.shape
        H, Dh = 1, self.head_dim
        
        q_raw = self.W_Q(x).view(B, T, H, Dh).transpose(1, 2)
        k_raw = self.W_K(x).view(B, T, H, Dh).transpose(1, 2)
        v_val = self.W_V(x).view(B, T, H, Dh).transpose(1, 2)
        gate = torch.sigmoid(self.gate_proj(x)).view(B, T, H, Dh).transpose(1, 2)

        num_chunks = math.ceil(T / self.chunk_size)
        Q_output_list = []
        history_cache = [] 
        
        for i in range(num_chunks):
            t_start = i * self.chunk_size
            t_end = min((i + 1) * self.chunk_size, T)
            
            q_chunk = q_raw[:, :, t_start:t_end, :]
            k_chunk = k_raw[:, :, t_start:t_end, :]
            gate_chunk = gate[:, :, t_start:t_end, :]
            
            if len(history_cache) > 0:
                history_stack = torch.cat(history_cache, dim=2)
                resonance_scores = (k_chunk @ history_stack.transpose(-2, -1)) * self.scale
                resonance_weights = F.softmax(resonance_scores, dim=-1)
                q_retrieved = resonance_weights @ history_stack
                base_state = history_cache[-1].expand(-1, -1, k_chunk.size(2), -1)
                q_initial = base_state + q_retrieved
            else:
                q_initial = torch.zeros_like(k_chunk)

            masked_input = gate_chunk * q_chunk
            q_integrated = torch.cumsum(masked_input, dim=2)
            q_chunk_out = q_integrated + q_initial
            
            Q_output_list.append(q_chunk_out)
            final_state = q_chunk_out[:, :, -1:, :]
            history_cache.append(final_state)
            
        Q_context = torch.cat(Q_output_list, dim=2)
        K_pulse = k_raw
        
        # --- POINTWISE SYMBIOTIC NORM (Dim=-1 only) ---
        mean_q = Q_context.mean(dim=-1, keepdim=True)
        std_q = Q_context.std(dim=-1, keepdim=True)
        mean_k = K_pulse.mean(dim=-1, keepdim=True)
        std_k = K_pulse.std(dim=-1, keepdim=True)
        
        mean_sym = 0.5 * (mean_q + mean_k)
        std_sym = 0.5 * (std_q + std_k)
        
        Q_context = (Q_context - mean_sym) / (std_sym + 1e-6)
        K_pulse = (K_pulse - mean_sym) / (std_sym + 1e-6)
            
        Q_geo = self.wedge(self.rope(Q_context))
        K_geo = self.wedge(self.rope(K_pulse))
        
        Attn = (Q_geo @ K_geo.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        Attn.masked_fill_(mask, float('-inf'))
        
        null_scores = self.sink_scalar.expand(B, H, T, 1)
        Attn_full = torch.cat([Attn, null_scores], dim=-1)
        probs_full = self.softmax(Attn_full)
        out = probs_full[..., :T] @ v_val + probs_full[..., T:] * self.v_null
            
        return out.transpose(1, 2).reshape(B, T, self.d_model)

# --- 3. HARNESS ---
class ToyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        # Chunk size 8 for seq_len 64
        self.attn = ContextPulseAttention(d_model, d_model, chunk_size=8)
        self.fc = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        h = self.embed(x)
        h = self.attn(h)
        return self.fc(h)

def get_batch(batch_size=32, seq_len=64, vocab_size=128, device='cpu'):
    data = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    half = seq_len // 2
    data[:, half:] = data[:, :half] 
    inputs = data[:, :-1]
    targets = data[:, 1:]
    return inputs, targets

def run_simple():
    print("--- SIMPLE TASK: Causal Chunked Context-Pulse ---")
    VOCAB = 64
    DIM = 32
    STEPS = 500
    LR = 3e-3
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = ToyTransformer(VOCAB, DIM).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR)
    
    loss_hist = []
    for i in range(STEPS):
        x, y = get_batch(seq_len=64, vocab_size=VOCAB, device=DEVICE)
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        loss.backward()
        opt.step()
        loss_hist.append(loss.item())

    plt.figure(figsize=(10, 6))
    plt.plot(loss_hist)
    plt.title("Simple Recall: Causal & Chunked")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_simple()
