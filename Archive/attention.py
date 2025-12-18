import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import math

# --- 1. Locked Primitives ---
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
        # x: (B, H, T, D).
        # To use matmul with S (H, D, D), we need to align D.
        # (B, H, T, D) @ (1, H, D, D) -> (B, H, T, D)
        # This works if D is the last dim.
        S = self.A - self.A.transpose(-1, -2)
        S_broadcast = S.unsqueeze(0) # (1, H, D, D)
        flow = torch.matmul(x, S_broadcast)
        return x + flow

class ConvexSoftmax(nn.Module):
    def forward(self, scores):
        m, _ = scores.max(dim=-1, keepdim=True)
        y = scores - m
        ex = y.exp()
        lse = m + ex.sum(dim=-1, keepdim=True).log()
        return torch.exp(scores - lse)

def build_alpert_basis(block_size, poly_order=1, device='cpu'):
    t = torch.linspace(-1, 1, block_size, device=device)
    p0 = torch.ones_like(t)
    basis_list = [p0]
    if poly_order >= 1: basis_list.append(t)
    if poly_order >= 2: basis_list.append(3 * t**2 - 1)
    W = torch.stack(basis_list, dim=1)
    W = F.normalize(W, p=2, dim=0)
    return W

# --- 2. Post-Wedge Alpert Model ---
class ContextPulseAttention(nn.Module):
    def __init__(self, d_model, n_head, chunk_size=16, poly_order=1):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.chunk_size = chunk_size
        self.poly_order = poly_order
        self.scale = self.head_dim ** -0.5

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        
        nn.init.orthogonal_(self.W_Q.weight)
        nn.init.normal_(self.W_K.weight, mean=0.0, std=0.02)
        nn.init.xavier_normal_(self.W_V.weight)

        self.gate_proj = nn.Linear(d_model, d_model)
        self.jump_scale = nn.Parameter(torch.tensor(1.0))
        self.jump_bias = nn.Parameter(torch.tensor(0.0))
        
        self.rope = RoPE(self.head_dim)
        self.wedge = WedgeTransform(d_model, n_head)
        self.softmax = ConvexSoftmax()
        
        self.sink_scalar = nn.Parameter(torch.zeros(1, n_head, 1, 1))
        self.v_null = nn.Parameter(torch.zeros(1, n_head, 1, self.head_dim)) 
        
        basis = build_alpert_basis(chunk_size, poly_order)
        self.register_buffer('alpert_basis', basis)

    def forward(self, x):
        B, T, C = x.shape
        H, Dh = self.n_head, self.head_dim
        
        # 1. Projections
        q_raw = self.W_Q(x).view(B, T, H, Dh).transpose(1, 2)
        k_raw = self.W_K(x).view(B, T, H, Dh).transpose(1, 2)
        v_val = self.W_V(x).view(B, T, H, Dh).transpose(1, 2)
        gate = torch.sigmoid(self.gate_proj(x)).view(B, T, H, Dh).transpose(1, 2)
        
        # 2. Symbiotic Norm (Global Pointwise)
        mean_q = q_raw.mean(dim=-1, keepdim=True)
        std_q = q_raw.std(dim=-1, keepdim=True)
        mean_k = k_raw.mean(dim=-1, keepdim=True)
        std_k = k_raw.std(dim=-1, keepdim=True)
        mean_sym = 0.5 * (mean_q + mean_k)
        std_sym = 0.5 * (std_q + std_k)
        
        q_norm = (q_raw - mean_sym) / (std_sym + 1e-6)
        k_norm = (k_raw - mean_sym) / (std_sym + 1e-6)
        
        # 3. Geometry (Global Application - Pre-Integration)
        # RoPE then Wedge
        q_geo = self.wedge(self.rope(q_norm))
        k_geo = self.wedge(self.rope(k_norm))
        
        # 4. Chunk Loop
        num_chunks = math.ceil(T / self.chunk_size)
        Q_output_list = []
        history_coeffs_cache = [] # Stores Q_geo moments
        
        W_basis = self.alpert_basis 
        
        for i in range(num_chunks):
            t_start = i * self.chunk_size
            t_end = min((i + 1) * self.chunk_size, T)
            
            # Slices of Geometrized Tensors
            q_chunk = q_geo[:, :, t_start:t_end, :]
            k_chunk = k_geo[:, :, t_start:t_end, :]
            gate_chunk = gate[:, :, t_start:t_end, :]
            
            current_L = q_chunk.size(2)
            if current_L < self.chunk_size:
                W_curr = build_alpert_basis(current_L, self.poly_order, device=x.device)
            else:
                W_curr = W_basis
            
            # Encode K_geo Moments
            k_coeffs = torch.einsum('bhld, lk -> bhkd', k_chunk, W_curr)
            
            # Wave Retrieval
            if len(history_coeffs_cache) > 0:
                history_stack = torch.cat(history_coeffs_cache, dim=2)
                
                # Score: K_moments vs Q_moments
                resonance_scores = torch.einsum('bhkd, bhnkd -> bhn', k_coeffs, history_stack) * self.scale
                resonance_scores = resonance_scores.unsqueeze(-2)
                resonance_weights = F.softmax(resonance_scores, dim=-1)
                
                # Retrieve Q Moments
                coeffs_retrieved = torch.einsum('bhn, bhnkd -> bhkd', resonance_weights.squeeze(2), history_stack)
                
                # Reconstruct Trajectory
                q_reconstructed = torch.einsum('bhkd, lk -> bhld', coeffs_retrieved, W_curr)
                
                # Resonance Gate
                max_resonance = resonance_scores.max(dim=-1, keepdim=True).values
                jump_gate = torch.sigmoid(max_resonance * self.jump_scale + self.jump_bias)
                jump_gate = jump_gate.expand(-1, -1, current_L, -1)
                
                if 'last_state' in locals():
                    continuity_stream = last_state.expand(-1, -1, current_L, -1)
                    q_injected = (1.0 - jump_gate) * continuity_stream + jump_gate * q_reconstructed
                else:
                    q_injected = q_reconstructed * jump_gate
            else:
                q_injected = torch.zeros_like(q_chunk)

            # Integration
            # Note: We are integrating Q_geo. 
            # Logic: Q_geo_t = Q_geo_{t-1} + gate * Q_geo_input ??
            # Original logic: Q_raw_t = Q_raw_{t-1} + gate * Q_raw_input.
            # By linearity, if we integrate Wedged inputs, we get Wedged context.
            # So we integrate q_chunk (which is q_geo_input).
            
            masked_input = gate_chunk * q_chunk
            q_integrated = torch.cumsum(masked_input, dim=2)
            q_chunk_out = q_integrated + q_injected
            
            Q_output_list.append(q_chunk_out)
            
            # Cache Moments of the Output Context
            q_out_coeffs = torch.einsum('bhld, lk -> bhkd', q_chunk_out, W_curr)
            history_coeffs_cache.append(q_out_coeffs.unsqueeze(2))
            last_state = q_chunk_out[:, :, -1:, :]
            
        Q_final = torch.cat(Q_output_list, dim=2)
        
        # 5. Readout
        # Q_final is already Geometrized and Integrated.
        # K_geo is the Pulse.
        
        Attn = (Q_final @ k_geo.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        Attn.masked_fill_(mask, float('-inf'))
        
        null_scores = self.sink_scalar.expand(B, H, T, 1)
        Attn_full = torch.cat([Attn, null_scores], dim=-1)
        probs_full = self.softmax(Attn_full)
        
        out = probs_full[..., :T] @ v_val + probs_full[..., T:] * self.v_null
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_O(out)

# --- Harness ---
class ToyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_head):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.attn = ContextPulseAttention(d_model, n_head=n_head, chunk_size=16)
        self.fc = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        h = self.embed(x)
        h = self.attn(h)
        return self.fc(h)

def get_gap_copy_batch(batch_size=32, seq_len=256, vocab_size=64, device='cpu'):
    START_TOKEN = 0
    STOP_TOKEN = 1
    data = torch.randint(2, vocab_size, (batch_size, seq_len), device=device)
    pattern_len = 16
    for b in range(batch_size):
        pattern = torch.randint(2, vocab_size, (pattern_len,), device=device)
        data[b, 0] = START_TOKEN
        data[b, 1:1+pattern_len] = pattern
        data[b, 1+pattern_len] = STOP_TOKEN
        trigger_start = torch.randint(seq_len // 2, seq_len - pattern_len - 1, (1,)).item()
        data[b, trigger_start] = START_TOKEN
        data[b, trigger_start+1 : trigger_start+1+pattern_len] = pattern
    inputs = data[:, :-1]
    targets = data[:, 1:]
    return inputs, targets

def run_post_wedge_hard():
    print("--- POST-WEDGE ALPERT: 4 Heads, 64 Dim ---")
    VOCAB = 64
    DIM = 64
    HEADS = 4
    STEPS = 1500
    SEQ_LEN = 256
    LR = 3e-3
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = ToyTransformer(VOCAB, DIM, HEADS).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR)
    
    loss_hist = []
    
    for i in range(STEPS):
        x, y = get_gap_copy_batch(seq_len=SEQ_LEN, vocab_size=VOCAB, device=DEVICE)
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        loss.backward()
        opt.step()
        loss_hist.append(loss.item())
        
        if i % 100 == 0:
            print(f"Step {i}: Loss {loss.item():.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(loss_hist)
    plt.title("Post-Wedge Alpert Retrieval")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_post_wedge_hard()
