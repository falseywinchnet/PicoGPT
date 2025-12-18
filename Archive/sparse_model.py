#copyright joshuah.rainstar@gmail.com 2025
#MIT with attribution

import math
import copy
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

class RoPE(nn.Module):
    """Rotary Positional Embeddings."""
    def __init__(self, dim, max_len=4096):
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

class ConvexSoftmax(nn.Module):
    """Convex LSE (Float32 Precision)."""
    def forward(self, scores):
        m, _ = scores.max(dim=-1, keepdim=True)
        y = scores - m
        ex = y.exp()
        lse = m + ex.sum(dim=-1, keepdim=True).log()
        return torch.exp(scores - lse)

class BiasedWedge(nn.Module):
    """
    Symplectic Geometry with an Escape Hatch.
    S = (A - A^T) + D
    """
    def __init__(self, head_dim, total_heads):
        super().__init__()
        self.head_dim = head_dim
        self.total_heads = total_heads
        
        # 1. Shared Skew (Global Twist): (D, D)
        self.A = nn.Parameter(torch.zeros(head_dim, head_dim))
        
        # 2. Identity Bias (Local Preservation): (TotalHeads, D)
        self.id_bias = nn.Parameter(torch.zeros(total_heads, head_dim))

    def forward(self, x):
        # x: (B, TotalHeads, T, D)
        
        # Construct S
        skew = self.A - self.A.transpose(-1, -2) # (D, D)
        diag = torch.diag_embed(self.id_bias)      # (H, D, D)
        S = skew + diag # Broadcasts to (H, D, D)
        
        # Apply Flow: x @ S
        # USE EINSUM TO PREVENT BROADCASTING ERRORS
        # b: Batch
        # h: TotalHeads (Matches S dim 0)
        # t: Time (Ignored by S)
        # d: Input Dim (Contracted)
        # e: Output Dim
        flow = torch.einsum('bhtd,hde->bhte', x, S)
        
        return x + flow

class Attention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.d_model = d_model
        # System Hierarchy
        self.n_branches = n_head 
        self.head_dim = d_model//n_head
        self.n_sub_heads = n_head
        assert d_model % self.head_dim == 0, "d_model must be divisible by head_dim"
        
        # Unified Head Dimension
        self.n_total_heads = self.n_branches * self.n_sub_heads
        
        self.scale = self.head_dim ** -0.5

        # --- 1. Projections ---
        self.W_V = nn.Linear(d_model, d_model, bias=True)
        self.W_K = nn.Linear(d_model, d_model, bias=True)
        self.W_Q_all = nn.Linear(d_model, d_model * self.n_branches, bias=False)

        # --- 2. Geometry ---
        # Ensure we use the BiasedWedge with einsum
        self.wedge = BiasedWedge(self.head_dim, self.n_total_heads)
        self.rope = RoPE(self.head_dim)

        # --- 3. Sink Parameters ---
        self.sink_scalars = nn.Parameter(torch.zeros(self.n_total_heads, 1, 1))
        self.v_nulls = nn.Parameter(torch.zeros(self.n_branches, d_model))

        # --- 4. Output ---
        self.W_O_params = nn.Parameter(torch.empty(self.n_branches, d_model, d_model))
        self.W_O_bias = nn.Parameter(torch.zeros(self.n_branches, d_model))

        nn.init.xavier_uniform_(self.W_O_params)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, T, C = x.shape
        H_tot = self.n_total_heads
        N_br = self.n_branches
        N_sh = self.n_sub_heads
        Dh = self.head_dim

        # 1. Projections
        # Q: (B, T, TotalHeads, Dh) -> (B, TotalHeads, T, Dh)
        q = self.W_Q_all(x).view(B, T, H_tot, Dh).permute(0, 2, 1, 3)

        # K: (B, T, SubHeads, Dh) -> Expand to TotalHeads
        k_base = self.W_K(x).view(B, T, N_sh, Dh).permute(0, 2, 1, 3)
        k = k_base.repeat(1, N_br, 1, 1) 


        # V: Expand to TotalHeads
        v_base = self.W_V(x).view(B, T, N_sh, Dh).permute(0, 2, 1, 3)
        v = v_base.repeat(1, N_br, 1, 1)

        # 2. Geometry
        q = self.wedge(q)
        k = self.wedge(k)
        
        q = self.rope(q)
        k = self.rope(k)

        # 3. Attention
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn_scores.masked_fill_(mask, float('-inf'))

        # Sinks
        sinks = self.sink_scalars.view(1, H_tot, 1, 1).expand(B, -1, T, -1)
        attn_scores_full = torch.cat([attn_scores, sinks], dim=-1)
        probs_full = self.softmax(attn_scores_full)

        # 4. Value Aggregation
        probs_tokens = probs_full[..., :T]
        probs_sink   = probs_full[..., T:]

        out_tokens = probs_tokens @ v

        # Sinks (Reshape v_nulls to broadcast correctly)
        # v_nulls: (Br, D) -> (Br*Sh, Dh) -> (1, H_tot, 1, Dh)
        v_null_expanded = self.v_nulls.view(N_br * N_sh, Dh).view(1, H_tot, 1, Dh)
        out_sinks = probs_sink * v_null_expanded

        context = out_tokens + out_sinks # (B, H_tot, T, Dh)

        # 5. Output
        # Recover Branch dim
        context = context.view(B, N_br, N_sh, T, Dh)
        context = context.permute(0, 1, 3, 2, 4).contiguous().view(B, N_br, T, C)
        
        # Projection & Bias (Explicit broadcast fix for Bias)
        y_proj = torch.einsum('bntc,ncd->bntd', context, self.W_O_params)
        bias = self.W_O_bias.view(1, N_br, 1, C)
        
        y = y_proj + bias

        return y.mean(dim=1)


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.use_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(ndim))
        else:
            self.register_parameter("bias", None)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.bias if self.use_bias else None
        return F.layer_norm(x, self.weight.shape, self.weight, b, 1e-5)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear( config.n_embd,4* config.n_embd, bias=config.bias)
        self.scale = math.pi / math.sqrt(3.0)

        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = x * torch.sigmoid(self.scale * x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = Attention(config.n_embd,config.n_head)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self,x):
        x1 = self.attn(self.ln_1(x))
        x = x + x1
        x = x + self.mlp(self.ln_2(x))
        return x

import torch.utils.checkpoint as checkpoint # New: Required for activation checkpointing
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, idx, targets=None):
        device = idx.device
        b, T = idx.size()        
        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        for block in self.transformer.h:
            x  = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)

            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

        import torch

@dataclass
class SSAConfig:
    # RENAMED: Distinct variable for the Sparse Chunk size
    ssa_block_size: int = 16      
    top_k: int = 8                
    sink_size: int = 64           
    alignment_alpha: float = 0.5  
    
    # GPT Context parameters
    block_size: int = 1024        
    vocab_size: int = 50304 
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

class SSA_Attention(nn.Module):
    def __init__(self, config: SSAConfig):
        super().__init__()
        self.d_model = config.n_embd
        
        # --- Hierarchy ---
        self.n_branches = config.n_head
        self.head_dim = self.d_model // self.n_branches
        self.n_sub_heads = self.n_branches
        
        self.n_total_heads = self.n_branches * self.n_sub_heads
        self.scale = self.head_dim ** -0.5

        # --- 1. Projections ---
        self.W_V = nn.Linear(self.d_model, self.d_model, bias=True)
        self.W_K = nn.Linear(self.d_model, self.d_model, bias=True)
        self.W_Q_all = nn.Linear(self.d_model, self.d_model * self.n_branches, bias=False)

        # --- 2. Geometry ---
        self.wedge = BiasedWedge(self.head_dim, self.n_total_heads)
        self.rope = RoPE(self.head_dim)

        # --- 3. Sink Parameters ---
        self.sink_scalars = nn.Parameter(torch.zeros(self.n_total_heads, 1, 1))
        self.v_nulls = nn.Parameter(torch.zeros(self.n_branches, self.d_model))

        # --- 4. Output ---
        self.W_O_params = nn.Parameter(torch.empty(self.n_branches, self.d_model, self.d_model))
        self.W_O_bias = nn.Parameter(torch.zeros(self.n_branches, self.d_model))
        nn.init.xavier_uniform_(self.W_O_params)
        self.softmax = nn.Softmax(dim=-1)
        
        # --- 5. SSA Config ---
        # CORRECTED: Use the distinct SSA block size variable
        self.ssa_block_size = config.ssa_block_size 
        self.top_k = config.top_k
        self.sink_size = config.sink_size

    def get_qkv(self, x):
        B, T, C = x.shape
        H_tot = self.n_total_heads
        N_br = self.n_branches
        N_sh = self.n_sub_heads
        Dh = self.head_dim

        q = self.W_Q_all(x).view(B, T, H_tot, Dh).permute(0, 2, 1, 3)

        k_base = self.W_K(x).view(B, T, N_sh, Dh).permute(0, 2, 1, 3)
        k = k_base.repeat(1, N_br, 1, 1)

        v_base = self.W_V(x).view(B, T, N_sh, Dh).permute(0, 2, 1, 3)
        v = v_base.repeat(1, N_br, 1, 1) 
        
        q = self.wedge(q)
        k = self.wedge(k)
        q = self.rope(q)
        k = self.rope(k)
        
        return q, k, v

    def project_output(self, context, B, T, C):
        N_br = self.n_branches
        N_sh = self.n_sub_heads
        Dh = self.head_dim
        
        context = context.view(B, N_br, N_sh, T, Dh)
        context = context.permute(0, 1, 3, 2, 4).contiguous().view(B, N_br, T, C)
        
        y_proj = torch.einsum('bntc,ncd->bntd', context, self.W_O_params)
        bias = self.W_O_bias.view(1, N_br, 1, C)
        
        y = y_proj + bias 
        return y

    def forward_full_logic(self, q, k, v, T):
        B, H_tot, _, Dh = q.shape
        
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=q.device), diagonal=1).bool()
        attn_scores.masked_fill_(mask, float('-inf'))

        sinks = self.sink_scalars.view(1, H_tot, 1, 1).expand(B, -1, T, -1)
        attn_scores_full = torch.cat([attn_scores, sinks], dim=-1)
        probs_full = self.softmax(attn_scores_full)

        probs_tokens = probs_full[..., :T]
        probs_sink   = probs_full[..., T:]

        out_tokens = probs_tokens @ v

        v_null_expanded = self.v_nulls.view(1, H_tot, 1, Dh)
                          
        out_sinks = probs_sink * v_null_expanded
        context = out_tokens + out_sinks
        
        return context

    def forward_sparse_logic(self, q, k, v, T):
        """
        SSA Logic applied to the TotalHeads.
        Includes padding fix for inference robustness.
        """
        B, H_tot, T_orig, Dh = q.shape
        
        # --- 0. Pad to multiple of ssa_block_size ---
        # Use the specific SSA block size for calculation
        blk_sz = self.ssa_block_size
        
        pad_len = (blk_sz - (T_orig % blk_sz)) % blk_sz
        
        if pad_len > 0:
            q_pad = F.pad(q, (0, 0, 0, pad_len))
            k_pad = F.pad(k, (0, 0, 0, pad_len))
        else:
            q_pad, k_pad = q, k
            
        B, H, T_pad, _ = q_pad.shape
        num_blocks = T_pad // blk_sz
        
        # 1. Pooling
        k_blocked = k_pad.view(B, H_tot, num_blocks, blk_sz, Dh)
        k_means = k_blocked.mean(dim=3)
        
        q_blocked = q_pad.view(B, H_tot, num_blocks, blk_sz, Dh)
        q_means = q_blocked.mean(dim=3)
        
        # 2. Scoring
        block_scores = (q_means @ k_means.transpose(-2, -1))
        
        block_mask = torch.triu(torch.ones(num_blocks, num_blocks, device=q.device), diagonal=1).bool()
        block_scores.masked_fill_(block_mask, float('-inf'))
        
        # 3. Selection
        curr_k = min(self.top_k, num_blocks)
        if curr_k > 0:
            _, topk_indices = torch.topk(block_scores, k=curr_k, dim=-1)
            keep_block_mask = torch.zeros(B, H_tot, num_blocks, num_blocks, device=q.device, dtype=torch.bool)
            keep_block_mask.scatter_(3, topk_indices, True)
            
            # Upsample using blk_sz
            token_mask = keep_block_mask.repeat_interleave(blk_sz, dim=2).repeat_interleave(blk_sz, dim=3)
        else:
            # Fallback for empty/singularity
            token_mask = torch.zeros(B, H_tot, T_pad, T_pad, device=q.device, dtype=torch.bool)
        
        # Sinks & Locals
        token_mask[..., :, :self.sink_size] = True
        local_band = torch.ones(T_pad, T_pad, device=q.device).tril(0).triu(-blk_sz).bool()
        token_mask = token_mask | local_band.view(1, 1, T_pad, T_pad)
        
        # 4. Attention
        scores = (q_pad @ k_pad.transpose(-2, -1)) * self.scale
        causal_mask = torch.triu(torch.ones(T_pad, T_pad, device=q.device), diagonal=1).bool()
        
        final_mask = causal_mask | (~token_mask)
        
        # Mask out padding columns if they exist
        if pad_len > 0:
            padding_mask = torch.zeros(T_pad, T_pad, device=q.device, dtype=torch.bool)
            padding_mask[:, T_orig:] = True 
            final_mask = final_mask | padding_mask

        scores.masked_fill_(final_mask, float('-inf'))
        
        # 5. Output with Sinks
        sinks = self.sink_scalars.view(1, H_tot, 1, 1).expand(B, -1, T_pad, -1)
        scores_full = torch.cat([scores, sinks], dim=-1)
        probs_full = self.softmax(scores_full)
        
        probs_tokens = probs_full[..., :T_pad]
        probs_sink   = probs_full[..., T_pad:]
        
        if pad_len > 0:
            v_pad = F.pad(v, (0, 0, 0, pad_len))
        else:
            v_pad = v
            
        out_tokens = probs_tokens @ v_pad
        
        v_null_expanded = self.v_nulls.view(1, H_tot, 1, Dh)
        context_pad = out_tokens + (probs_sink * v_null_expanded)
        
        # Unpad
        context = context_pad[..., :T_orig, :]
        return context

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.get_qkv(x)
        align_loss = 0.0
        
        if self.training:
            if torch.rand(1).item() < 0.5:
                # Main: Full
                context_main = self.forward_full_logic(q, k, v, T)
                with torch.no_grad():
                    context_aux = self.forward_sparse_logic(q, k, v, T)
                
                y = self.project_output(context_main, B, T, C)
                y_aux = self.project_output(context_aux, B, T, C)
                align_loss = F.smooth_l1_loss(y, y_aux.detach())
                
            else:
                # Main: Sparse
                context_main = self.forward_sparse_logic(q, k, v, T)
                with torch.no_grad():
                    context_aux = self.forward_full_logic(q, k, v, T)
                
                y = self.project_output(context_main, B, T, C)
                y_aux = self.project_output(context_aux, B, T, C)
                align_loss = F.smooth_l1_loss(y, y_aux.detach())
        else:
            context_main = self.forward_sparse_logic(q, k, v, T)
            y = self.project_output(context_main, B, T, C)
            
        return y.mean(dim=1), align_loss


class SSA_Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SSA_Attention(config) # Replaced
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.alpha = config.alignment_alpha

    def forward(self, x):
        # Handle tuple return from SSA_Attention
        attn_out, align_loss = self.attn(self.ln_1(x))
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        
        # We must return the loss to be aggregated at the model level
        return x, align_loss * self.alpha

class SSA_GPT(GPT):
    def __init__(self, config):
        # CORRECT FIX: Explicitly initialize the base nn.Module
        # This sets up self._modules, self._parameters, etc.
        nn.Module.__init__(self) 
        
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        
        # Now we can safely assign modules
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([SSA_Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Report parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    # We inherit forward and get_num_params from GPT, 
    # but we must override forward if we want to handle the tuple return 
    # (logits, loss) correctly regarding the alignment loss aggregation.
    
    def forward(self, idx, targets=None):
        device = idx.device
        b, T = idx.size()
        x = self.transformer.wte(idx)
        x = self.transformer.drop(x) # Explicitly use dropout if desired
        
        total_align_loss = 0.0
        
        for block in self.transformer.h:
            x, al_loss = block(x)
            total_align_loss += al_loss 
            
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            
            # SSA Objective: L = L_mode + alpha * L_alignment
            final_loss = ce_loss + total_align_loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            final_loss = None

        return logits, final_loss
