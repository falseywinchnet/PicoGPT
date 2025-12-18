#copyright 2025 Joshuah Rainstar joshuah.rainstar@gmail.com 
#MIT License Terms Apply

import torch
import torch.nn as nn
import torch.nn.functional as F

class ZGate(nn.Module):
    def __init__(self):
        super().__init__()
    #original math preserved in forward_exact, forward is faster but some small rounding changes
    #20% better and 20% faster than GELU on tail classes if implemented efficiently
    #gotta be more responsible with your ln, lr, clipping, ie, theres more late-range spiking.
    #what is this? basically its a softer relu, but it behaves like a relu. softplus doesnt.
    #will collapse just like a relu in terms of dead neurons. beware.
    #better gates need to exist, this is just a preliminary experiment. 
    #joshuah.rainstar@gmail.com wrote this gate
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reuse intermediates; remove redundant ReLU; collapse z to 0.5*sp^2
        sp = x.clamp_min(0)
        sp2 = sp * sp
        x2 = x * x
        q1 = 1.0 / torch.pow(1.0 + 0.5 * x2, 0.5)  # keep identical math to preserve results
        out = F.relu(sp2 * x / (x2 + q1))  # since (0.5*sp^2)*(2*x)/(x2+q1) == sp^2*x/(x2+q1)
        return out
        
    def forward_exact(self, x: torch.Tensor) -> torch.Tensor:
        sp = F.relu(x)
        sa = F.relu(0.5 * x)
        one_minus_sa = 1.0 - sa
        ba = sa * one_minus_sa
        z = sp - 2.0 * ba
        x2_pow = x**2
        q1 = 1 / (1 + 0.5 * x2_pow)**0.5
        denom = (x * x) + q1  
        q = 2.0 * x / denom
        y = z * q
        return F.relu(y)

#in the following experiments:
#the ideal gate survives dead neuron stress test, approximates manifolds well without straight striping(Zgate does well here)
#does well at recovering tail end class behavior, has smooth higher order derivatives,
#and resists hysteris well. Gelu resists hysteris and dead neuron and does acceptably on tail end, but tends to collapse on manifolds.
#zgate does not resist hysteris or dead neuron but does well on tail end and manifold approximation.
'''
# Dead-neuron stress test: ZGate vs GELU vs ReLU
# Goal: strongly trigger "dead ReLU" by biasing first layer negative & using small weight init.
# Architecture stays: layer -> activation -> layer -> head. Optimizer: AdamW.

import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import AdamW
import numpy as np, matplotlib.pyplot as plt

# ---- setup ----
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)

# synthetic dataset (moderately non-linear; balanced classes)
N = 24000
num_classes = 6
d_in = 64
X = torch.randn(N, d_in)
# add non-linear target with feature interactions
W_true = torch.randn(d_in, num_classes)
quad = (X[:, :8]**2) @ torch.randn(8, num_classes) * 0.2
logits = X @ W_true + quad + 0.2*torch.randn(N, num_classes)
Y = logits.argmax(dim=1)

# split
perm = torch.randperm(N)
X, Y = X[perm], Y[perm]
ntr = int(0.8*N)
Xtr, Ytr = X[:ntr].to(device), Y[:ntr].to(device)
Xva, Yva = X[ntr:].to(device), Y[ntr:].to(device)

# ---- activations ----
class ZGate(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sp = x.clamp_min(0)
        sp2 = sp * sp
        x2 = x * x
        q1 = 1.0 / torch.pow(1.0 + 0.5 * x2, 0.5)
        out = F.relu(sp2 * x / (x2 + q1))
        return out

# ---- model ----
class MLP(nn.Module):
    def __init__(self, act="gelu", h=512):
        super().__init__()
        self.fc1 = nn.Linear(d_in, h)
        self.act = nn.GELU() if act=="gelu" else ZGate() if act=="zgate" else nn.ReLU()
        self.fc2 = nn.Linear(h, h)
        self.head = nn.Linear(h, num_classes)
        # **dead-neuron trigger**: tiny weights + strongly negative bias in the FIRST layer only
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.constant_(self.fc1.bias, -3.0)       # pushes most preacts negative at start
        # vanilla init for others
    def forward(self, x):
        pre1 = F.linear(x, self.fc1.weight, self.fc1.bias)  # save preactivations for diagnostics
        x = self.act(pre1)
        x = self.fc2(x)                                     # per spec: no extra activation
        return self.head(x), pre1

# ---- train/eval ----
def run(act, epochs=60, bs=1024, lr=3e-3):
    model = MLP(act=act).to(device)
    # keep it simple: no weight decay (don’t let biases drift toward zero and accidentally “revive” ReLU)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=2e-3)

    steps = int(np.ceil(Xtr.size(0)/bs))
    acc_hist, dead_hist, gradb1_hist = [], [], []

    for ep in range(epochs):
        model.train()
        idx = torch.randperm(Xtr.size(0), device=device)
        for b in range(steps):
            j = idx[b*bs:(b+1)*bs]
            logits, pre1 = model(Xtr[j])
            loss = F.cross_entropy(logits, Ytr[j])
            opt.zero_grad(); loss.backward()
            # record grad magnitude of first-layer bias BEFORE the step
            g = model.fc1.bias.grad.detach()
            gradb1_hist.append(g.abs().mean().item())
            opt.step()

        # diagnostics on the full VAL set
        model.eval()
        with torch.no_grad():
            logits, pre1 = model(Xva)
            pred = logits.argmax(1)
            acc = (pred == Yva).float().mean().item()
            # fraction of *negative preactivations* in layer1 (proxy for deadness trigger zone)
            dead_frac = (pre1 <= 0).float().mean().item()
        acc_hist.append(acc); dead_hist.append(dead_frac)

    return {"acc":acc_hist, "dead":dead_hist, "gradb1":gradb1_hist}

res_relu  = run("relu")
res_gelu  = run("gelu")
res_zgate = run("zgate")

# ---- plot ----
E = len(res_relu["acc"]); xs = np.arange(1, E+1)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.plot(xs, res_relu["acc"],  label="ReLU")
plt.plot(xs, res_gelu["acc"],  label="GELU")
plt.plot(xs, res_zgate["acc"], label="ZGate")
plt.title("Validation accuracy"); plt.xlabel("Epoch"); plt.ylabel("Acc"); plt.legend()

plt.subplot(1,3,2)
plt.plot(xs, res_relu["dead"],  label="ReLU")
plt.plot(xs, res_gelu["dead"],  label="GELU")
plt.plot(xs, res_zgate["dead"], label="ZGate")
plt.title("Frac. negative preacts in layer1"); plt.xlabel("Epoch"); plt.ylabel("Fraction"); plt.legend()

# smooth grad curve for readability
def smooth(v, k=200):
    v = np.array(v, float); k = max(1, k)
    ker = np.ones(k)/k
    return np.convolve(v, ker, mode="valid")

plt.subplot(1,3,3)
plt.plot(smooth(res_gelu["gradb1"]),  label="GELU")
plt.plot(smooth(res_zgate["gradb1"]), label="ZGate")
plt.plot(smooth(res_relu["gradb1"]),  label="ReLU")

plt.title("Mean |grad| of first-layer bias"); plt.xlabel("Step"); plt.ylabel("|grad|"); plt.legend()

plt.tight_layout(); plt.show()


# Harder, higher-rank, more realistic tail benchmark (MLP: layer → activation → layer → head)
# Assumes ZGate is already defined above.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ----- config -----
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)

K = 200                  # number of classes ("entities")
R = 6                    # number of relations / contexts per sample
total_train = 40000      # total train samples (heavy-tailed across classes)
noise_std = 0.5          # covariate noise
epochs = 600
batch_size = 1024
lr = 2e-3

# feature geometry (bilinear -> higher effective rank)
d_in = 128
d_e  = 32
d_r  = 8
d_er = d_e * d_r  # bilinear term dimension

# ----- long-tail class counts (Zipf) -----
alpha = 1.5
w = 1.0 / (np.arange(1, K+1) ** alpha)
w = w / w.sum()
counts = np.maximum(1, np.floor(w * total_train)).astype(int)
# ensure exact total
while counts.sum() < total_train: counts[np.random.randint(K)] += 1

# head/tail split indices (top/bottom 20% by frequency)
order = np.argsort(-counts)
head_idx = set(order[:K//5])
tail_idx = set(order[-K//5:])

# ----- entity / relation embeddings -----
E = torch.randn(K, d_e)           # entity/fact embeddings
Rmat = torch.randn(R, d_r)        # relation embeddings

# linear maps for generative process
A = torch.randn(d_in, d_e)
B = torch.randn(d_in, d_r)
M = torch.randn(d_in, d_er)       # projects (e ⊗ r) into input space

def sample_class_data(k, n):
    # draw relations uniformly to create realistic intra-class variation
    rel_ids = torch.randint(0, R, (n,))
    e = E[k]                        # (d_e,)
    r = Rmat[rel_ids]               # (n, d_r)
    e_tile = e.expand(n, -1)        # (n, d_e)
    # bilinear cross-term (Kron) -> projected
    kron = torch.einsum("ni,nj->nij", e_tile, r).reshape(n, d_er)  # (n, d_e*d_r)
    x = (e_tile @ A.T) + (r @ B.T) + (kron @ M.T)                  # (n, d_in)
    x = x + noise_std * torch.randn_like(x)
    y = torch.full((n,), k, dtype=torch.long)
    return x, y

# ----- build train/val -----
X_tr, Y_tr, X_val, Y_val = [], [], [], []
for k, n in enumerate(counts):
    n_val = max(5, int(0.2 * n))
    n_tr  = max(5, n - n_val)
    x, y = sample_class_data(k, n_tr + n_val)
    X_tr.append(x[:n_tr]); Y_tr.append(y[:n_tr])
    X_val.append(x[n_tr:]); Y_val.append(y[n_tr:])

X_tr = torch.cat(X_tr).to(device)
Y_tr = torch.cat(Y_tr).to(device)
X_val = torch.cat(X_val).to(device)
Y_val = torch.cat(Y_val).to(device)

# per-class indices for head/tail eval (on VAL)
val_indices_by_class = defaultdict(list)
for i, c in enumerate(Y_val.cpu().tolist()):
    val_indices_by_class[c].append(i)
val_head = torch.tensor([i for c in head_idx for i in val_indices_by_class[c]], device=device)
val_tail = torch.tensor([i for c in tail_idx for i in val_indices_by_class[c]], device=device)

# ----- model (layer → activation → layer → head) -----
class MLP(nn.Module):
    def __init__(self, act="gelu", h1=512, h2=512):
        super().__init__()
        self.fc1 = nn.Linear(d_in, h1)
        self.act = ZGate() if act == "zgate" else nn.GELU()
        self.fc2 = nn.Linear(h1, h2)   # no activation here (per spec)
        self.head = nn.Linear(h2, K)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.head(x)

def run(act):
    model = MLP(act=act).to(device)
    opt = AdamW(model.parameters(), lr=lr)
    N = X_tr.size(0)
    idx = torch.arange(N, device=device)

    overall_hist, head_hist, tail_hist = [], [], []

    for _ in range(epochs):
        # shuffle minibatches
        perm = idx[torch.randperm(N)]
        for b in range(0, N, batch_size):
            j = perm[b:b+batch_size]
            logits = model(X_tr[j])
            loss = F.cross_entropy(logits, Y_tr[j])
            opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():
            logits = model(X_val)
            pred = logits.argmax(1)
            overall = (pred == Y_val).float().mean().item()
            if len(val_head) > 0:
                head = (pred[val_head] == Y_val[val_head]).float().mean().item()
            else:
                head = float("nan")
            if len(val_tail) > 0:
                tail = (pred[val_tail] == Y_val[val_tail]).float().mean().item()
            else:
                tail = float("nan")

        overall_hist.append(overall)
        head_hist.append(head)
        tail_hist.append(tail)

    return overall_hist, head_hist, tail_hist

# ----- run both activations -----
hist_g_over, hist_g_head, hist_g_tail = run("gelu")
hist_z_over, hist_z_head, hist_z_tail = run("zgate")

# ----- plot -----
xs = np.arange(1, epochs+1)
plt.figure(figsize=(9,4))

plt.subplot(1,3,1)
plt.plot(xs, hist_g_over, label="GELU")
plt.plot(xs, hist_z_over, label="ZGate")
plt.title("Overall acc"); plt.xlabel("Epoch"); plt.ylabel("Acc"); plt.legend()

plt.subplot(1,3,2)
plt.plot(xs, hist_g_head, label="GELU")
plt.plot(xs, hist_z_head, label="ZGate")
plt.title("Head acc (top 20%)"); plt.xlabel("Epoch")

plt.subplot(1,3,3)
plt.plot(xs, hist_g_tail, label="GELU")
plt.plot(xs, hist_z_tail, label="ZGate")
plt.title("Tail acc (bottom 20%)"); plt.xlabel("Epoch")

plt.tight_layout(); plt.show()


# %%
# PyTorch + matplotlib demos comparing three activations:
# 1) nn.GELU
# 2) softplus(x) + 1  (implemented as torch.logaddexp(x + 1, 1))
# 3) nn.ReLU  (baseline)
#
# Experiments:
# A) Tail-end class learning (imbalanced classification) + decision field + latent PCA
# B) Descent smoothness (regression) + loss/grad curves
# C) Difficult manifold approximation (2D field) + heatmaps + 1D slice + latent PCA
#
# Outputs:
# - Multiple matplotlib figures (each chart its own figure)
# - Printed metrics table
# - CSV: ./mlp_activation_comparison_metrics.csv

import torch
import torch.nn as nn

@torch.jit.script
def _smoothstep_cinf(t: torch.Tensor, k: float):
    # C∞ smoothstep: s(t) in [0,1] with all derivatives 0 at t=0 and t=1
    s = torch.zeros_like(t)
    m = (t > 0.0) & (t < 1.0)
    tm = t[m]
    # exp(-k/t) and exp(-k/(1-t)); numerically stable because we mask to (0,1)
    a = torch.exp(-k / (tm + 1e-12))
    b = torch.exp(-k / (1.0 - tm + 1e-12))
    s_val = a / (a + b)
    s[m] = s_val
    s = torch.where(t >= 1.0, torch.ones_like(s), s)
    return s  # t<=0 already 0
     
class SReLUInf(nn.Module):
    """
    C∞ Smooth ReLU with exact linear regions outside [-eps, eps].
    - eps: half-width of the smooth transition band
    - k:   sharpness of the C∞ bridge (bigger => sharper)
    - negative_slope: optional left linear slope (0 == ReLU; >0 -> smooth leaky variant)
    """
    def __init__(self, eps: float = 0.5, k: float = 1.5, negative_slope: float = 0.0):
        super().__init__()
        self.eps = float(eps)
        self.k = float(k)
        self.negative_slope = float(negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = self.eps
        k = self.k
        # Left & right linear targets
        left = self.negative_slope * x
        right = x

        # Regions
        y_left  = left
        y_right = right

        # Smooth bridge: blend left/right lines with C∞ step
        # Map x in (-eps, eps) -> t in (0,1)
        t = (x + eps) / (2.0 * eps)
        s = _smoothstep_cinf(t, k)
        y_mid = (1.0 - s) * left + s * right

        # Exact piecewise selection (matches values & all derivatives at boundaries)
        y = torch.where(x <= -eps, y_left,
            torch.where(x >=  eps, y_right, y_mid))
        return y


import math
import random
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt

# Reproducibility
SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")

# ---------- Utilities ----------
class ZLQGate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # smooth gate
        sp = F.relu(x)
        sa = F.relu(0.5 * x)
        ba = sa * (1.0 - sa)
        z = sp - 2.0 * ba 
        q1 = 1 / (1 + 0.5 * x**2)**0.5
        q = 2.0 * x / (x * x + q1)  # define q(0)=0 by continuity

        return F.relu(z* q)


def make_mlp(in_dim: int, hidden: int, out_dim: int, activation: nn.Module):
    # 2-layer MLP with the activation reused
    layers = [
        nn.Linear(in_dim, hidden),
        activation,
        nn.Linear(hidden, hidden),
        activation,
        nn.Linear(hidden, out_dim),
    ]
    return nn.Sequential(*layers)

def grad_l2_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += (p.grad.detach()**2).sum().item()
    return math.sqrt(total)

def pca2d(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:2].T  # top-2 components
    return Xc @ W

def capture_hidden_2nd_activation_features(model: nn.Sequential, X: np.ndarray) -> np.ndarray:
    """
    Capture activations AFTER the second activation layer (index 3).
    IMPORTANT: forward hooks that return a value REPLACE the output.
    We store and return nothing to avoid breaking the graph/type.
    """
    holder = {}

    def _hook(module, inp, out):
        holder["z"] = out.detach().cpu().numpy()
        # Do NOT return anything; returning would replace 'out'!

    handle = model[3].register_forward_hook(_hook)
    try:
        with torch.no_grad():
            _ = model(torch.from_numpy(X.astype(np.float32)).to(device))
    finally:
        handle.remove()

    return holder["z"]
# ---------- Simple metrics (no sklearn) ----------
def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(int)
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    s = y_score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            avg = 0.5 * (i + j + 2)
            ranks[order[i:j+1]] = avg
        i = j + 1
    sum_pos = ranks[y_true == 1].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)

def average_precision_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(int)
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    tp = 0
    fp = 0
    precisions = []
    recalls = []
    n_pos = (y_true == 1).sum()
    if n_pos == 0:
        return float("nan")
    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / n_pos)
    ap = 0.0
    prev_recall = 0.0
    for p, r in zip(precisions, recalls):
        ap += p * max(r - prev_recall, 0.0)
        prev_recall = r
    return float(ap)

def recall_at_fixed_precision(y_true: np.ndarray, y_score: np.ndarray, target_precision: float = 0.8) -> float:
    y_true = y_true.astype(int)
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    tp = 0
    fp = 0
    best_recall = 0.0
    n_pos = (y_true == 1).sum()
    if n_pos == 0:
        return float("nan")
    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp)
        recall = tp / n_pos
        if precision >= target_precision:
            best_recall = max(best_recall, recall)
    return float(best_recall)

# ---------- Experiment A: Tail-end class learning ----------
def make_imbalanced_dataset(n_major=6000, n_minor=400, noise=0.7, shift=2.5) -> Tuple[np.ndarray, np.ndarray]:
    # Majority (y=0): two overlapping blobs near origin
    m1 = np.random.randn(n_major//2, 2).astype(np.float32) + np.array([0.0, 0.0], np.float32)
    m2 = np.random.randn(n_major - n_major//2, 2).astype(np.float32) + np.array([1.0, -0.5], np.float32)
    X_major = np.vstack([m1, m2]).astype(np.float32)
    # Minority (y=1): a small blob in the tail
    X_minor = (np.random.randn(n_minor, 2) * noise + np.array([shift, shift])).astype(np.float32)
    X = np.vstack([X_major, X_minor]).astype(np.float32)
    y = np.hstack([np.zeros(len(X_major), dtype=np.int64), np.ones(len(X_minor), dtype=np.int64)])
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]

def train_classifier(X_train, y_train, X_val, y_val, activation_name: str, activation: nn.Module,
                     epochs=60, batch_size=256, lr=1e-3):
    model = make_mlp(2, 64, 1, activation).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    dl = data.DataLoader(data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).float()),
                         batch_size=batch_size, shuffle=True)

    losses = []
    grad_norms = []
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device).view(-1, 1)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            gnorm = grad_l2_norm(model)
            opt.step()
            losses.append(loss.item())
            grad_norms.append(gnorm)

    # Evaluate
    model.eval()
    with torch.no_grad():
        val_logits = model(torch.from_numpy(X_val).to(device)).cpu().numpy().reshape(-1)
    y_score = 1 / (1 + np.exp(-val_logits))
    y_true = y_val.astype(int)

    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    r80 = recall_at_fixed_precision(y_true, y_score, 0.8)

    # Decision field
    xmin, ymin = X_train.min(axis=0) - 1.0
    xmax, ymax = X_train.max(axis=0) + 1.0
    xs = np.linspace(xmin, xmax, 200)
    ys = np.linspace(ymin, ymax, 200)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    with torch.no_grad():
        grid_logits = model(torch.from_numpy(grid).to(device)).cpu().numpy().reshape(-1)
    grid_prob = 1 / (1 + np.exp(-grid_logits))
    zz = grid_prob.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, zz, levels=20)
    plt.scatter(X_train[y_train==0][:,0], X_train[y_train==0][:,1], s=1, alpha=0.3, label="majority")
    plt.scatter(X_train[y_train==1][:,0], X_train[y_train==1][:,1], s=6, alpha=0.7, label="minority")
    plt.title(f"Decision field — {activation_name}")
    plt.legend()
    plt.show()

    # Loss & grad-norm curves
    plt.figure()
    plt.plot(losses)
    plt.title(f"Training loss (per step) — {activation_name}")
    plt.xlabel("step")
    plt.ylabel("BCE loss")
    plt.show()

    plt.figure()
    plt.plot(grad_norms)
    plt.title(f"Gradient L2 norm (per step) — {activation_name}")
    plt.xlabel("step")
    plt.ylabel("||grad||")
    plt.show()

    # Latent manifold PCA (color by predicted probability on validation set)
    feats = capture_hidden_2nd_activation_features(model, X_val.astype(np.float32))
    proj = pca2d(feats)
    with torch.no_grad():
        val_logits = model(torch.from_numpy(X_val.astype(np.float32)).to(device)).cpu().numpy().reshape(-1)
    val_prob = 1 / (1 + np.exp(-val_logits))
    plt.figure()
    plt.scatter(proj[:,0], proj[:,1], s=6, c=val_prob, alpha=0.8)
    plt.title(f"Latent (2nd act) PCA — {activation_name} — colored by P(y=1)")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.show()

    return {
        "activation": activation_name,
        "AUC": auc,
        "AP": ap,
        "Recall@P>=0.8": r80,
        "loss_volatility": float(np.std(np.diff(losses))),
        "grad_volatility": float(np.std(np.diff(grad_norms))),
        "model": model,  # return for downstream PCA if needed
    }

# ---------- Experiment B: Descent smoothness (regression) ----------
def make_regression_dataset(n=4000):
    X = np.random.uniform(-2.5, 2.5, size=(n, 2)).astype(np.float32)
    y = (np.sin(3*X[:,0]) * np.cos(2*X[:,1]) + 0.2*(X[:,0]**2) - 0.1*(X[:,1]**2)).astype(np.float32)
    y += np.random.normal(0, 0.05, size=y.shape).astype(np.float32)
    return X, y

def train_regressor(X_train, y_train, X_val, y_val, activation_name: str, activation: nn.Module,
                    epochs=60, batch_size=256, lr=1e-3):
    model = make_mlp(2, 64, 1, activation).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    dl = data.DataLoader(data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).view(-1,1)),
                         batch_size=batch_size, shuffle=True)

    losses = []
    grad_norms = []
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            gnorm = grad_l2_norm(model)
            opt.step()
            losses.append(loss.item())
            grad_norms.append(gnorm)

    with torch.no_grad():
        val_pred = model(torch.from_numpy(X_val).to(device)).cpu().numpy().reshape(-1)
    mse = float(np.mean((val_pred - y_val)**2))

    plt.figure()
    plt.plot(losses)
    plt.title(f"Regression loss (per step) — {activation_name}")
    plt.xlabel("step")
    plt.ylabel("MSE")
    plt.show()

    plt.figure()
    plt.plot(grad_norms)
    plt.title(f"Gradient L2 norm (per step) — {activation_name}")
    plt.xlabel("step")
    plt.ylabel("||grad||")
    plt.show()

    return {
        "activation": activation_name,
        "Val_MSE": mse,
        "loss_volatility": float(np.std(np.diff(losses))),
        "grad_volatility": float(np.std(np.diff(grad_norms))),
        "model": model,
    }

# ---------- Experiment C: Difficult manifold approximation (2D field) ----------
def field_fn(X):
    x = X[:,0]
    y = X[:,1]
    return (np.sin(3*x) * np.cos(3*y) * np.exp(-0.2*(x**2 + y**2)) + 0.3*np.tanh(2*x - y)).astype(np.float32)

def make_field_samples(n=6000, lim=2.8):
    X = np.random.uniform(-lim, lim, size=(n,2)).astype(np.float32)
    y = field_fn(X)
    return X, y

def train_field_approximator(X_train, y_train, X_val, y_val, activation_name: str, activation: nn.Module,
                             epochs=80, batch_size=256, lr=1e-3):
    model = make_mlp(2, 64, 1, activation).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    dl = data.DataLoader(data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).view(-1,1)),
                         batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

    with torch.no_grad():
        val_pred = model(torch.from_numpy(X_val).to(device)).cpu().numpy().reshape(-1)
    mse = float(np.mean((val_pred - y_val)**2))

    # Heatmaps & residuals
    xs = np.linspace(-2.5, 2.5, 200)
    ys = np.linspace(-2.5, 2.5, 200)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    gt = field_fn(grid).reshape(xx.shape)
    with torch.no_grad():
        pred = model(torch.from_numpy(grid).to(device)).cpu().numpy().reshape(xx.shape)
    residual = (pred - gt)

    plt.figure()
    plt.imshow(gt, extent=[xs.min(), xs.max(), ys.min(), ys.max()], origin="lower", aspect="auto")
    plt.colorbar()
    plt.title("Ground truth field")
    plt.xlabel("x"); plt.ylabel("y")
    plt.show()

    plt.figure()
    plt.imshow(pred, extent=[xs.min(), xs.max(), ys.min(), ys.max()], origin="lower", aspect="auto")
    plt.colorbar()
    plt.title(f"Predicted field — {activation_name}")
    plt.xlabel("x"); plt.ylabel("y")
    plt.show()

    plt.figure()
    plt.imshow(residual, extent=[xs.min(), xs.max(), ys.min(), ys.max()], origin="lower", aspect="auto")
    plt.colorbar()
    plt.title(f"Residuals (pred - gt) — {activation_name}")
    plt.xlabel("x"); plt.ylabel("y")
    plt.show()

    # 1D slice (y=0)
    xline = np.linspace(-2.5, 2.5, 400).astype(np.float32)
    grid_line = np.stack([xline, np.zeros_like(xline)], axis=1)
    gt_line = field_fn(grid_line)
    with torch.no_grad():
        pred_line = model(torch.from_numpy(grid_line).to(device)).cpu().numpy().reshape(-1)

    plt.figure()
    plt.plot(xline, gt_line, label="ground truth")
    plt.plot(xline, pred_line, label=f"pred — {activation_name}")
    plt.title(f"1D slice at y=0 — {activation_name}")
    plt.xlabel("x"); plt.ylabel("f(x,0)")
    plt.legend()
    plt.show()

    # Latent manifold PCA on validation inputs, colored by ground-truth field value
    feats = capture_hidden_2nd_activation_features(model, X_val.astype(np.float32))
    proj = pca2d(feats)
    plt.figure()
    plt.scatter(proj[:,0], proj[:,1], s=6, c=y_val, alpha=0.8)
    plt.title(f"Latent (2nd act) PCA — {activation_name} — colored by f(x,y)")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.show()

    return {"activation": activation_name, "Val_MSE": mse, "model": model}

# ---------- Run all experiments ----------
activations: Dict[str, nn.Module] = {
    "GELU": nn.GELU(),
    "Softplus+1": ZLQGate(),
    "ReLU": nn.ReLU(),
    "SReLU∞(eps=0.5,k=1.5)": SReLUInf(eps=0.5, k=1.5),
}

# A) Tail-end classification
Xa, ya = make_imbalanced_dataset()
split = int(0.8 * len(Xa))
Xa_train, ya_train = Xa[:split], ya[:split]
Xa_val, ya_val = Xa[split:], ya[split:]

results_A = []
for name, act in activations.items():
    results_A.append(train_classifier(Xa_train, ya_train, Xa_val, ya_val, name, act))

# B) Descent smoothness (regression)
Xb, yb = make_regression_dataset()
split = int(0.8 * len(Xb))
Xb_train, yb_train = Xb[:split], yb[:split]
Xb_val, yb_val = Xb[split:], yb[split:]

results_B = []
for name, act in activations.items():
    results_B.append(train_regressor(Xb_train, yb_train, Xb_val, yb_val, name, act))

# C) Difficult manifold approximation
Xc, yc = make_field_samples()
split = int(0.8 * len(Xc))
Xc_train, yc_train = Xc[:split], yc[:split]
Xc_val, yc_val = Xc[split:], yc[split:]

results_C = []
for name, act in activations.items():
    results_C.append(train_field_approximator(Xc_train, yc_train, Xc_val, yc_val, name, act))

# Collate & print metrics
rows = []
for r in results_A:
    d = r.copy(); d.pop("model", None)
    rows.append({"Experiment": "A_tail_classification", **d})
for r in results_B:
    d = r.copy(); d.pop("model", None)
    rows.append({"Experiment": "B_descent_smoothness", **d})
for r in results_C:
    d = r.copy(); d.pop("model", None)
    rows.append({"Experiment": "C_manifold_approx", **d})

metrics_df = pd.DataFrame(rows)
print("\n=== Metrics Summary ===")
print(metrics_df)

# Save to CSV
metrics_df.to_csv("mlp_activation_comparison_metrics.csv", index=False)
print("\nSaved metrics to ./mlp_activation_comparison_metrics.csv")


'''
