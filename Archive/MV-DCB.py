# MV-DCB: Multi-Vector, Depth-Carried Backprop (standalone demo)
# - End uses standard CE gradient on logits
# - Each hidden layer receives a blended signal:
#     s = α * Proj_{span{J^T u_k}}(r_y) + β * (I - Proj)(r_y)
#   where {u_k} are a few logit-space anchors per sample and J is ∂logits/∂y_i.
# - No EMA. No hooks. Pure per-step geometry.
# - Quick demo on imbalanced 5-class 2D data; Adam+GELU for all paths.

import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0); np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------- data (same shape as before) --------
def make_imbalanced(n=24000, seed=1):
    rng = np.random.RandomState(seed); K=5
    means = np.array([[0,0],[2.6,2],[-2.6,2],[2.6,-2],[4.8,0.1]], np.float32)
    covs  = np.array([[[0.7,0],[0,0.7]], [[0.6,0.2],[0.2,0.9]], [[0.8,-0.2],[-0.2,0.6]],
                      [[0.6,0],[0,0.6]], [[0.9,0],[0,0.5]]], np.float32)
    probs = np.array([0.35,0.26,0.22,0.16,0.01], np.float32)
    counts = (probs*n).astype(int); counts[-1]=n-counts[:-1].sum()
    Xs, ys = [], []
    for k,c in enumerate(counts):
        Xk = rng.multivariate_normal(means[k], covs[k], size=c).astype(np.float32)
        yk = np.full(c, k, np.int64); Xs.append(Xk); ys.append(yk)
    X = np.vstack(Xs); y = np.concatenate(ys); idx = rng.permutation(n)
    return X[idx], y[idx], 5

# -------- model & activations --------
def gelu_exact(x):
    return 0.5*x*(1.0 + torch.erf(x / math.sqrt(2.0)))
def d_gelu_exact(x):
    return 0.5*(1.0 + torch.erf(x / math.sqrt(2.0))) + (x*torch.exp(-0.5*x*x))/math.sqrt(2*math.pi)

class MLP(nn.Module):
    def __init__(self, in_dim=2, width=128, depth=3, out_dim=5):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(in_dim if i==0 else width, width) for i in range(depth)])
        self.acts = ['gelu']*depth
        self.final = nn.Linear(width, out_dim)
    def forward(self, x):
        a = x
        for L in self.linears:
            a = gelu_exact(L(a))
        return self.final(a)

def act_and_deriv(name, y):
    if name=='gelu':
        return gelu_exact(y), d_gelu_exact(y)
    elif name=='relu':
        return F.relu(y), (y>0).to(y.dtype)
    else:
        raise ValueError

# -------- forward cache & tail maps --------
@torch.no_grad()
def forward_cache(model, x):
    ys, as_ = [], []
    a = x
    for L, name in zip(model.linears, ['gelu']*len(model.linears)):
        y = L(a); ys.append(y.clone())
        a,_ = act_and_deriv(name, y); as_.append(a.clone())
    logits = model.final(a)
    return ys, as_, logits

def make_tail(model, i):
    blocks = []
    blocks.append(lambda t: act_and_deriv('gelu', t)[0])
    for j in range(i+1, len(model.linears)):
        L = model.linears[j]
        blocks.append(lambda t, L=L: L(t))
        blocks.append(lambda t: act_and_deriv('gelu', t)[0])
    blocks.append(lambda t: model.final(t))
    def f_y(yin):
        t = yin
        for f in blocks: t = f(t)
        return t
    return f_y

# -------- anchors in logit space & projection in layer space --------
def build_anchors(p, y, K=3):
    """
    Per sample anchors u_k in logit space.
    Always include CE direction (p - onehot(y)), plus:
      - top non-target class (e_j - p)
      - random class (e_r - p) if K>=3
    Returns U: (B, K, C)
    """
    B, C = p.shape
    one = F.one_hot(y, C).to(p.dtype)                              # (B,C)
    u0 = (p - one)                                                 # CE dir
    # top non-target
    p_mask = p.clone()
    p_mask.scatter_(1, y.view(-1,1), -1e9)
    j = p_mask.argmax(dim=1)
    ej = F.one_hot(j, C).to(p.dtype)
    u1 = ej - p
    U = [u0, u1]
    if K >= 3:
        r = torch.randint(0, C, (B,), device=p.device)
        er = F.one_hot(r, C).to(p.dtype)
        U.append(er - p)
    U = torch.stack(U[:K], dim=1)                                  # (B,K,C)
    # zero-mean gauge per row (optional but harmless)
    U = U - U.mean(dim=2, keepdim=True)
    return U

def project_onto_span(D, r, lam=1e-6):
    """
    D: (B, K, H)  basis vectors per sample (rows are anchors mapped by J^T)
    r: (B, H)     vector to project
    returns proj(r) in span(D)
    """
    B, K, H = D.shape
    Dt = D.transpose(1,2)                  # (B, H, K)
    G  = torch.bmm(D, Dt)                  # (B, K, K)
    rhs = torch.bmm(D, r.unsqueeze(2))     # (B, K, 1)
    I = torch.eye(K, device=D.device, dtype=D.dtype).unsqueeze(0).expand(B, K, K)
    c = torch.linalg.solve(G + lam*I, rhs) # (B, K, 1)
    proj = torch.bmm(Dt, c).squeeze(2)     # (B, H)
    return proj

# -------- MV-DCB training step (manual backprop) --------
def trainstep_mv(model, opt, xb, yb, alpha=1.0, beta=0.3, proj_k=3, lam_proj=1e-6):
    """
    alpha: weight on projected (downstream-aware) component
    beta : weight on residual (orthogonal) component
    """
    model.train()
    opt.zero_grad(set_to_none=True)

    # forward cache
    ys, as_, logits = forward_cache(model, xb)
    p = torch.softmax(logits, dim=1)
    loss = F.cross_entropy(logits, yb)

    # top grads (standard CE)
    gL = (p - F.one_hot(yb, p.size(1)).to(p.dtype))     # (B,C)
    aLm1 = as_[-1]
    gW_last = gL.t() @ aLm1
    gb_last = gL.sum(dim=0)
    # upstream to y_{L-1}
    r_a = gL @ model.final.weight                       # (B,H)
    _, dphi = act_and_deriv('gelu', ys[-1])
    r_y = r_a * dphi                                    # (B,H)

    grads_W = [None]*len(model.linears) + [gW_last]
    grads_b = [None]*len(model.linears) + [gb_last]

    # hidden layers i = L-1..0
    for i in reversed(range(len(model.linears))):
        L = model.linears[i]
        a_im1 = xb if i==0 else as_[i-1]
        # build per-sample anchor subspace via VJPs
        y_req = ys[i].detach().requires_grad_(True)
        tail  = make_tail(model, i)
        logits_graph = tail(y_req)
        with torch.no_grad():
            U = build_anchors(torch.softmax(logits_graph, dim=1), yb, K=proj_k)  # (B,K,C)
        # Map anchors: d_k = J^T u_k
        Dk = []
        for k in range(U.size(1)):
            uk = U[:,k,:]
            dk = torch.autograd.grad(logits_graph, y_req, grad_outputs=uk,
                                     retain_graph=True)[0]                        # (B,H_i)
            Dk.append(dk)
        D = torch.stack(Dk, dim=1)                                               # (B,K,H_i)

        # Blend projected + residual
        proj = project_onto_span(D, r_y, lam=lam_proj)                            # (B,H_i)
        s_i  = alpha*proj + beta*(r_y - proj)                                     # (B,H_i)

        # parameter grads
        grads_W[i] = s_i.t() @ a_im1
        grads_b[i] = s_i.sum(dim=0)

        # upstream for next layer
        if i > 0:
            r_a = s_i @ L.weight                                                 # (B, in_i)
            _, dphi_prev = act_and_deriv('gelu', ys[i-1])
            r_y = r_a * dphi_prev

    # write & step
    for i,L in enumerate(model.linears):
        L.weight.grad = grads_W[i].to(L.weight.dtype)
        L.bias.grad   = grads_b[i].to(L.bias.dtype)
    model.final.weight.grad = grads_W[-1].to(model.final.weight.dtype)
    model.final.bias.grad   = grads_b[-1].to(model.final.bias.dtype)

    opt.step()
    return float(loss.item())

# -------- Baseline step --------
def trainstep_baseline(model, opt, xb, yb):
    opt.zero_grad(set_to_none=True)
    logits = model(xb)
    loss = F.cross_entropy(logits, yb)
    loss.backward()
    opt.step()
    return float(loss.item())

# -------- run small demo --------
X, y, K = make_imbalanced(24000, seed=1)
n_tr = 18000
Xtr, ytr = torch.from_numpy(X[:n_tr]).to(device), torch.from_numpy(y[:n_tr]).to(device)
Xva, yva = torch.from_numpy(X[n_tr:]).to(device), torch.from_numpy(y[n_tr:]).to(device)

def run(variant="baseline", epochs=10, batch=512, width=128, depth=3, lr=2e-3):
    model = MLP(2, width, depth, K).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch, shuffle=True)
    for _ in range(epochs):
        for xb, yb in dl:
            if variant == "baseline":
                trainstep_baseline(model, opt, xb, yb)
            else:
                trainstep_mv(model, opt, xb, yb, alpha=1.0, beta=0.3, proj_k=3, lam_proj=1e-6)
    with torch.no_grad():
        logits = model(Xva)
        acc = (logits.argmax(dim=1) == yva).float().mean().item()
    return acc, model

acc_b, m_b = run("baseline", epochs=8)
acc_m, m_m = run("mv",       epochs=8)
print(f"Baseline acc: {acc_b:.4f}    MV-DCB acc: {acc_m:.4f}")



# ================== MV-DCB vs Baseline on Rich Problems (Adam+GELU) ==================
# - Tasks:
#   A) Tail-imbalanced 5-class Gaussians (tail & calibration focus)
#   B) Checkerboard 4-class (geometry/decision map)
#   C) Periodic wells 5-class (quantile-binned; geometry/decision map)
#
# - Variants:
#   * baseline: standard autograd CE
#   * mv: MV-DCB (multi-vector, depth-carried backprop)
#
# - Outputs:
#   * printed metrics: acc, macro-F1, per-class P/R/F1, tail (rare) class P/R/F1, ECE
#   * plots: confusion matrices, reliability curves (ECE), 2D decision maps for B & C
#
# No EMA. Adam+GELU for all paths.

import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

torch.manual_seed(0); np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------- Utilities: metrics -----------------------------
def confusion_matrix(preds, targets, K):
    cm = np.zeros((K, K), dtype=np.int64)
    for p, t in zip(preds, targets):
        cm[int(t), int(p)] += 1
    return cm

def precision_recall_f1_per_class(cm):
    # cm[y_true, y_pred]
    K = cm.shape[0]
    P = np.zeros(K); R = np.zeros(K); F1 = np.zeros(K)
    for k in range(K):
        tp = cm[k,k]
        fp = cm[:,k].sum() - tp
        fn = cm[k,:].sum() - tp
        P[k] = tp / (tp+fp+1e-12)
        R[k] = tp / (tp+fn+1e-12)
        F1[k] = 2*P[k]*R[k]/(P[k]+R[k]+1e-12)
    return P, R, F1

def macro_f1(cm):
    _, _, F1 = precision_recall_f1_per_class(cm)
    return float(F1.mean())

def expected_calibration_error(probs, labels, bins=15):
    # probs: (N, K) softmax; labels: (N,)
    N, K = probs.shape
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    accs = (pred == labels).astype(np.float64)
    ece = 0.0
    bin_edges = np.linspace(0,1,bins+1)
    for i in range(bins):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (conf >= lo) & (conf < hi) if i < bins-1 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0: 
            continue
        ece += np.abs(accs[mask].mean() - conf[mask].mean()) * (mask.sum()/N)
    return float(ece)

def print_metrics(name, probs, labels, tail_class=None):
    pred = probs.argmax(axis=1)
    K = probs.shape[1]
    cm = confusion_matrix(pred, labels, K)
    mf1 = macro_f1(cm)
    acc = float((pred == labels).mean())
    P, R, F1 = precision_recall_f1_per_class(cm)
    ece = expected_calibration_error(probs, labels, bins=15)

    print(f"{name:10s}: acc={acc:.4f}  macroF1={mf1:.4f}  ECE={ece:.3f}")
    if tail_class is not None:
        k = tail_class
        print(f"  tail(c{k}) P={P[k]:.3f} R={R[k]:.3f} F1={F1[k]:.3f}")
    return {"acc":acc, "macroF1":mf1, "ECE":ece, "cm":cm, "P":P, "R":R, "F1":F1}

# ----------------------------- Data: three tasks -----------------------------
def make_tail_imbalanced(n=24000, seed=1):
    rng = np.random.RandomState(seed); K=5
    means = np.array([[0,0],[2.6,2],[-2.6,2],[2.6,-2],[4.8,0.1]], np.float32)
    covs  = np.array([[[0.7,0],[0,0.7]], [[0.6,0.2],[0.2,0.9]], [[0.8,-0.2],[-0.2,0.6]],
                      [[0.6,0],[0,0.6]], [[0.9,0],[0,0.5]]], np.float32)
    probs = np.array([0.35,0.26,0.22,0.16,0.01], np.float32)  # tail = class 4
    counts = (probs*n).astype(int); counts[-1] = n - counts[:-1].sum()
    Xs, ys = [], []
    for k,c in enumerate(counts):
        Xk = rng.multivariate_normal(means[k], covs[k], size=c).astype(np.float32)
        yk = np.full(c, k, np.int64); Xs.append(Xk); ys.append(yk)
    X = np.vstack(Xs); y = np.concatenate(ys); idx = rng.permutation(n)
    return X[idx], y[idx], 5, 4  # K=5, tail_class=4

def make_checkerboard(n=24000, lim=3.0, cell=1.0, seed=2):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-lim, lim, size=(n,2)).astype(np.float32)
    gx = np.floor((X[:,0] + lim) / cell).astype(int)
    gy = np.floor((X[:,1] + lim) / cell).astype(int)
    # two bits -> 4 classes
    labels = ( (gx & 1) << 1 ) | (gy & 1)
    return X, labels.astype(np.int64), 4, None

def field_fn(X):
    x = X[:,0]; y = X[:,1]
    return (np.sin(3*x)*np.cos(3*y)*np.exp(-0.2*(x**2+y**2)) + 0.3*np.tanh(2*x - y)).astype(np.float32)

def make_periodic_wells_binned(n=24000, lim=2.8, K=5, seed=3):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-lim, lim, size=(n,2)).astype(np.float32)
    z = field_fn(X)
    # quantile bins -> K classes
    qs = np.quantile(z, np.linspace(0,1,K+1))
    # ensure unique bins
    eps = 1e-6
    qs = np.maximum.accumulate(qs + np.arange(len(qs))*eps)
    y = np.clip(np.searchsorted(qs, z, side='right')-1, 0, K-1)
    return X, y.astype(np.int64), K, None

# ----------------------------- Model & activations -----------------------------
def gelu_exact(x):
    return 0.5*x*(1.0 + torch.erf(x / math.sqrt(2.0)))
def d_gelu_exact(x):
    return 0.5*(1.0 + torch.erf(x/ math.sqrt(2.0))) + (x*torch.exp(-0.5*x*x))/math.sqrt(2*math.pi)

def act_and_deriv(y):  # GELU only (per request)
    return gelu_exact(y), d_gelu_exact(y)

class MLP(nn.Module):
    def __init__(self, in_dim=2, width=128, depth=3, out_dim=5):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(in_dim if i==0 else width, width) for i in range(depth)])
        self.final = nn.Linear(width, out_dim)
    def forward(self, x):
        a = x
        for L in self.linears:
            a = gelu_exact(L(a))
        return self.final(a)

# ----------------------------- Forward caches & tails -----------------------------
@torch.no_grad()
def forward_cache(model, x):
    ys, as_ = [], []
    a = x
    for L in model.linears:
        y = L(a); ys.append(y.clone())
        a,_ = act_and_deriv(y); as_.append(a.clone())
    logits = model.final(a)
    return ys, as_, logits

def make_tail(model, i):
    blocks = []
    blocks.append(lambda t: act_and_deriv(t)[0])        # act at layer i
    for j in range(i+1, len(model.linears)):
        L = model.linears[j]
        blocks.append(lambda t, L=L: L(t))
        blocks.append(lambda t: act_and_deriv(t)[0])
    blocks.append(lambda t: model.final(t))
    def f_y(yin):
        t = yin
        for f in blocks: t = f(t)
        return t
    return f_y

# ----------------------------- MV-DCB core -----------------------------
def build_anchors(p, y, K=3):
    # CE direction + top competing + random
    B, C = p.shape
    one = F.one_hot(y, C).to(p.dtype)
    u0 = (p - one)
    pm = p.clone()
    pm.scatter_(1, y.view(-1,1), -1e9)
    j = pm.argmax(dim=1)
    ej = F.one_hot(j, C).to(p.dtype)
    u1 = ej - p
    U = [u0, u1]
    if K >= 3:
        r = torch.randint(0, C, (B,), device=p.device)
        er = F.one_hot(r, C).to(p.dtype)
        U.append(er - p)
    U = torch.stack(U[:K], dim=1)
    U = U - U.mean(dim=2, keepdim=True)
    return U

def project_onto_span(D, r, lam=0.0):
    """
    Robust projection of r onto span(D^T) using QR.
    D: (B, K, H)   r: (B, H)
    Returns: proj (B, H)
    """
    # Orthonormal basis Q for the columns of D^T via QR
    # Dt: (B, H, K)  -> Q: (B, H, K) with Q^T Q = I_K
    Dt = D.transpose(1, 2)
    Q, _ = torch.linalg.qr(Dt, mode='reduced')
    proj = (Q @ (Q.transpose(1, 2) @ r.unsqueeze(-1))).squeeze(-1)
    return proj

def trainstep_baseline(model, opt, xb, yb):
    opt.zero_grad(set_to_none=True)
    logits = model(xb)
    loss = F.cross_entropy(logits, yb)
    loss.backward()
    opt.step()
    return float(loss.item())

def trainstep_mv(model, opt, xb, yb, alpha=1.0, beta=0.3, proj_k=3, lam_proj=1e-6):
    model.train(); opt.zero_grad(set_to_none=True)

    ys, as_, logits = forward_cache(model, xb)
    p = torch.softmax(logits, dim=1)
    loss = F.cross_entropy(logits, yb)

    # top grads
    gL   = (p - F.one_hot(yb, p.size(1)).to(p.dtype))     # (B,C)
    aLm1 = as_[-1]
    gW_L = gL.t() @ aLm1
    gB_L = gL.sum(dim=0)

    # upstream to y_{L-1}
    r_a = gL @ model.final.weight
    _, dphi = act_and_deriv(ys[-1])
    r_y = r_a * dphi

    grads_W = [None]*len(model.linears) + [gW_L]
    grads_B = [None]*len(model.linears) + [gB_L]

    # hidden layers
    for i in reversed(range(len(model.linears))):
        L = model.linears[i]
        a_im1 = xb if i==0 else as_[i-1]

        y_req = ys[i].detach().requires_grad_(True)
        tail  = make_tail(model, i)
        logits_graph = tail(y_req)
        with torch.no_grad():
            U = build_anchors(torch.softmax(logits_graph, dim=1), yb, K=proj_k)  # (B,K,C)

        # D_k = J^T u_k
        Dk = []
        for k in range(U.size(1)):
            uk = U[:,k,:]
            dk = torch.autograd.grad(logits_graph, y_req, grad_outputs=uk,
                                     retain_graph=True)[0]          # (B,H_i)
            Dk.append(dk)
        D = torch.stack(Dk, dim=1)                                  # (B,K,H_i)

        proj = project_onto_span(D, r_y, lam=lam_proj)
        s_i  = alpha*proj + beta*(r_y - proj)

        grads_W[i] = s_i.t() @ a_im1
        grads_B[i] = s_i.sum(dim=0)

        if i>0:
            r_a = s_i @ L.weight
            _, dphi_prev = act_and_deriv(ys[i-1])
            r_y = r_a * dphi_prev

    # write & step
    for i,L in enumerate(model.linears):
        L.weight.grad = grads_W[i].to(L.weight.dtype)
        L.bias.grad   = grads_B[i].to(L.bias.dtype)
    model.final.weight.grad = grads_W[-1].to(model.final.weight.dtype)
    model.final.bias.grad   = grads_B[-1].to(model.final.bias.dtype)

    opt.step()
    return float(loss.item())

# ----------------------------- Training & evaluation harness -----------------------------
def train_eval_task(task_name, make_data_fn, width=128, depth=3, epochs=8, batch=512, lr=2e-3,
                    mv_alpha=1.0, mv_beta=0.3, mv_k=3, mv_lam=1e-6, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    X, y, K, tail_cls = make_data_fn()
    n_tr = int(0.75*len(X))
    Xtr, ytr = torch.from_numpy(X[:n_tr]).to(device), torch.from_numpy(y[:n_tr]).to(device)
    Xva, yva = torch.from_numpy(X[n_tr:]).to(device), torch.from_numpy(y[n_tr:]).to(device)

    def run_variant(kind):
        model = MLP(2, width, depth, K).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch, shuffle=True)
        for _ in range(epochs):
            for xb, yb in dl:
                if kind=="baseline":
                    trainstep_baseline(model, opt, xb, yb)
                else:
                    trainstep_mv(model, opt, xb, yb, alpha=mv_alpha, beta=mv_beta,
                                 proj_k=mv_k, lam_proj=mv_lam)
        with torch.no_grad():
            logits = model(Xva)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
        return model, probs

    m_b, pb = run_variant("baseline")
    m_m, pm = run_variant("mv")

    print(f"\n=== {task_name} ===")
    Mb = print_metrics("Baseline", pb, y[n_tr:], tail_cls)
    Mm = print_metrics("MV-DCB ", pm, y[n_tr:], tail_cls)

    # confusion matrices
    fig, ax = plt.subplots(1,2, figsize=(10,4))
    cb = confusion_matrix(pb.argmax(axis=1), y[n_tr:], K)
    cm = confusion_matrix(pm.argmax(axis=1), y[n_tr:], K)
    ax[0].imshow(cb, cmap='Blues'); ax[0].set_title(f"{task_name} — Baseline CM")
    ax[1].imshow(cm, cmap='Blues'); ax[1].set_title(f"{task_name} — MV-DCB CM")
    for a in ax:
        a.set_xlabel("pred"); a.set_ylabel("true")
    plt.tight_layout(); plt.show()

    # reliability curves
    def plot_reliability(ax, probs, labels, title):
        pred = probs.argmax(axis=1)
        conf = probs.max(axis=1)
        bins = np.linspace(0,1,16)
        mids = 0.5*(bins[1:]+bins[:-1])
        accs = []; confs = []
        for i in range(len(bins)-1):
            lo, hi = bins[i], bins[i+1]
            m = (conf >= lo) & (conf < hi) if i < len(bins)-2 else (conf >= lo) & (conf <= hi)
            if m.sum()==0: 
                accs.append(np.nan); confs.append(np.nan); continue
            accs.append((pred[m]==labels[m]).mean())
            confs.append(conf[m].mean())
        accs = np.array(accs); confs = np.array(confs)
        ax.plot([0,1],[0,1],'k--',lw=1)
        ax.plot(np.nan_to_num(confs), np.nan_to_num(accs), marker='o')
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.set_title(title); ax.set_xlabel("confidence"); ax.set_ylabel("empirical acc")

    fig, ax = plt.subplots(1,2, figsize=(10,4))
    plot_reliability(ax[0], pb, y[n_tr:], f"{task_name} — Baseline")
    plot_reliability(ax[1], pm, y[n_tr:], f"{task_name} — MV-DCB")
    plt.tight_layout(); plt.show()

    # decision maps for 2D geometry tasks
    if task_name in ["Checkerboard (4c)","Periodic Wells (5c, binned)"]:
        xs = np.linspace(-3.0, 3.0, 300)
        ys = np.linspace(-3.0, 3.0, 300)
        xx, yy = np.meshgrid(xs, ys)
        grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
        with torch.no_grad():
            pb_map = torch.softmax(m_b(torch.from_numpy(grid).to(device)), dim=1).argmax(dim=1).cpu().numpy()
            pm_map = torch.softmax(m_m(torch.from_numpy(grid).to(device)), dim=1).argmax(dim=1).cpu().numpy()
        fig, ax = plt.subplots(1,2, figsize=(10,4))
        ax[0].imshow(pb_map.reshape(xx.shape), extent=[xs.min(), xs.max(), ys.min(), ys.max()],
                     origin='lower', aspect='auto', interpolation='nearest')
        ax[0].set_title(f"{task_name} — Baseline decision")
        ax[1].imshow(pm_map.reshape(xx.shape), extent=[xs.min(), xs.max(), ys.min(), ys.max()],
                     origin='lower', aspect='auto', interpolation='nearest')
        ax[1].set_title(f"{task_name} — MV-DCB decision")
        for a in ax: a.set_xlabel("x"); a.set_ylabel("y")
        plt.tight_layout(); plt.show()

    return {"baseline":Mb, "mv":Mm}

# ----------------------------- Run all tasks -----------------------------
if __name__ == "__main__":
    A = train_eval_task("Tail Imbalanced (5c)", make_tail_imbalanced,
                        epochs=8, width=128, depth=3, lr=2e-3,
                        mv_alpha=1.0, mv_beta=0.3, mv_k=3, mv_lam=1e-6)

    B = train_eval_task("Checkerboard (4c)", make_checkerboard,
                        epochs=8, width=128, depth=3, lr=2e-3,
                        mv_alpha=1.0, mv_beta=0.3, mv_k=3, mv_lam=1e-6)

    C = train_eval_task("Periodic Wells (5c, binned)", make_periodic_wells_binned,
                        epochs=8, width=128, depth=3, lr=2e-3,
                        mv_alpha=1.0, mv_beta=0.3, mv_k=3, mv_lam=1e-6)
