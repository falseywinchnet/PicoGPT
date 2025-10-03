
# The Fast Fourier Transform cannot be (presently) Learned

A forensic and mathematical analysis of why backpropagation fails to discover the radix‑2 RFFT factorization from data is provided showing a useful problem for the advancement of current optimizer and backpropagation algorithmic designs, aided by the target factorization being known in closed form.

----------

## 0) Setup and exact object of study

We consider the 512‑point real FFT (RFFT), producing 257 complex outputs (DC through Nyquist). The butterfly network depth is:

-   depth = log2(512) = 9 butterfly levels,
    
-   of which 8 levels carry twiddle multipliers (the initial 2×2 split has none).
    

Parameters:

-   M: a 2×2 complex “butterfly” matrix.
    
-   Twiddles w[s,l]: complex scalars on stage s, reused across local butterflies (s = 1…8 here).
    

Connectivity at stage s:

-   e = 2^s wires (top half and bottom half paired),
    
-   q = 2^(8−s) local butterflies per twiddle,
    
-   a single twiddle at stage s influences exactly 2^(8−s) output frequency bins.
    

Target operator F:

-   the real‑to‑complex rfft; the reference implementation reproduces numpy’s rfft to about 1e−16.
    

Training objective (informal):

-   inputs x ∈ R^512 sampled i.i.d. N(0,1),
    
-   loss L(θ) = E[ || f_θ(x) − F x ||^2 ] (MSE on real and imaginary parts), with θ = (M, all twiddles).
    
Code:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np

class Radix2RFFT(nn.Module):
    def __init__(self):
        super().__init__()
        # Register as a parameter if you want to train them,
        # otherwise buffer is fine.
        self.M = nn.Parameter(torch.ones((2, 2), dtype=torch.complex128))
        # Trainable twiddle factors (511 total for 512-point RFFT)
        self.twiddle_factors = nn.Parameter(torch.ones(511, dtype=torch.complex128))


    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        input_data: real tensor (512,)
        returns: complex tensor (257,)
        """
        input_data = input_data.to(torch.float64)

        # Stage 0
        X_stage_0 = torch.zeros((2, 256), dtype=torch.complex128, device=input_data.device)
        for i in range(256):
            X_stage_0[0, i] = self.M[0, 0] * input_data[i]     + self.M[0, 1] * input_data[i + 256]
            X_stage_0[1, i] = self.M[1, 0] * input_data[i]     + self.M[1, 1] * input_data[i + 256]

        # Stage function
        def fft_stage(X_in, e, q, offset):
            tmp = torch.zeros((2*e, q), dtype=torch.complex128, device=X_in.device)
            for i in range(e):
                for j in range(q):
                    twiddle = self.twiddle_factors[offset + i]
                    product = twiddle * X_in[i, j + q]
                    tmp[i, j]     = X_in[i, j] + product
                    tmp[i + e, j] = X_in[i, j] - product
            return tmp

        # Apply stages (keeping running offset into twiddle_factors)
        offset = 0
        X_stage_1 = fft_stage(X_stage_0, 2,   128, offset); offset += 2
        X_stage_2 = fft_stage(X_stage_1, 4,   64,  offset); offset += 4
        X_stage_3 = fft_stage(X_stage_2, 8,   32,  offset); offset += 8
        X_stage_4 = fft_stage(X_stage_3, 16,  16,  offset); offset += 16
        X_stage_5 = fft_stage(X_stage_4, 32,  8,   offset); offset += 32
        X_stage_6 = fft_stage(X_stage_5, 64,  4,   offset); offset += 64
        X_stage_7 = fft_stage(X_stage_6, 128, 2,   offset); offset += 128
        X_stage_8 = fft_stage(X_stage_7, 256, 1,   offset); offset += 256

        # Nyquist handling
        X_stage_8[257, 0] = X_stage_8[257, 0].real + 0.0j

        # Output
        return X_stage_8[:257, 0]


# --- Config ---
SEED = 42
STEPS = 3000
BATCH_SIZE = 16
LR = 1e-2
PRINT_EVERY = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)

# --- Instantiate model ---
# The constructor's argument isn't used in your class; we pass a dummy to satisfy the signature.
model = Radix2RFFT()
model = model.to(DEVICE)

opt = optim.Adam(model.parameters(), lr=LR)

def numpy_rfft_target(x_1d_float64: np.ndarray) -> np.ndarray:
    """
    x_1d_float64: numpy array of shape (512,), dtype float64
    Returns numpy complex128 array of shape (257,) from np.fft.rfft.
    """
    return np.fft.rfft(x_1d_float64, n=512).astype(np.complex128)

def torchify_complex(np_c128: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Converts a numpy complex128 array to a torch.complex128 tensor on device.
    """
    # torch.from_numpy supports complex128; ensure dtype is exactly complex128
    return torch.from_numpy(np_c128).to(torch.complex128).to(device)

def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    pred, target: complex tensors of shape (batch, 257)
    Use MSE on real/imag parts.
    """
    return F.mse_loss(pred.real, target.real) + F.mse_loss(pred.imag, target.imag)

@torch.no_grad()
def eval_on_random_examples(n: int = 3):
    model.eval()
    errs = []
    for _ in range(n):
        x = np.random.randn(512).astype(np.float64)
        y = numpy_rfft_target(x)
        y_hat = model(torch.from_numpy(x).to(torch.float64).to(DEVICE))
        y_t = torchify_complex(y, DEVICE)
        errs.append(F.mse_loss(y_hat.real, y_t.real).item() + F.mse_loss(y_hat.imag, y_t.imag).item())
    print(f"[eval] mean MSE over {n} random examples: {np.mean(errs):.6e}")
    model.train()

# --- Training loop ---
model.train()
for step in range(1, STEPS + 1):
    opt.zero_grad()

    # Create a minibatch of random real signals
    # (model is single-sample, so we'll loop over batch)
    batch_preds = []
    batch_targets = []

    # Generate the entire batch first (numpy), then push to model
    xs_np = np.random.randn(BATCH_SIZE, 512).astype(np.float64)
    ys_np = np.stack([numpy_rfft_target(x) for x in xs_np], axis=0)  # (B, 257) complex128

    # Forward per sample (model expects shape (512,))
    for b in range(BATCH_SIZE):
        x_b = torch.from_numpy(xs_np[b]).to(torch.float64).to(DEVICE)
        y_hat_b = model(x_b)  # (257,) complex128
        batch_preds.append(y_hat_b)

    preds = torch.stack(batch_preds, dim=0)  # (B, 257) complex128
    targets = torchify_complex(ys_np, DEVICE)  # (B, 257) complex128

    loss = loss_fn(preds, targets)
    loss.backward()
    opt.step()

    if step % PRINT_EVERY == 0 or step == 1:
        print(f"step {step:5d} | loss {loss.item():.6e}")
        eval_on_random_examples(n=3)

# Final quick check
eval_on_random_examples(n=10)

```

----------

## 1) Algebraic structure: a deep multilinear map in the parameters

Each output y[k] is linear in x but multilinear in the parameters. Along any path from an input sample to an output bin, the signal is multiplied by:

-   one entry from M at the first split, then
    
-   one twiddle per subsequent stage (8 of them in total).
    

Therefore:

-   y[k] is a complex polynomial of degree 9 in the parameters (1 from M, 8 from twiddles),
    
-   the squared‑error loss L is a polynomial of total degree 18.
    

This yields a highly non‑convex landscape with deep, high‑order parameter couplings.

----------

## 2) Gauge symmetries ⇒ flat manifolds of global minima

Butterfly factorizations admit scaling “gauges”. If one multiplies the two wires entering a sub‑tree by a nonzero complex α and divides the matching wires downstream by α, the overall linear map is unchanged. In other words, many parameter settings produce the same operator.

Consequence:

-   the parameterization is not identifiable,
    
-   there exist high‑dimensional flat manifolds of exact minima,
    
-   the Hessian at a true solution has many zero eigenvalues (directions of indifference),
    
-   gradient methods encounter plateaus and poorly conditioned curvature even when very near the right answer.
    

This gets worse when M is learned rather than fixed to the canonical values.

----------

## 3) Depth‑induced multiplicative conditioning

Write w_t for a twiddle on a typical path. The gradient dL/dw_s contains products of the remaining twiddles on that path. If r = typical magnitude of |w_t| during training, then (ignoring phases) the scale of dL/dw_s behaves like r^8.

Consequence:

-   if r < 1 ⇒ gradients vanish roughly like r^8,
    
-   if r > 1 ⇒ gradients explode roughly like r^8,
    
-   only the unit‑circle manifold |w| = 1 is well‑conditioned, but ordinary optimizers do not enforce unit modulus.
    

This is the multiplicative analogue of vanishing/exploding gradients, with fixed product depth 8 in the twiddles (9 if you include the factor from M).

----------

## 4) Gradient‑signal allocation: why the problem is tail‑heavy

A parameter at stage s influences exactly 2^ (8−s) output bins. With i.i.d. Gaussian inputs and per‑bin MSE, residuals across bins are roughly uncorrelated with comparable variance, so the stochastic gradient for that parameter aggregates 2^(8−s) random contributions.

Signal scaling:

-   std of dL/dw_s ∝ sqrt( 2^(8−s) ).
    
-   Stage 1 parameters receive ~sqrt(128) ≈ 11.3× stronger raw gradient signal than stage 8 parameters.
    

Sample‑complexity implication:

-   to match gradient SNR between stage 1 and stage 8 at fixed batch size, the deepest stage needs about 128× more samples accumulated,
    
-   with batch size 16, last‑stage updates are noise‑dominated, so training stalls in the tail even if early stages make progress.
    

----------

## 5) Expected‑gradient symmetry barriers

At near‑symmetric initialization (e.g., all twiddles = 1 and M ≈ [[1,1],[1,−1]]), the forward map has no preferred global phase. Under isotropic Gaussian inputs the expected residual in any frequency bin has no preferred phase direction; many parameter directions have zero mean gradient. Adam then executes a small‑step random walk in phase space dominated by noise.

----------

## 6) Curvature near the true solution: skewed Hessian

Parametrize each twiddle as w = exp(i·θ). Linearizing the loss near the exact FFT gives a quadratic form in the phase offsets θ with coefficients proportional to the energy carried by the affected sub‑trees. Those coefficients decay geometrically with stage depth because deeper stages touch fewer outputs. Together with exact zero‑modes from gauges, the local condition number is enormous even at the optimum.

----------

## 7) Why learning the DFT matrix is easy but learning the FFT factorization is hard

Learning the full RFFT as a single linear map W (i.e., solve W x ≈ F x) is just linear least squares: convex, with unique solution W = F once you have enough diverse x. No products of parameters, no unit‑circle constraints, and almost no gauge freedom beyond trivial scalings.

By contrast, the FFT factorization imposes a nonlinear map W(θ) with deep multiplicative couplings. The same operator becomes a hard non‑convex identification problem with poor conditioning.

----------

## 8) Quantitative summary for the 512‑point RFFT

-   Depth and degree: 8 twiddle stages after the initial split; outputs are degree‑9 in parameters; loss is degree‑18.
    
-   Parameter count: hundreds of complex parameters (≈511 twiddles for the RFFT), tightly coupled across stages.
    
-   Gauges: many equivalent parameterizations ⇒ flat manifolds; singular Hessian at minima.
    
-   Conditioning: gradient magnitudes scale like r^8 with r = typical |w|; any deviation from |w|=1 produces strong vanish/explode.
    
-   Tail heaviness: per‑parameter gradient std scales like sqrt(2^(8−s)); deepest stage gets ~1/11 of the raw signal of the earliest stage; needs ~128× more data to match SNR.
    
-   Curvature spread: even at the solution, eigenvalues fall off geometrically with depth and include exact zeros, so local condition numbers are huge.
    

These make the backpropagation problem algorithmically hostile: slow, noisy, and structurally under‑determined at practical batch sizes and learning rates.

----------

## 9) Forensic notes on the provided modules

-   NumPy/Numba path: twiddles are fixed to the exact roots of unity, so the FFT is correct by construction and matches numpy to ~1e−16.
Code:
```py
import numba
import numpy as np

tw = [np.exp(-1.0 * 1.0j * np.pi * np.arange(((2**i)/2), dtype=np.complex256) / ((2**i)/2)) for i in range(1,11)]
list_of_lists = [list(map(lambda x: [x.astype(np.complex128)], arr)) for arr in tw]
twiddlefactors = np.concatenate(list_of_lists)
inverse = np.conj(twiddlefactors) #use this for irfft

@numba.jit(numba.complex128[:](numba.float64[:],numba.complex128[:,:]),fastmath=True,nopython=True)
def unrolled_numba_rfft(input_data:np.ndarray, twiddlefactors: np.ndarray):
    #input must be float64, exactly 512 elements. Outputs 257 elements, complex128.
    #this function has been constrained to only work on real data(float64), power of two arrays of size 512, and only returns 257 elements.
    #Cooley-Tukey radix-2 Decimation in Time unrolled loop, precomputed twiddlefactors



    ## Initialization Section: This part of the code sets up the matrix required for "Butterfly" operations in FFT (Fast Fourier Transform). 
    #It then initializes arrays for storing the result at each stage of the FFT calculation. 
    #Each array represents a stage in the FFT computation, with the dimensions set according to the 'radix-2' nature of the calculation. T
    #This means that at each stage, the data is divided in half.
    
  
    M = np.asarray([[1.+0.0000000e+00j, 1.+0.0000000e+00j],[ 1.+0.0000000e+00j, -1.-1.2246468e-16j]],dtype=numpy.complex128) #butterfly matrix

    X_stage_0 = np.zeros((2, 256), dtype=np.complex128)
    X_stage_1 = np.zeros((4, 128), dtype=np.complex128)
    X_stage_2 = np.zeros((8, 64), dtype=np.complex128)
    X_stage_3 = np.zeros((16, 32), dtype=np.complex128)
    X_stage_4 = np.zeros((32, 16), dtype=np.complex128)
    X_stage_5 = np.zeros((64, 8), dtype=np.complex128)
    X_stage_6 = np.zeros((128, 4), dtype=np.complex128)
    X_stage_7 = np.zeros((256, 2), dtype=np.complex128)
    X_stage_8 = np.zeros((512, 1), dtype=np.complex128)

    #Stage-0 FFT Calculation: In this section, the code computes the first stage of the FFT calculation. 
    #It performs "Butterfly" operations on the input data using the previously initialized matrix.
    #A "Butterfly" operation involves a simple pattern of multiplications and additions between pairs of input data points.
    
    X_stage_0[0, :] = M[0, 0] * input_data[:256]  + M[0, 1] * input_data[256:]
    X_stage_0[1, :] = M[1, 0] * input_data[:256]  + M[1, 1] * input_data[256:]
    
    # Multi-Stage FFT Calculations: This section of the code computes the rest of the stages in the FFT calculation. 
    #At each stage, the code computes the "Butterfly" operations, now involving the 'twiddle factors' -
    #complex roots of unity that are integral to the FFT algorithm. 
    #Each stage reduces the data's size by half and computes the FFT of these smaller chunks.




    e = 2 #twiddle_index
    q = 128 #quarter length
    X_stage_1[:e, :q] = X_stage_0[:e, :q] + twiddlefactors[(e-1):(2*e)-1] * X_stage_0[:e, q:2*q]
    X_stage_1[e:e*2, :q] = X_stage_0[:e, :q] - twiddlefactors[(e-1):(2*e)-1]  *  X_stage_0[:e, q:2*q]

    e = 4
    q = 64
    X_stage_2[:e, :q] = X_stage_1[:e, :q] + twiddlefactors[(e-1):(2*e)-1] * X_stage_1[:e, q:2*q]
    X_stage_2[e:e*2, :q] = X_stage_1[:e, :q] - twiddlefactors[(e-1):(2*e)-1]  *  X_stage_1[:e, q:2*q]

    e = 8
    q = 32
    X_stage_3[:e, :q] = X_stage_2[:e, :q] + twiddlefactors[(e-1):(2*e)-1] * X_stage_2[:e, q:2*q]
    X_stage_3[e:e*2, :q] = X_stage_2[:e, :q] - twiddlefactors[(e-1):(2*e)-1]  *  X_stage_2[:e, q:2*q]
    
    e = 16
    q = 16
    X_stage_4[:e, :q] = X_stage_3[:e, :q] + twiddlefactors[(e-1):(2*e)-1] * X_stage_3[:e, q:2*q]
    X_stage_4[e:e*2, :q] = X_stage_3[:e, :q] - twiddlefactors[(e-1):(2*e)-1]  *  X_stage_3[:e, q:2*q]

    e = 32
    q = 8
    X_stage_5[:e, :q] = X_stage_4[:e, :q] + twiddlefactors[(e-1):(2*e)-1] * X_stage_4[:e, q:2*q]
    X_stage_5[e:e*2, :q] = X_stage_4[:e, :q] - twiddlefactors[(e-1):(2*e)-1]  *  X_stage_4[:e, q:2*q]

    e = 64
    q = 4
    X_stage_6[:e, :q] = X_stage_5[:e, :q] + twiddlefactors[(e-1):(2*e)-1] * X_stage_5[:e, q:2*q]
    X_stage_6[e:e*2, :q] = X_stage_5[:e, :q] - twiddlefactors[(e-1):(2*e)-1]  *  X_stage_5[:e, q:2*q]

    e = 128
    q = 2
    X_stage_7[:e, :q] = X_stage_6[:e, :q] + twiddlefactors[(e-1):(2*e)-1] * X_stage_6[:e, q:2*q]
    X_stage_7[e:e*2, :q] = X_stage_6[:e, :q] - twiddlefactors[(e-1):(2*e)-1]  *  X_stage_6[:e, q:2*q]

    e = 256
    q = 1
    X_stage_8[:e, :q] = X_stage_7[:e, :q] + twiddlefactors[(e-1):(2*e)-1] * X_stage_7[:e, q:2*q]
    X_stage_8[e:e*2, :q] = X_stage_7[:e, :q] - twiddlefactors[(e-1):(2*e)-1]  *  X_stage_7[:e, q:2*q]
    X_stage_8[512//2 + 1,0] = X_stage_8[512//2 + 1,0].real + 0.0j#nyquist handling

    return  X_stage_8[:512//2 + 1,0]#return only the first half- the real complex
```
    
-   PyTorch path: twiddles are free complex parameters initialized at 1; the model must discover unit‑modulus phases through depth‑8 products. This triggers the conditioning and tail‑SNR issues above.
    
-   Twiddle count: the forward RFFT as written uses 511 twiddles; alternative generators (e.g., a full complex FFT including both directions) can produce longer tables (e.g., ~1023). The exact count is not the blocker; the multiplicative structure is.
    
-   Data and loss: i.i.d. Gaussian inputs with per‑bin MSE maximize the symmetry‑barriers and tail‑heaviness (no curriculum or localized supervision to break symmetry).
    

----------

## 10) Thesis

“The Fast Fourier Transform cannot be (presently) learned” — by naïve gradient‑based methods over unconstrained complex parameters — because the factorized representation turns a simple linear regression problem into a deep, high‑degree, gauge‑symmetric multiplicative optimization with:

1.  exponentially skewed gradient allocation across stages (tail heaviness),
    
2.  vanishing/exploding sensitivity unless parameters stay on the unit circle,
    
3.  large flat manifolds of equivalent optima (non‑identifiability), and
    
4.  a Hessian spectrum that is both singular and exponentially ill‑conditioned even at the solution.
    

These obstacles are structural, not incidental. They persist regardless of optimizer choice (SGD/Adam/L‑BFGS), batch size (unless impractically large), or minor architectural tweaks — explaining why the DFT (a shallow linear regression) is trivially learnable while the FFT (a deep multiplicative factorization) is not.

In short: the FFT is easy to use and easy to write down, but — without hard algebraic priors — it is intractable to discover by gradient descent.
