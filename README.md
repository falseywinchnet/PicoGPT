#### On the Structural Limits of Attention: Eight Failure Modes and a Phase-Transport Multiscale Alternative
# Abstract

We catalog eight structural failure modes of attention-based sequence models that persist despite scale, data, and modern efficiency tricks. The problems originate from (i) the foundational partition imposed by tokenization, (ii) computational scaling of pairwise mixing, (iii) the absence of intrinsic multi-scale structure, (iv) geometric rather than semantic partitioning, (v) dynamic context collapse under autoregressive training, (vi) in-place evolution of internal representations that erases early conditioning, (vii) preconditioning decay with depth, and (viii) the continuous-decoder limitation (MLPs as phase shifters/codebooks that cannot implement discrete selection). We review near/far-field attention (FMMformer) and multipole clustering (MuSe), which address cost but not structure

FMMformer

2509.10406v2

. We then outline a causal, dyadic phase-transport feature pyramid that encodes token-to-token and sequence-to-sequence relations at multiple granularities, plus small, explicit decision slots for persistent plan/state. This architecture reduces semantic search cost but still inherits the selection problem: equal-importance blending across slices. We close by discussing how these failure modes reinforce one another and what a principled “final act” must provide.

1. Introduction

Transformer attention is a powerful continuous mixing operator for sequences. However, its apparent “reasoning” arises largely from excess capacity conditioned by vast experience, not from structural mechanisms that discover, choose, and enforce semantic partitions on the fly. We articulate eight concrete limits that hold back intrinsic structure, explain how they interact, examine efficiency-focused remedies, and propose a multiscale, purely causal alternative that relocates—but does not yet solve—the core problem of selection.

2. Background (signal-processing lens)

Self-attention computes content-dependent mixing via 
𝑄
𝐾
⊤
QK
⊤
 scores, normalized and applied to 
𝑉
V. Efficiency variants decompose interactions into near-field (local, exact) and far-field (coarse, low-rank/clustered) terms. FMMformer realizes a banded + low-rank split inspired by the Fast Multipole Method

FMMformer

; MuSe clusters keys and queries (separately) and adds multipole (monopole+dipole) corrections to approximate softmax attention in pretraining

2509.10406v2

.

While these reduce cost, they do not give the model an internal, learned semantic partition controller; partitions remain geometric, externally fixed or weakly adapted, and selection across partitions remains continuous and uniformized.

3. Eight failure modes
F1 — Foundational partition (tokenization)

Tokenization sets the atomic units ex ante. Attention cannot evolve “semantic atoms”; it only relates pre-discretized units. Higher structures (words, clauses, motifs) are emergent, noisy, and hostage to the subword grid.

F2 — Quadratic pairwise cost

Full attention scales as 
𝑂
(
𝑇
2
𝐷
)
O(T
2
D). Even with memory-efficient kernels or flash implementations, compute remains quadratic. Efficient splits (sparse, low-rank, FMM-like) trade accuracy/structure for speed

FMMformer

.

F3 — No intrinsic multi-scale structure

Standard stacks do not provide an explicit hierarchy that decides when local syntax vs. global narrative dominates. Heads act as static geometric projections; “scale” is not a first-class control variable.

F4 — Geometric, not semantic partitioning

Windowing, striding, clustering, and low-rank approximations define spatial/geometric partitions. They are not learned semantic groupings tied to task or context; thus partition quality is incidental.

F5 — Dynamic context collapse (autoregressive)

At step 
𝑡
t, the model must learn importance over 
[
1..
𝑡
−
1
]
[1..t−1]. Softmax normalization equalizes competition across a variable-sized set, pushing solutions toward uniform/diffuse weighting for long contexts and over-sharp local bias for short ones—local overfit, global underfit.

F6 — In-place evolution of representations

Residual/FFN updates rewrite the vector manifold in place each layer. Early structure (even if carefully injected) is continuously transformed, so fixed partitioning or conditioning at the input does not persist.

F7 — Preconditioning failure with depth

External structure (positional tags, masks, distance features) dissipates with depth unless re-injected. The system keeps “re-discovering” structure rather than carrying it forward explicitly.

F8 — Continuous-decoder limitation (MLPs as phase shifters)

Within a block, attention “decides where to shift,” while the MLP executes the shift—i.e., it is a codebook/phase transformer implementing smooth affine/polynomial maps. Such continuous decoders do sorting, scaling, fitting, but not symbolic selection/branching. Discrete control remains out of reach without extra machinery.

4. How they reinforce one another

F1 → F3/F4: Fixed tokens nudge all multi-scale strategies to be geometric; semantics can’t become first-class.

F2 → F4/F5: To cut cost, we prune structure geometrically; softmax over variable field sizes then collapses importance globally.

F6/F7 → F8: As layers rewrite the manifold, preconditioning decays; the only persistent operator (MLP) is continuous, so selection reverts to blending.

F5 + F8: Equal-importance pressure meets continuous decoding → coherent fluency without commitment (grammatical, semantically shallow).

5. Prior efficiency remedies & their limits

FMMformer decomposes attention into banded near-field + low-rank far-field, achieving linear complexity and good accuracy, but the split is still geometric and fixed relative to the sequence grid

FMMformer

.
MuSe performs query/key clustering and multipole (monopole/dipole) approximations, gaining runtime speedups at long contexts and acceptable loss increases, but it still externalizes partition choice and performs continuous weighting across clusters
