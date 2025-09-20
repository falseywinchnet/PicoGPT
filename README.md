# On the Structural Limits of Language Modeling Systems: Eight Failure Modes and a Phase-Transport Multiscale Option

## Abstract

We catalog eight structural failure modes of sequential language prediction models that persist even as we scale data, parameters, and engineering tricks. The failures arise from (1) the foundational partition imposed by tokenization, (2) the computational scaling of pairwise mixing, (3) the absence of intrinsic multi-scale structure, (4) geometric rather than semantic partitioning, (5) dynamic context collapse under autoregressive training, (6) in-place evolution of internal representations, (7) preconditioning decay with depth, and (8) the continuous-decoder limitation (MLPs as phase shifters/codebooks that cannot implement discrete selection). We review various prior art, which reduce cost but not structural deficiency. We then propose a causal, dyadic phase-transport feature pyramid that encodes token-to-token and sequence-to-sequence relations at multiple granularities, plus small, explicit “decision slots” for persistent plan/state. This reframes the problem from on-the-fly discovery to signal encoding, shrinking the semantic search budget. The remaining obstacle is selection: replacing uniformized, continuous blending with decisive, persistent routing across multiscale signals.

---

## 1. Introduction

Transformer models mix positions based on content, and their capability as predictors(and therefore generators) of useful text, emerge largely from excess capacity and extensive conditioning on data, not from built-in mechanisms that discover, choose, and enforce semantic partitions. When examined through a signal-processing lens, attention is a high-bandwidth, continuous mixing operator applied to a discretized stream. The eight failure modes below explain why transformers often yield fluent but shallow outputs, why efficiency tricks help but do not cure brittleness, and why a structural rethink that encodes relations as signals (rather than re-discovering them) may be preferable.

---

## 2. The Eight Failure Modes (with detailed reasoning)

### Failure 1 — Foundational partition (tokenization)

**What it is.** Tokenization decides the atomic units before any learning begins. Attention must relate these fixed atoms; it cannot redefine them.

**Why it exists.** Practical training requires discrete indices to map bytes/characters to vectors. Once chosen, these boundaries constrain all subsequent representation learning.

**Consequences.** Higher-level units (morphemes, words, phrases) become emergent patterns over subword grids. Semantics are smeared across multiple tokens. The model spends capacity reconstructing units it was never given natively. If “semantic atoms” are misaligned with task units, the model must first reconstruct those atoms statistically before it can use them—spending capacity on rediscovery rather than reasoning.

**Reinforcement.** This feeds Failure 3 (no explicit multi-scale) and Failure 4 (geometric, not semantic partitions), because any hierarchy that follows is built on a subword lattice rather than task-induced units.

**Intuition.** If your ruler marks are wrong, every later measurement inherits that bias.

---

### Failure 2 — Quadratic pairwise cost

**What it is.** Full attention compares a query to all prior keys. Cost scales with the square of sequence length.

**Why it exists.** Attention is a dense, global operator by design: all-to-all content-dependent mixing.

**Consequences.** Long contexts are expensive. We either truncate, sparsify, approximate, or pay the bill. Efficiency methods optimize memory and flops but leave semantics untouched.

**Reinforcement.** The urge to prune leads to geometric shortcuts (Failure 4), which further disconnect partitions from meaning.

**Intuition.** A room where everyone talks to everyone is expressive—but noisy and costly. Microphones help; they don’t create an agenda.

---

### Failure 3 — No intrinsic multi-scale structure

**What it is.** Standard blocks don’t provide a native hierarchy that says when local syntax should dominate and when global narrative should.

**Why it exists.** Heads are fixed-dimensional projections. Layers stack depth-wise, but there is no structural mechanism that declares “this layer owns syllables, the next owns words,” etc. Scales are not first-class citizens.

**Consequences.** The model must infer multi-scale structure implicitly every time. Early in training it latches onto short loops (repetition), while global consistency remains weak. Later, it may become fluent yet shallow: grammatically smooth but semantically undercommitted.

**Reinforcement.** Failure 1 fixes atoms at the wrong level, Failure 2 pushes us to prune globally, and the absence of an explicit hierarchy multiplies the difficulty of learning stable long-range dependencies.

**Intuition.** Without a map’s zoom levels, navigation alternates between too much detail and not enough context.

---

### Failure 4 — Geometric, not semantic partitioning

**What it is.** Windows, strides, buckets, clusters, low-rank factors—all define partitions via position or geometry rather than meaning.

**Why it exists.** Efficient schemes must be simple and general: they gate by distance, block, or cluster centroids. True semantic grouping is input- and task-dependent and thus hard to specify cheaply.

**Consequences.** The model receives partitions that are easy to compute but misaligned with semantic units. It then must repair this mismatch through capacity and data rather than structure.

**Reinforcement.** Failures 2 and 3 encourage such geometric partitions; Failure 5 (equalizing pressure) then washes out any tentative semantic signal.

**Intuition.** Cutting a novel into five-page chunks is convenient for printing, not for understanding its themes.

---

### Failure 5 — Dynamic context collapse (autoregressive equal-importance pressure)

**What it is.** At time t, losses equally weight predictions made with drastically different history sizes (from one token to thousands). Softmax normalization across a variable-size field tends to equalize competition.

**Why it exists.** Autoregressive training computes a per-token loss; attention normalizes scores over all available past tokens. As t grows, the denominator grows, diffusing attention; when t is small, sharp pairwise correlations dominate.

**Consequences.** Local overfit (repetition loops, echoing) early; global underfit (diffuse, noncommittal) later. Adding more past often adds noise, not signal. Reappearance outside attention. The same equal-importance pressure re-emerges when multiscale meta-signals are stacked (e.g., scale-1/2/4/… features): continuous mixers tend to blend scales uniformly unless a selector breaks symmetry.

**Reinforcement.** This interacts with Failure 8 (continuous decoders) to promote blending over choosing; with Failure 3, it impedes robust multi-scale control.

**Intuition.** If you must weigh a handful of items and a warehouse with the same scale, you won’t measure either well.

---

### Failure 6 — In-place evolution of representations

**What it is.** Each layer rewrites the current vector space. Residual connections add new signals while preserving the old coordinates; the manifold itself drifts layer by layer.

**Why it exists.** Residual MLPs and attention are affine updates in the same space. There is no explicit mechanism to preserve tagged structure across depth.

**Consequences.** Any structure injected at the input (special tags, distances) is re-encoded repeatedly and loses legibility. Downstream layers must re-derive what upstream once knew. This drift converts crisp tags into distributed hints; without protected channels, later layers inherit only a shadow of early structure.

**Reinforcement.** Fuels Failure 7 (preconditioning decay) and amplifies Failure 8 (decoders default to smooth blends when structure is ambiguous).

**Intuition.** Redrawing the map after every turn makes early annotations unreliable.

---

### Failure 7 — Preconditioning decay with depth

**What it is.** Externally injected structure (positional tags, hierarchy hints, distance features) fades unless reintroduced at every stage.

**Why it exists.** Without a persistent channel reserved for structure, generic mixing erodes explicit signals. Depth acts like a lowpass on tags that aren’t protected.

**Consequences.** The model spends compute re-discovering the same relations. Training feels like pushing a boulder uphill: every layer forgets what the last layer wrote.

**Reinforcement.** Coupled with Failure 6, this ensures that any initial semantic scaffolding dissolves unless maintained explicitly.

**Intuition.** Sticky notes fall off during a long trip unless you keep re-sticking them.

---

### Failure 8 — Continuous-decoder limitation (MLPs as phase shifters/codebooks)

**What it is.** Within a block, attention decides where information should move; the MLP executes the move as a continuous transform (codebook). MLPs can sort, scale, and fit polynomials, but they do not natively implement discrete selection or branching.

**Why it exists.** Feedforward nets are universal approximators of continuous maps. Discrete control (symbolic routing, hard selection) is not their native operation; it emerges only statistically with large capacity and data.

**Consequences.** The system blends competing signals rather than committing. It yields fluent, grammatical outputs that can lack semantic backbone or long-horizon consistency.

**Reinforcement.** Combined with Failure 5, this leads to “equal-importance blending”; with Failures 6–7, any early discrete hints get washed into smooth averages.

**Intuition.** A mixing board can balance channels exquisitely, but it cannot decide which instrument should play the solo.

---

### Failure 9 — Local pattern erosion vs. memorization

**What it is.** Models pass through a stage where genuine local predictive skill (e.g., character continuity, morpheme consistency, syllable-like cues) is learned — but then discarded as global semantic regularities dominate optimization. In small models, these fragile local anchors are overwritten long before convergence: good habits vanish in service of minimizing perplexity on broader distributions. In large models, capacity prevents loss — but the anchor is retained largely through memorization rather than compositional integration.

**Why it exists.** The autoregressive training loss rewards broad, averaged semantic fitness more than preserving small-scale predictive fidelity. Optimization drives the network toward fluency and coherence at scale, even if it means undermining the fragile local structures that would otherwise scaffold stronger semantics.

**Consequences.**  Small models: look fluent but sloppy; local predictive sharpness deteriorates, producing typos, unstable sequences, or loss of crisp form. Large models: rapidly integrate and then connect examples; local consistency is preserved but also immediately contributes to memorization. 

**Reinforcement.** Failure 1 (tokenization) forces subwords to be emergent, making local cues expendable. Failure 3 (no multi-scale) prevents the integration of local → higher structures, so small anchors are either dropped or rote-memorized. Failure 5 (equal-importance pressure) ensures that global signals dominate loss reduction, washing out fragile short-range skills.

**intuition.** muscle memory of a snail vs an elephant


## 3. How the failures entangle

* **F1 → F3/F4.** Fixed atoms force emergent hierarchy; practical partitions become geometric.
* **F2 → F4/F5.** Cost pressures prefer simple partitions and uniform normalization, encouraging equal-importance behavior.
* **F6 → F7/F8.** In-place rewrites cause structural drift; continuous decoders prefer blending as legibility degrades.
* **F5 + F8.** Variable-field normalization meets continuous decoding, yielding “coherent fluency without commitment.”

---

## 4: Prior Efficacy Remedies (for integration)

This survey shows many ways to move information better (scale, speed, stability), but very few ways to choose information better. They narrow the search budget and close benchmark gaps—but leave partitioning geometric, selection continuous, structure drifting, and preconditioning fading.he theme: most advances optimize *how information is moved*, not *how information is chosen*. They close performance gaps by lowering the semantic search budget or stabilizing training, but leave selection and semantic partitioning largely unresolved. they constrain geometry, linearize cost, stabilize gradients, and bias toward useful long-range patterns. Yet, across families, the core limitations remain: partitions are still **geometric**, selection is **continuous blending**, structure **drifts in place**, and preconditioning **fades with depth**. They elevate efficacy, not structural semantics.  

---

### a) Attention Engineering

#### Memory-/compute-optimized kernels

* **FlashAttention / xFormers kernels**: IO-aware tiling, fused softmax, better memory locality.
* **MQA/GQA**: Multi-Query / Grouped-Query Attention reduce KV bandwidth and cache size.
* **KV-cache compression/quantization**: Smaller state at inference.
  **Efficacy:** Dramatic speed/latency gains, enabling longer contexts in practice.
  **Limits:** Only targets **F2 (cost)**. No change to semantic structure, selection, or hierarchy (**F3–F8 persist**).

#### Sparse patterns (geometric pruning)

* **Longformer / BigBird / Block-Sparse**: Sliding windows + a few global tokens; dilations/strides.
* **Routing by position patterns**: Local+global tokens with fixed templates.
  **Efficacy:** Linear or near-linear complexity, good on tasks where locality dominates.
  **Limits:** Partitions are **geometric**, not semantic (**F4**). Selection stays continuous (**F5/F8**). Multi-scale is still implicit (**F3**).

#### Low-rank / landmark approximations

* **Linformer** (projection in sequence), **Nyströmformer** (landmark points), **AFT** (attention-free transformers with fixed kernels).
  **Efficacy:** Reduce quadratic mixing by low-rank structure; scale to longer contexts.
  **Limits:** Impose fixed geometric/low-rank views (**F4**); no learned semantic partition controller (**F3**); equal-importance pressure remains (**F5**).

#### Kernel/linear attention

* **Performer / FAVOR+**, **Linear Transformers**, **Implicit kernelized variants**.
  **Efficacy:** Sub-quadratic; sometimes better stability for long contexts.
  **Limits:** Still continuous blending (**F8**), soft selection pressure (**F5**), no explicit hierarchy (**F3**).

#### Clustering/routing & multipole approximations

* **Reformer** (LSH buckets), **Routing Transformer** (k-means clustering), **inference-time k-means** on keys, **MuSe** (monopole+dipole cluster summaries), **FMM-like** near/far-field splits.
  **Efficacy:** Big runtime savings; better long-range coverage than fixed windows; sometimes competitive pretraining loss.
  **Limits:** Clusters are primarily **geometric**; selection across clusters is still weighted averaging (**F4/F5/F8**). No end-to-end semantic controller (**F3**).

#### Positional schemes for long context

* **RoPE, ALiBi, PI/NTK scaling, relative positions**.
  **Efficacy:** Better length extrapolation and stability; enables longer windows without retraining from scratch.
  **Limits:** Improves geometry of the manifold, not selection or hierarchy (**F3–F5, F8** unchanged).

---

### b) State-Space & Convolutional Families

#### Structured State Space Models (SSMs)

* **S4/S5/S6**, **Selective SSMs**, **Mamba**: learn long-range dependencies with linear-time recurrences and stable spectral parameterizations.
  **Efficacy:** Excellent scaling and gradient stability; strong long-context performance; controllable inductive bias (decay kernels, frequency responses).
  **Limits:** Processing remains **continuous**; partitions are still **temporal/geometric**; no intrinsic discrete selection (**F4/F8**). In-place evolution and decay of preconditioning persist (**F6/F7**).

#### Convolutional/long-filter models

* **Hyena/HyenaDNA**, long convolutions with learned filters; frequency-domain parameterizations; hierarchical compositions.
  **Efficacy:** Linear-time, large receptive fields, competitive or superior to attention on long-seq benchmarks; favorable hardware utilization.
  **Limits:** Filters are global but fixed-form; no semantic partition controller; selection is continuous averaging; hierarchy implicit (**F3/F5/F8**).

#### Retention / hybrid recurrent families

* **RetNet** (retention mechanism), **RWKV** (RNN/Transformer hybrid): approximate attention with recurrent operators and better memory footprints.
  **Efficacy:** Streaming-friendly; long horizons with stable compute; sometimes matches attention on perplexity.
  **Limits:** Still continuous routing; lacks discrete, persistent semantic selection; geometric rather than semantic grouping.

---

### c) Capacity & Routing at the Model Scale

#### Mixture-of-Experts (MoE)

* **Switch/GShard/GLaM-style** sparse expert routing.
  **Efficacy:** Scales parameter count without proportional compute; conditional computation boosts capacity.
  **Limits:** Routing is learned but remains a **soft/continuous gate** in practice; does not instantiate semantic hierarchy within sequences; selection often noisy without strong priors.

#### External or learned memory

* **RAG/kNN-LM**, **Memory Transformers**, **DNC/NTM-style** controllers.
  **Efficacy:** Offloads recall; improves factual grounding and long-horizon tasks.
  **Limits:** Memory addresses are learned but remain differentiable/continuous; selection often reduces to soft attention over memory—reintroducing **F5/F8**.

---

### d) Why these remedies “catch up” in performance

* **They shrink the semantic search space.** By constraining geometry (windows, kernels, filters) or by linearizing cost, models spend less capacity on *finding* relations and more on *using* them.
* **They improve signal-to-noise at long range.** SSMs/Hyena stabilize gradients and preserve useful low-frequency structure, reducing over-smoothing.
* **They align with hardware.** Fused kernels, linear-time recurrences, and FFT-based filters make better use of memory bandwidth and parallelism, translating directly into throughput and trainable context length.
* **They add mild inductive bias.** Positional scalings, decaying kernels, and hierarchical filters bias models toward plausible long-range patterns, closing gaps on long-seq benchmarks.

**But:** They optimize **manipulation** of information, not **selection** of information. As a result, they narrow the performance gap without resolving the structural issues that cause uniform blending, emergent brittleness, and shallow semantic commitment.

---

### e) Mapping remedies to structural failures

| Remedy family                          | F1 Tokenization | F2 Cost             | F3 Multi-scale          | F4 Geometric vs. Semantic | F5 Equal-importance                   | F6 In-place evolution | F7 Preconditioning decay | F8 Continuous decoder |
| -------------------------------------- | --------------- | ------------------- | ----------------------- | ------------------------- | ------------------------------------- | --------------------- | ------------------------ | --------------------- |
| Flash/xFormers/MQA/GQA                 | –               | **✓✓**              | –                       | –                         | –                                     | –                     | –                        | –                     |
| Sparse attention (Longformer/BigBird)  | –               | **✓**               | (±) implicit            | **✗** geometric           | –                                     | –                     | –                        | –                     |
| Low-rank/landmarks (Linformer/Nyström) | –               | **✓**               | –                       | **✗** geometric           | –                                     | –                     | –                        | –                     |
| Kernel/linear (Performer, Linear Attn) | –               | **✓**               | –                       | **✗** geometric           | –                                     | –                     | –                        | **✗** continuous      |
| Clustering/FMM/MuSe                    | –               | **✓**               | (±) hierarchical flavor | **✗** geometric clusters  | –                                     | –                     | –                        | **✗** continuous      |
| Positional schemes (RoPE/ALiBi)        | –               | –                   | (±) better scaling      | –                         | –                                     | –                     | –                        | –                     |
| SSMs (S4/Mamba)                        | –               | **✓✓**              | (±) via filters         | **✗** temporal geometry   | –                                     | –                     | –                        | **✗** continuous      |
| Long convs (Hyena)                     | –               | **✓✓**              | (±) via hier. filters   | **✗** temporal geometry   | –                                     | –                     | –                        | **✗** continuous      |
| Retention/RNN hybrids (RetNet/RWKV)    | –               | **✓**               | (±)                     | **✗** temporal geometry   | –                                     | –                     | –                        | **✗** continuous      |
| MoE (Switch/GLaM)                      | –               | (compute per token) | –                       | –                         | –                                     | –                     | –                        | **✗** soft gates      |
| External memory / RAG                  | –               | (offloads)          | –                       | –                         | – (often reintroduces soft attention) | –                     | –                        | **✗** continuous      |

Legend: **✓✓** strong improvement, **✓** moderate, (±) partial/indirect, **✗** unaddressed; “–” = no material effect.


---

## 5. A multiscale, phase-transport alternative (causal, dyadic, streaming)

**Goal.** Re-express relations as explicit signals so the model computes over encoded structure instead of re-discovering it every step.

**Token–token phase transport (scale 1).** Compute a guarded, phase-preserving delta between consecutive embeddings. Guards handle near-zero vectors and antipodal cases; the result carries direction-of-change information.

**Sequence–sequence phase transport (higher scales).** Build a dyadic, causal pyramid of centroids over blocks of size 2, 4, 8, … For each scale, compute a phase-transport delta between the current block centroid and the previous block centroid. Broadcast these per-block features to positions inside the block. This yields a stacked feature tower per token: local change plus multi-scale, sequence-level change.

**Streaming state.** Maintain ring buffers per level so updates are O(scales) per token. Early-region masking ensures causal safety at boundaries.

**Decision slots (persistent plan/state).** Reserve small sub-vectors carried alongside the main stream where the model can “write” commitments (theme, plan, speaker, constraint) and “read” them later. These slots are the place to keep decision trees—lightweight persistent state that survives layer drift.

**Decoding.** Predict over the slice of the current representation; the multiscale tower and slots inform the distribution without quadratic attention.

**What this improves.** Cuts semantic search cost, provides explicit multi-scale signals, and introduces persistent memory for choices.

**What it doesn’t.** It does not, by itself, solve selection. Without a discrete or sharply sparse controller, the system still tends toward continuous blending across scales.

---

## 6. Computational considerations

* **Training (vectorized):** Linear in sequence length per scale; with logarithmic number of scales, total cost scales like T times log T for feature construction.
* **Inference (streaming):** O(scales) per token via ring buffers; practical and cache-friendly.
* **Boundary handling:** Power-of-two blocks avoid partial centroids; masks cover early regions; optional dual-offset trees can reduce left-bias artifacts.

---

## 7. Epistemic axioms: information as embedded meta-vector

We adopt the following operational axioms: information is only information insofar as it is embedded as a coordinate in some manifold. Numbers are meaningful as positions on a pre-encoded axis; language tokens are meaningful as coordinates in a learned space. Phase-transport can work if embeddings are literal coordinates; relations can be encoded as stable geometric transforms. We must consider that the model computes while the information thinks: intelligence-like behavior emerges out of relational structure and latent procession in a compute fabric.  Epistemic acquisition, then, is the process of encoding relations so that downstream computation can evolve them predictably. The system improves not by uncovering truth at inference time, but by making structure explicit before mixing.

---

## 8. Outlook: the missing piece (selection)

All roads lead to selection. A satisfactory controller must:

* Choose among multi-scale features without imposing equal-importance normalization across variable fields.
* Persist choices across time (so plans survive depth and layer drift).
* Remain computationally light.

Absent this, the eight failures are merely relocated: from token mixing to signal mixing. The multiscale, phase-transport tower plus decision slots sets the stage; the final act is decisiveness.


*(Citations omitted here for format; see accompanying files for specific works.)*

some of the human opinions from which this AI generated explanation were derived:
at its most basic level, attention does three things in sequence: 1, it attempts to learn a semantic projection of the sequence provided. 2: it uses either a segmented or continuous manifold calculation to learn association and importance, word to word, letter to letter. This is conditioned using patterns stored in mechanisms called heads, but in fact, those too, are an attempt to encode discrete semantic partitions of the input sequence. finally, the results are weighed against the input and the projection is reversed. This has the effect of steering the conversation itself as an object in polydimensional space. Notably, this is done in-place as affine transformation on the information itself. This forces the information to take a phase-distance path through the manifold of all conceptual realities entrained by the model. this is, of course, a BROKEN paradigm. Fundamentally. to start with, the model cannot evolve its own conceptualization of semantic atoms. it must work with tokens, and tokenization, not attention determines the foundational partition. Secondly- far-field/near field partitioning approaches, like sempole and FMM, in addition to improving efficiency, are an attempt to more evenly assign importance and allow far and near field importance to balance. This, too, is an attempt to juggle competing targets- coherent word, sentence, and idea constructs, all at the same time. Attention has the problem that it cannot readily identify what sequence is important, or where, or how often to repeat it. Many forms of strided/elliptical attention share a common behavior early in descent- they begin to repeat words earlier in the prompt. This means that they've picked up repetition. They may also rapidly associate related words: a prompt that begins with just romeo and juliet, even in a tiny model, if correctly built, will elicit words like nurse. The model DOES rapidly learn- but it does so long before it has achieved grammatical or structured consistency at any scale. This can happen as early as CE 1.8. The model will then overshoot. The term "overfitting" is not accurate. The reality is, the model begins to fit to approximate the entire manifold across all dimensions, losing the local importance. Thus, it begins to generate sequences which are procedurally affine but meaningless- they have great grammatical consistency, but no real structure. If the model is large enough, it will even discover memorization, but this requires models of a certain size >100M to achieve, because even memorization is a phase trajectory and to memorize a sufficiently large passage the model must have enough dimensions and depth in its space to perfectly fit the trajectory of the information. What is truly happening is that larger models are capable of on-the-fly extracting and then manipulating artificial vectors which represent concepts like play characters or actions, because they have the capacity for this. A model under a certain size finds it VERY hard to perform this. the model has no capacity to learn what partitioning is correct, let alone to be able to apply it on the fly. thus there is no reasonable structuring possible- it is fixed. with state space models and polynomial fitting there is some capacity to explore different partitioning, but the model simply is learning an approximator that is roughly useful for a specific attention layer- again, no structuring possible. so there is a fourth failure as well- the granularity is geometric, not semantic. Additionally, with CE autoregressive loss, the model is obligated to attempt to learn importance over a varying size vector. this means that with autoregressive training, the model is being forced to learn predictions with between 1 and T-1 worth of state, and treat them as roughly equally important. intelligent partitioning requires three simultaneous and expensive steps be performed: 1, on the fly, we must attempt a number of partitions. 2, we must select the partition that is apropos to the content 3 , we must meaningfully learn to use that to adjust the prediction. The model can, in fact, be conditioned to learn to partition on spaces(word level) On vowels(syllable level in character levels). it can attempt local prediction for letters in a near-field segmentation, word level prediction at word level. But these only work for a one-layer model, because- you missed a 6th problem that i mentioned earlier- the model works in-place. This forces it to evolve context, which means, realistically, that the embeddings being worked on in deeper layers as a projection through manifold space are conditioned to become essentially higher order representations of model conceptualization, alongside lower order summaries. So, preconditioned partitioning(really, input conditioning) only works on the first layer, a seventh problem- we cannot condition the data well enough to help the model deeply.But to accomplish this, all of the evolution possible has to happen in excess capacity with a lot of conditioning. it isnt structurally achieved, which would be efficient, its achieved through an experience of endless training. essentially, the information thinks, and the model computes. because the model computes in an inflexible way, the model isnt the one doing the thinking, the information is. which brings its own limitations to model capabilities, but those lie outside of the scope of discussion on attention.
