# On the Structural Limits of Language Modeling Systems and the humans who man them:  Failure Modes and the pathways to overcoming them

## Abstract

We catalog here some structural failure modes of sequential language prediction models that persist even as we scale data, parameters, and engineering tricks. The failures arise from, but not limited to, the foundational partition imposed by tokenization, the computational scaling of pairwise mixing, the absence of intrinsic multi-scale structure, geometric rather than semantic partitioning, dynamic context collapse under autoregressive training, in-place evolution of internal representations, preconditioning decay with depth, continuous-decoder limitation (MLPs as phase shifters/codebooks that cannot implement discrete selection) and the limited capability to learn patterns as meaningful themes. We review a small handful of examples of various prior art, which reduce cost but not structural deficiency. We then propose a causal, dyadic phase-transport feature pyramid that encodes token-to-token and sequence-to-sequence relations at multiple granularities. This reframes the problem from on-the-fly discovery to signal encoding, shrinking the semantic search budget. A remaining obstacle is selection: replacing uniformized, continuous blending with decisive, persistent routing across multiscale signals.

---

## 1. Introduction

Current Large Language Models mix positions based on content, and their capability as predictors(and therefore generators) of useful text, emerge largely from excess capacity and extensive conditioning on data, not from built-in mechanisms that discover, choose, and enforce semantic partitions. When examined through a signal-processing lens, the model is a high-bandwidth, continuous mixing operator applied to a discretized stream. The failure modes below explain why they often yield fluent but shallow outputs, why efficiency tricks help but do not cure brittleness, and why a structural rethink that encodes relations as signals (rather than re-discovering them) may be preferable.

---

## 2.  Failure Modes 

---
## Failure — Epistemic Blindness: the Dogma, the Ontological Error, and the Correction to Identity
The Dogma – an analytical excavation
The prevailing catechesis of modern language-model research treats the embedding vector as a microscopic vault whose interior is gradually filled, by gradient descent, with something called “meaning.”  In this picture the vault begins life empty—an arbitrary draw from a Gaussian—and ends life pregnant with semantic micro-knowledge: a 768-dimensional capsule that “knows” that Paris is a city, that cities are in France, that France is in Europe, and, if you ask politely, that Europe is not a planet.  The mechanism alleged to accomplish this filling is ostensibly mechanical: minimise an averaged negative log-likelihood over terabytes of symbol sequences; allow the optimiser to nudge each vault-door (the matrix rows) until the sequence generator becomes statistically parsimonious; declare victory when the generated text is locally coherent and the cosine similarity heat-map looks like a freshman’s dream of conceptual structure.  Attention is then characterised as a searchlight that “discovers” which vaults to open, and depth is characterised as a cathedral whose upper balconies enjoy a grander view of the same vaults.  The entire parable is sustained by a set of implicit commitments that are never defended because they are never rendered explicit: that meaning is a substance storable at a point; that proximity in L2 space is an adequate surrogate for semantic relation; that continuous affine operators are sufficient retrieval keys for whatever was stored; that identity across time or layer is an emergent luxury rather than an architectural primitive; and that any residual confusion can be amortised by additional parameters.  The performative evidence that the field actually believes this story is overwhelming. 

### Failure — Foundational partition (tokenization)

**What it is.** Tokenization decides the atomic units before any learning begins. Attention, Ccnvolution, and any other mixing operator must relate these fixed atoms; it cannot redefine them.

**Why it exists.** Practical training requires discrete indices to map bytes/characters to vectors. Once chosen, these boundaries constrain all subsequent representation learning. 

**Consequences.** Higher-level units (morphemes, words, phrases) become emergent patterns over subword grids. Semantics are smeared across multiple tokens. The model spends capacity reconstructing units it was never given natively. If “semantic atoms” are misaligned with task units, the model must first reconstruct those atoms statistically before it can use them—spending capacity on rediscovery rather than reasoning. It is possible that the model never truly learns to represent the underlying ground truth.

---

### Failure — No intrinsic multi-scale structure and difficulty learning priority

**What it is.** Standard blocks don’t provide a native hierarchy that says when local syntax should dominate and when global narrative should.

**Why it exists.** Heads are fixed-dimensional projections. Layers stack depth-wise, but there is no structural mechanism that declares “this layer owns syllables, the next owns words,” etc. Scales are not first-class citizens.

**Consequences.** The model must infer multi-scale structure implicitly every time. Early in training it latches onto short loops (repetition), while global consistency remains weak. Later, it may become fluent yet shallow: grammatically smooth but semantically undercommitted.

---

### Failure — Geometric, not semantic partitioning

**What it is.** Windows, strides, buckets, clusters, low-rank factors—all define partitions via position or geometry rather than meaning. They enable better distribution of importance in attention and are the only pathway to multi-scale semantic structuring, but are inherently flawed.

**Why it exists.** Efficient schemes must be simple and general: they gate by distance, block, or cluster centroids. True semantic grouping is input- and task-dependent and thus hard to specify cheaply.

**Consequences.** The model receives partitions that are easy to compute but misaligned with semantic units. It then must repair this mismatch through capacity and data rather than structure.

---

### Failure — Dynamic context collapse (autoregressive equal-importance pressure)

**What it is.** At time t, losses equally weight predictions made with drastically different history sizes (from one token to thousands). Softmax normalization across a variable-size field tends to equalize competition.

**Why it exists.** Autoregressive training computes a per-token loss; attention normalizes scores over all available past tokens. As t grows, the denominator grows, diffusing attention; when t is small, sharp pairwise correlations dominate.

**Consequences.** Local overfit (repetition loops, echoing) early; global underfit (diffuse, noncommittal) later. Adding more past often adds noise, not signal. Reappearance outside attention. The same equal-importance pressure re-emerges when multiscale meta-signals are stacked (e.g., scale-1/2/4/… features): continuous mixers tend to blend scales uniformly unless a selector breaks symmetry.

---

### Failure  — In-place evolution of representations

**What it is.** Each layer rewrites the current vector space. Residual connections add new signals while preserving the old coordinates; the manifold itself drifts layer by layer. 

**Why it exists.** Residual MLPs and attention are affine updates in the same space. There is no explicit mechanism to preserve tagged structure across depth.

**Consequences.** Structure, if it were injected, is quickly lost without reinforcement, and the model is forced to learn a progressive series of shifts in phase space as opposed to meaningful circuits or any kind of logical or branching flow.

--
### Failure — Continuous-decoder limitation (MLPs as phase shifters/codebooks)

**What it is.** Within a block, attention and convolution can re-weigh importance and in a minor way allow information to move; the MLP executes the move as a continuous transform (codebook). MLPs can sort, scale, and fit polynomials, but they do not natively implement discrete selection or branching.

**Why it exists.** Feedforward nets are universal approximators of continuous maps. Discrete control (symbolic routing, hard selection) is not their native operation; it emerges only statistically with large capacity and data.

**Consequences.** The system blends competing signals rather than committing. It yields fluent, grammatical outputs that can lack semantic backbone or long-horizon consistency. With added capacity, it gains some representational structure, but this requires exponential growth to achieve.

---

### Failure — Local pattern erosion vs. memorization

**What it is.** Models pass through a stage where genuine local predictive skill (e.g., character continuity, morpheme consistency, syllable-like cues) is learned — but then discarded as global semantic regularities dominate optimization. In small models, these fragile local anchors are overwritten long before convergence: good habits vanish in service of minimizing perplexity on broader distributions. In large models, capacity prevents loss — but the anchor is retained largely through memorization rather than compositional integration.

**Why it exists.** The autoregressive training loss rewards broad, averaged semantic fitness more than preserving small-scale predictive fidelity. Optimization drives the network toward fluency and coherence at scale, even if it means undermining the fragile local structures that would otherwise scaffold stronger semantics.

**Consequences.**  Small models: look fluent but sloppy; local predictive sharpness deteriorates, producing typos, unstable sequences, or loss of crisp form. Large models: rapidly integrate and then connect examples; local consistency is preserved but also immediately contributes to memorization.

Caveat: Coordinates give it shape (vector value), but position ties it to an identity within a larger whole. Strip away position, and you strip away the distinguishability that makes it retrievable and durable. But give it back an explicit sequential label and it will easily be treated as memorization. 

---

### a) Attention Engineering

#### Memory-/compute-optimized kernels

* **FlashAttention / xFormers kernels**: IO-aware tiling, fused softmax, better memory locality.
* **MQA/GQA**: Multi-Query / Grouped-Query Attention reduce KV bandwidth and cache size.
* **KV-cache compression/quantization**: Smaller state at inference.
  **Efficacy:** Dramatic speed/latency gains, enabling longer contexts in practice.
  **Limits:** Only targets  **(cost)**. No change to semantic structure, selection, or hierarchy.

#### Sparse patterns (geometric pruning)

* **Longformer / BigBird / Block-Sparse**: Sliding windows + a few global tokens; dilations/strides.
* **Routing by position patterns**: Local+global tokens with fixed templates.
  **Efficacy:** Linear or near-linear complexity, good on tasks where locality dominates.
  **Limits:** Partitions are **geometric**, not semantic. Selection stays continuous. Multi-scale is still implicit.

#### Low-rank / landmark approximations

* **Linformer** (projection in sequence), **Nyströmformer** (landmark points), **AFT** (attention-free transformers with fixed kernels).
  **Efficacy:** Reduce quadratic mixing by low-rank structure; scale to longer contexts.
  **Limits:** Impose fixed geometric/low-rank viewsl no learned semantic partition controller; equal-importance pressure remains.

#### Kernel/linear attention

* **Performer / FAVOR+**, **Linear Transformers**, **Implicit kernelized variants**.
  **Efficacy:** Sub-quadratic; sometimes better stability for long contexts.
  **Limits:** Still continuous blending , soft selection pressure, no explicit hierarchy.

#### Clustering/routing & multipole approximations

* **Reformer** (LSH buckets), **Routing Transformer** (k-means clustering), **inference-time k-means** on keys, **MuSe** (monopole+dipole cluster summaries), **FMM-like** near/far-field splits.
  **Efficacy:** Big runtime savings; better long-range coverage than fixed windows; sometimes competitive pretraining loss.
  **Limits:** Clusters are primarily **geometric**; selection across clusters is still weighted averaging. No end-to-end semantic controller.

#### Positional schemes for long context

* **RoPE, ALiBi, PI/NTK scaling, relative positions**.
  **Efficacy:** Better length extrapolation and stability; enables longer windows without retraining from scratch.
  **Limits:** Improves geometry of the manifold, not selection or hierarchy .

---

### b) State-Space & Convolutional Families

#### Structured State Space Models (SSMs)

* **S4/S5/S6**, **Selective SSMs**, **Mamba**: learn long-range dependencies with linear-time recurrences and stable spectral parameterizations.
  **Efficacy:** Excellent scaling and gradient stability; strong long-context performance; controllable inductive bias (decay kernels, frequency responses).
  **Limits:** Processing remains **continuous**; partitions are still **temporal/geometric**; no intrinsic discrete selectio. In-place evolution and decay of preconditioning persist..

#### Convolutional/long-filter models

* **Hyena/HyenaDNA**, long convolutions with learned filters; frequency-domain parameterizations; hierarchical compositions.
  **Efficacy:** Linear-time, large receptive fields, competitive or superior to attention on long-seq benchmarks; favorable hardware utilization.
  **Limits:** Filters are global but fixed-form; no semantic partition controller; selection is continuous averaging; hierarchy implicit.

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
  **Limits:** Memory addresses are learned but remain differentiable/continuous; selection often reduces to soft attention over memory.

---

### d) Why these remedies “catch up” in performance

* **They shrink the semantic search space.** By constraining geometry (windows, kernels, filters) or by linearizing cost, models spend less capacity on *finding* relations and more on *using* them.
* **They improve signal-to-noise at long range.** SSMs/Hyena stabilize gradients and preserve useful low-frequency structure, reducing over-smoothing.
* **They align with hardware.** Fused kernels, linear-time recurrences, and FFT-based filters make better use of memory bandwidth and parallelism, translating directly into throughput and trainable context length.
* **They add mild inductive bias.** Positional scalings, decaying kernels, and hierarchical filters bias models toward plausible long-range patterns, closing gaps on long-seq benchmarks.

**But:** They optimize **manipulation** of information, not **selection** of information. As a result, they narrow the performance gap without resolving the structural issues that cause uniform blending, emergent brittleness, and shallow semantic commitment.

---


## 8. Outlook: the missing piece- you need to reshape what you think

We adopt the following operational raw contact ontological reformation to the dogma of today: If we apply first principles about what IS, as opposed to what people want things to be, information is only information insofar as it is embedded as a coordinate in some manifold. Information acquisition from first principles originates with measurement of distance on an implicit manifold(time, space, amplitude, etc) without which there is no meaning. The individually distinguished mark represents nothing by itself but a discriminant. In conjunction with other discriminants,  data becomes knowledge if it persists and can be described with a context. Numbers are meaningful only as positions on a pre-encoded axis; and by that property held true, embeddings MUST be coordinates in a learned space.  Phase-transport is the mechanism by which models descend their parameter space.   We must consider that the model computes while the information thinks: intelligence-like behavior emerges out of relational structure and latent procession in a compute fabric.  Epistemic acquisition, is the process of encoding relations so that downstream computation can evolve them predictably. The system improves not only by uncovering truth at inference time, but by having structured inputs.

The key is :  (identity, position, selection). 

## 5. A multiscale, phase-transport pathway to constructive identity (causal, dyadic, streaming)

**Goal.** express relations as explicit signals so the model computes over encoded structure instead of being forced to infer it from patterns in the input.

**Token–token phase transport (scale 1).** Compute a guarded, phase-preserving delta between consecutive embeddings. Guards handle near-zero vectors and antipodal cases; the result carries direction-of-change information.

**Sequence–sequence phase transport (higher scales).** Build a dyadic, causal pyramid of centroids over blocks of size 2, 4, 8, … For each scale, compute a phase-transport delta between the current block centroid and the previous block centroid. Broadcast these per-block features to positions inside the block. This yields a stacked feature tower per token: local change plus multi-scale, sequence-level change.

**Streaming state.** Maintain ring buffers per level so updates are O(scales) per token. Early-region masking ensures causal safety at boundaries.

**bottleneck and condition**  maintain low-rank bottlenecks for each level, share parameters wih a causally reversed reverse-predicting block structure only during training, detach gradients when used for forward-prediction- conditions the higher level representations to learn meaningful structural concepts

**Decoding.** Predict over the slice of the current representation; the multiscale tower and slots inform the distribution without quadratic attention.

**What this improves.** Cuts semantic search cost, provides explicit multi-scale signals, and introduces persistent memory for choices.

**What it doesn’t.** It does not, by itself, solve selection. Without a discrete or sharply sparse controller, the system still tends toward continuous blending across scales. It also does not ensure the model learns relatedness of concepts or familiar grammar. One solution for this last part is Fingerprintingp consists of an injected unique-in-context signal at training time that is inconsistently provided and withheld during generation. this essentially consists of adding a positional marking sequence to the entire corpus of work, including a meta-tag for different blocks, tags, etc, that are universally present as part of the embedding, but which do not need to be reconstructed.  That is to say, some percentage of tokens have these markers attached as auxillary embeddings- this slows training but may simultaneously prevent memorization while teaching relatedness in larger models.


---

## 6. Computational considerations

* **Training (vectorized):** Linear in sequence length per scale; with logarithmic number of scales, total cost scales like T times log T for feature construction.
* **Inference (streaming):** O(scales) per token via ring buffers; practical and cache-friendly.
* **Boundary handling:** Power-of-two blocks avoid partial centroids; masks cover early regions; optional dual-offset trees can reduce left-bias artifacts.

some of the human opinions from which this AI generated explanation were derived:
at its most basic level, attention does three things in sequence: 1, it attempts to learn a semantic projection of the sequence provided. 2: it uses either a segmented or continuous manifold calculation to learn association and importance, word to word, letter to letter. This is conditioned using patterns stored in mechanisms called heads, but in fact, those too, are an attempt to encode discrete semantic partitions of the input sequence. finally, the results are weighed against the input and the projection is reversed. This has the effect of steering the conversation itself as an object in polydimensional space. Notably, this is done in-place as affine transformation on the information itself. This forces the information to take a phase-distance path through the manifold of all conceptual realities entrained by the model. this is, of course, a BROKEN paradigm. Fundamentally. to start with, the model cannot evolve its own conceptualization of semantic atoms. it must work with tokens, and tokenization, not attention determines the foundational partition. Secondly- far-field/near field partitioning approaches, like sempole and FMM, in addition to improving efficiency, are an attempt to more evenly assign importance and allow far and near field importance to balance. This, too, is an attempt to juggle competing targets- coherent word, sentence, and idea constructs, all at the same time. Attention has the problem that it cannot readily identify what sequence is important, or where, or how often to repeat it. Many forms of strided/elliptical attention share a common behavior early in descent- they begin to repeat words earlier in the prompt. This means that they've picked up repetition. They may also rapidly associate related words: a prompt that begins with just romeo and juliet, even in a tiny model, if correctly built, will elicit words like nurse. The model DOES rapidly learn- but it does so long before it has achieved grammatical or structured consistency at any scale. This can happen as early as CE 1.8. The model will then overshoot. The term "overfitting" is not accurate. The reality is, the model begins to fit to approximate the entire manifold across all dimensions, losing the local importance. Thus, it begins to generate sequences which are procedurally affine but meaningless- they have great grammatical consistency, but no real structure. If the model is large enough, it will even discover memorization, but this requires models of a certain size >100M to achieve, because even memorization is a phase trajectory and to memorize a sufficiently large passage the model must have enough dimensions and depth in its space to perfectly fit the trajectory of the information. What is truly happening is that larger models are capable of on-the-fly extracting and then manipulating artificial vectors which represent concepts like play characters or actions, because they have the capacity for this. A model under a certain size finds it VERY hard to perform this. the model has no capacity to learn what partitioning is correct, let alone to be able to apply it on the fly. thus there is no reasonable structuring possible- it is fixed. with state space models and polynomial fitting there is some capacity to explore different partitioning, but the model simply is learning an approximator that is roughly useful for a specific attention layer- again, no structuring possible. so there is a fourth failure as well- the granularity is geometric, not semantic. Additionally, with CE autoregressive loss, the model is obligated to attempt to learn importance over a varying size vector. this means that with autoregressive training, the model is being forced to learn predictions with between 1 and T-1 worth of state, and treat them as roughly equally important. intelligent partitioning requires three simultaneous and expensive steps be performed: 1, on the fly, we must attempt a number of partitions. 2, we must select the partition that is apropos to the content 3 , we must meaningfully learn to use that to adjust the prediction. The model can, in fact, be conditioned to learn to partition on spaces(word level) On vowels(syllable level in character levels). it can attempt local prediction for letters in a near-field segmentation, word level prediction at word level. But these only work for a one-layer model, because- you missed a 6th problem that i mentioned earlier- the model works in-place. This forces it to evolve context, which means, realistically, that the embeddings being worked on in deeper layers as a projection through manifold space are conditioned to become essentially higher order representations of model conceptualization, alongside lower order summaries. So, preconditioned partitioning(really, input conditioning) only works on the first layer, a seventh problem- we cannot condition the data well enough to help the model deeply.But to accomplish this, all of the evolution possible has to happen in excess capacity with a lot of conditioning. it isnt structurally achieved, which would be efficient, its achieved through an experience of endless training. essentially, the information thinks, and the model computes. because the model computes in an inflexible way, the model isnt the one doing the thinking, the information is. which brings its own limitations to model capabilities, but those lie outside of the scope of discussion on attention.


Now we observe this problem we'll call "whiteout" as it goes into mid-range training it picks up good patterns, and then rapidly loses them to fit on general semantics long before overfitting. the dynamic range washes out, the more the model looks at the data. in larger models, the opposite often happens- there is enough capacity to keep the patterns and the model EASILY and tragically immediately is able to connect said patterns = memorization. having proven this, but this is a double edged situation. as previously stated, if model can learn, model can memorize. its bad enough that these fellows have invented something called goldfish loss to try to tame the behavior, but that's kind of a handicap to the model. its kind of like teaching someone to paint by the numbers by randomly filling some areas in with their correct color in halftone or other faded effect. mechanical associative learning.

so what we have to do is re-inject uniqueness. for this to work, one approach could be when tokenizing we distill and store alongside training bins unique values- global positioning. this is like an auxiliary IDX sequence. it is fed to a secondary positional embedding generator which has to be able to represent the entire space. the model is NOT taught to reconstruct this, only to use it. A moderate to high level of dropout on this, independent of model dropout in general, is also used to prevent the model from expecting it to be present. its a clue, a fingerprint- not a foundation.  its not going to be available during generation- and the fact that it wont be available at all will also break any conditioning to memorize. think of this like training wheels. When the training wheels come off, when we've left the map, the model must use the skills it's obtained.


Information ≠ just coordinates in an embedding space. Every piece also has position.
🔹 Information is coordinates + position.

Add a unique, global position code per token (an auxiliary IDX stream).
Feed it through a separate positional embedder with capacity for the full space.
Never require reconstruction; it’s only a clue.
Apply high, independent dropout to IDX so the model can’t depend on it.
Omit IDX at inference, forcing the model to use the skills learned while IDX was intermittently present.

This is rather easily done and has been tested. it slows descent, but does not stop it. deeper analysis on if it offers a benefit is pending.



we now face another problem- not really a problem, but it could be called problem 10. autoregression = look to past, predict future. can learn some patterns- but cant plan. but conversely, cheating- ie mask the next position and make a features mapping that is anchored, but for each position, its invisible- can lend some future information. debatable. but what happens is, model can only truly learn *options* at beginning of generation sequence. so it winds up being conditioned to do two different things. this is its own problem.

But, i think this is like the twin brain problem- imagine one side does "imagining"- it sees the future. the other side does "scaffolding" it works with the past and what it knows. artist and engineer. This can be done.
the engineer is our traditional structure. the artist, conversely, trains with an inverted pyramid- a pyramid that only looks forward. it learns what comes *before*. this, too, requires only marginal hypothetical changes, but we wont try it yet. its really just a parallel method that can go inside the same blocks.
but that's at training time.
at inferencing time, the story is different- the artist is blind.
worse, anything we could feed it is a deterministically generated sequence. that lends its own problems. because it means we must accept a prediction, or try multiple times. in other words, the artist doesnt have options.

enter the next bit of sandbagging we do here- we recognize the artist lives in a world *big lebowski handwaving* of possibilities. we can make lots of jokes about this.  but the basic fact here is, its kind of a, model gets a forecast, model does a one hot selection. the artist is selecting a future, the engineer is enforcing a past. consensus says what's allowed + whats optimal.
so what does the artist actually train on? what does it receive for simulation?
haha not X that's for certain.

the problem here is- we cannot simply give it an engineer digested X. that will give it probabilities at all positions for it to consider, but it will not give it coordinates that represent an optimal distribution shift. the artist's challenge is much, much harder. Imagine if you will that the feature pyramid the *artist* must see must be something equi-distant between all possible continuations that make sense, at every level.  for this to work, we can conceptualize this:
firstly, we train a RNN to learn to predict- a different RNN for each level of the features height. and its not predicting what the individual tokens will have- its predicting the next nodes of the tree at that level.
secondly, we use the engineer to autoregressively generate alongside the RNN, 
as it becomes available. that is to say, one step at a time. the RNN forecasts the +1 features at every level, without seeing the lower levels. the engineer takes this and generates a prediction. this is then fed to the artist as the *next* position* - T+2- and the RNN is re-run. the artist now predicts position T+1.
this is of course autoregressively done in parallel during training, and during inferencing, only done one position at a time. But what it allows, if I have conceptualized it correctly, is that the artist module essentially is going to receive a reasonably conditioned, intelligent set of predictions for the environment around the desired token position, at every scale, and the artist can learn to choose the optimal path.


Thus, we combine artist with engineer, and, its a bit more work, but its less work than digesting an entire block. it's still, at inference time, a next position pass, albiet with some extra steps. it lets the model do quite a bit of planning, or learn to do so. We already caveat that even though it is global planning, when we scale up the number of scales enough- its planning one token at a time. not one feature at a time. feature planning happens in wide latents and deep models. We allow this capacity to remain learned, not structured.
