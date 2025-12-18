# MV‑DCB (Multi‑Vector, Depth‑Coupled Backprop)

Traditional backprop sends error **through a single dot‑product channel** at each layer. That’s like trying to reposition a heavy table by pushing with **one finger** straight through the middle: you move, but you also skid, wobble, and waste effort.

**MV‑DCB** treats everything as **coordinates** (inputs, activations, weights, logits) and moves them with a **small team of coordinated pushes**. At each layer it builds a tiny, data‑conditioned **span of directions** (2–4 vectors) that reflect the relevant local geometry **coupled to deeper layers**, projects the error into that span, and only then updates. You still use Adam/SGD exactly the same way; we only change *what gradient you feed them*.

---

## What actually changes vs. vanilla backprop

**Vanilla** (per layer):

* Hidden‑grad `s = r_y · Wᵀ` (one direction implied by the weight matrix).

**MV‑DCB** (per layer):

1. Form a tiny basis **D = [d₁…d_K]** in hidden space (K≈2–4). The basis can include:

   * A row‑space direction from `Wᵀ r_y` (what vanilla already “wants”).
   * One or two **depth‑coupled** directions that line up with downstream sensitivity (what deeper layers care about *here*).
2. Compute the **projection** of the residual into that span: `proj = D (Dᵀ D + λI)⁻¹ Dᵀ r_y` (with small ridge λ for stability).
3. Blend with the orthogonal remainder: `s = α · proj + β · (r_y − proj)` (α≈1.0, β≈0.2–0.4).
4. Use `s` to form parameter grads as usual (`dW = sᵀ a_prev`, `db = sum(s)`), then call your optimizer.

> **Result:** we don’t let the gradient push from a single, brittle angle; we push from a **tripod** aligned with the layer’s meaningful coordinates.

---

## A concrete analogy

Imagine sliding a sofa across a room.

* **Dot‑product backprop** = one person shoving from wherever they happen to stand. It moves, but bumps into walls and spins.
* **MV‑DCB** = three friends at good handholds (edges the geometry already provides), coordinating a push **and a tiny pivot**. The path is straighter, and the first nudge already points roughly where you want.

---

## Why the **first step** lands “almost on the ball”

Early in training, models are most fragile. A single bad nudge can send logits to extremes, kill calibration, or stall rare classes. By **projecting the residual through a span that includes downstream‑aware directions**, the very first move:

* **Aligns with the class simplex** more faithfully (closer to a natural‑gradient step at the top).
* **Reduces flailing** in orthogonal, unhelpful directions.
* **Keeps confidence calibrated** from the start (we observed ECE drops relative to vanilla on multiple tasks).

Think of it as an **aimed first shove** that already respects the shape of the task.

---

## What we appear to gain (from the experiments)
Note- experiments were all done by human and variety tested.

* **Comparable accuracy** to vanilla on easy/mid tasks, sometimes a small trade (±1–2%),
  but **better calibration** (lower ECE) and **steadier early training**.
* **Tail behavior:** on imbalanced data, recall on the rarest class improved in some runs (projection keeps gradients from collapsing onto majority modes).
* **Decision boundaries** that are smoother and less noisy without extra regularizers.
* **Drop‑in usability:** works with Adam/SGD/Muon/Lion—optimizers are unchanged.

---

## Cost and practicality

* Overhead is **small**: extra work is building a K×K Gram and solving a tiny system per batch item and layer. With **K=2–4** this is typically **<5–10%** of step time, dwarfed by the big GEMMs.
* Memory: adds `B×K×H` temporaries, negligible for small K.
* Numerics: add a tiny ridge `λ` (1e‑6…1e‑4). If a batch basis is degenerate, project only onto the non‑degenerate part.

---

## When it helps most

* **Early training** (first steps, cold start): less erratic loss/accuracy, better initial calibration.
* **Imbalanced / tail‑heavy** regimes: prevents gradients from falling into majority‑class grooves.
* **Noisy or periodic geometry**: projection filters unhelpful directions.

When tasks are trivial and well‑conditioned, MV‑DCB typically matches vanilla (and sometimes trades a point of accuracy for noticeably lower ECE).

---

## Failure modes & dials

* Too large **K** → overfitting the span; try K=2–3 first.
* **λ** too small → singular Gram; too big → over‑smoothed updates. Start at 1e‑5.
* Blend **(α,β)**: α=1.0, β=0.3 is a good default; tuning β manages how much orthogonal exploration you keep.

---

## Why this can be a big upgrade

Backprop’s single-direction push is a historical convenience, not a law of nature. If everything in a network is a **coordinate on a manifold**, the right move is rarely along a single axis. MV‑DCB gives each layer a **tiny, data‑driven coordinate frame** and chooses its update **within that frame**.

You keep your model, your optimizer, your training loop. You gain:

* **Better aimed updates** (especially at the start),
* **Improved calibration without extra losses**,
* **More respectful handling of rare signals**,
* **Minimal engineering lift** and low runtime overhead.

code demo in the corresponding.py file
