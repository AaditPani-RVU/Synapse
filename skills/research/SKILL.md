---
name: research
description: Lightweight technical research method — scope a question, gather current sources from the web, cross-check claims, and distill into a decision-ready brief with cited evidence. Use before writing a plan, choosing a stack, or making an architecture call.
metadata:
  type: reference
---

# research

A fast, repeatable loop for turning an open technical question into a grounded,
cited brief you can act on. Drafted ad hoc (no research skill was installed) and
used to produce `PLAN.md`.

## When to use
- Before committing to a stack, library, or architecture.
- When "what's the current state of X?" matters and your training data may be stale.
- When a plan's credibility depends on real, citable evidence.

## The loop

1. **Frame** — Write the decision the research must serve in one sentence.
   Bad: "research inference engines." Good: "decide whether to build Synapse on
   a Rust runtime (mistral.rs/candle) vs Python (llama.cpp bindings)."

2. **Decompose** — Break it into 3–6 concrete sub-questions. Each should be
   answerable by a focused search and should change the decision if the answer flips.

3. **Gather (parallel)** — Run web searches per sub-question. Prefer:
   primary docs > maintainer repos/issues > recent papers (arXiv) > engineering
   blogs > listicles. Note publication dates; current month is the baseline for
   "recent." Run independent searches in the same batch.

4. **Cross-check** — A claim counts as "established" only if a primary/maintainer
   source or two independent sources agree. Flag single-source or vendor-marketing
   claims as tentative. Capture exact numbers (µs/token, x-speedup, memory) — they
   are the load-bearing evidence.

5. **Distill** — For each sub-question: the answer, the evidence (with numbers),
   the source link, and a confidence tag (solid / tentative). Surface
   disagreements rather than averaging them away.

6. **Decide** — State the recommendation the evidence supports, the key risk, and
   what would change your mind. Carry citations into the downstream artifact.

## Output contract
- Every non-obvious claim carries a source link.
- Numbers are quoted, not paraphrased ("~50 µs/token", not "fast").
- Tentative vs solid is explicit.
- Ends with a recommendation, not a survey.

## Anti-patterns
- Researching breadth-first with no decision in mind (infinite scroll).
- Trusting a single blog/listicle for a load-bearing number.
- Letting the brief become a link dump instead of a decision.
