# Literature Survey — Synapse

**A vertically-integrated local inference layer for safe, fast on-device agents**

*Survey supporting the Synapse design (see `PLAN.md`). Scope: the four technical
pillars Synapse builds on — (A) inference runtimes & memory management,
(B) KV-cache reuse / prefix caching, (C) constrained / grammar-guided decoding,
(D) speculative decoding — plus (E) the security framing that motivates fusing
safety policy into the decoder. Compiled June 2026.*

---

## 1. Motivation & scope

NORA serves local models through a *generic* engine (Ollama → llama.cpp). A generic
engine is blind to the system it serves: it does not know that NeuroSym validates
every action plan, that NORA emits a fixed structured command schema, that the
system prompt is identical every turn, or that the user's command history is highly
repetitive. Synapse's hypothesis is that a **vertically-integrated** engine can
exploit exactly that knowledge to be *faster and safer at once*.

This survey reviews the literature underpinning that hypothesis. For each theme we
trace the **foundational work**, the **current state of the art (2024–2026)**, and
its **relevance to a Synapse pillar**, and we close (§7) with a gap analysis: what
the literature does *not* yet combine, and where Synapse's contribution sits.

## 2. Method

Built with the ad-hoc `skills/research/SKILL.md` loop (frame → decompose → gather →
cross-check → distill). Sources were prioritized primary-docs / maintainer-repos /
peer-reviewed papers (SOSP, ICML, NeurIPS, EMNLP, NAACL) over secondary blogs;
load-bearing numbers (µs/token, ×-speedup) were taken from primary sources where
possible. Recency baseline: mid-2026.

---

## 3. Theme A — Inference runtimes & memory management

**Foundational.** *PagedAttention* (Kwon et al., SOSP 2023, arXiv:2309.06180)
reframed KV-cache memory after OS virtual-memory paging: fixed-size blocks, a
per-sequence block table, and a global free pool, giving near-zero KV waste and —
crucially for Synapse — **flexible KV sharing within and across requests**. It
powers vLLM, reported at 2–4× throughput over FasterTransformer/Orca at equal
latency. This is the conceptual ancestor of both prefix caching (Theme B) and
efficient batched serving.

**State of the art / Rust ecosystem.** Synapse targets a Rust base to make the
"altitude switch" (systems depth) real:
- *candle* (Hugging Face) — minimalist Rust ML framework, CPU/GPU, the substrate.
- *mistral.rs* (Buehler) — pure-Rust engine on candle; ships speculative decoding,
  In-Situ Quantization (ISQ), PagedAttention, continuous batching; targets
  CUDA / Apple Metal / CPU incl. low-end devices; v0.8.2 added CUDA graphs and
  FlashInfer paged kernels. → Synapse reuses this for the forward pass/quant rather
  than re-implementing kernels.

**Relevance.** Establishes the substrate decision (build *on* mistral.rs/candle)
and the memory model (paged KV blocks) that the Prefix Vault and Echo Drafter
pillars sit on top of.

---

## 4. Theme B — KV-cache reuse & prefix caching (→ *Prefix Vault*)

**Foundational.** PagedAttention's cross-request KV sharing (above) is the
enabling primitive. *RadixAttention*, introduced with *SGLang* (Zheng et al.,
NeurIPS 2024, arXiv:2312.07104), made reuse **automatic**: KV for prompts *and*
generations is retained in a **radix tree** supporting efficient prefix search,
insertion, and LRU eviction — reuse across many generation calls, reported up to
6.4× throughput. SGLang also contributes *compressed finite-state machines* for
faster structured-output decoding, an early bridge between Themes B and C.

**State of the art.** Production engines now ship **automatic prefix caching**:
vLLM uses hash-based block matching (`get_computed_blocks` over hashed prompt
tokens); SGLang uses the radix-tree LRU. Both target the shared-prefix pattern —
identical system prompts, guardrail preambles, shared documents — exactly NORA's
situation. *KVFlow* (arXiv:2507.07400) extends prefix caching to **multi-agent
workflows**, the most direct analogue to a NORA + NeuroSym pipeline that re-issues
overlapping context across cooperating components. Recent work also explores
cross-model KV reuse (e.g., Activated-LoRA multi-adapter serving,
arXiv:2512.17910). Open practical issue across all: finite GPU memory forces
eviction / memory-tiering policy.

**Relevance.** Directly motivates the **Prefix Vault**: a radix-tree KV cache keyed
on the fixed NORA system + NeuroSym preamble, reused across turns, with LRU
eviction under a memory cap. The novelty for Synapse is not the data structure but
*what* it caches — a known, fixed, multi-component agent preamble.

---

## 5. Theme C — Constrained / grammar-guided decoding (→ *Grammar Gate*, the headline)

**Foundational.** *Grammar-Constrained Decoding for Structured NLP Tasks without
Finetuning* (Geng, Josifoski, Peyrard, West; EMNLP 2023, arXiv:2305.13971)
established that a formal grammar can describe the output space of a wide range of
tasks, and that grammar-constrained LMs can **match or beat task-specific
finetuned models** off-the-shelf. In parallel, *Efficient Guided Generation for
Large Language Models* (Willard & Louf, 2023, arXiv:2307.09702 — the *Outlines*
library) compiled schemas/regex into finite-state machines giving **O(1) valid-token
lookup per step**, making constraint enforcement cheap enough for production.

**State of the art.** The field has converged on fast, low-overhead constraint
engines:
- *llguidance* (Microsoft/guidance-ai) — Rust Earley parser, ~50 µs/token,
  negligible startup; OpenAI publicly credited it for Structured Outputs (May 2025).
- *XGrammar* (arXiv:2411.15100) — <40 µs/token, up to 14× faster on JSON / 80× on
  complex CFGs; the default structured-generation backend for vLLM, SGLang, and
  TensorRT-LLM as of Mar 2026. *XGrammar-2* (arXiv:2601.04426) targets dynamic
  structured generation for **agentic** LLMs specifically.
- *GBNF* (llama.cpp) — practical context-free grammar format (beyond JSON; suits a
  command DSL).
- Quality/efficiency refinements: *Grammar-Aligned Decoding* (Park et al., NeurIPS
  2024) corrects the distributional distortion constraints can introduce;
  *Guiding LLMs the Right Way / DOMINO* (arXiv:2403.06988) gives fast, non-invasive
  constrained generation using pre-computation + speculation; *Flexible and
  Efficient Grammar-Constrained Decoding* (arXiv:2502.05111); *AdapTrack*
  (arXiv:2510.17376) constrains without distorting output intent.

**Relevance.** This is Synapse's **Grammar Gate** and the mechanism that *fuses all
three repos*: compile NeuroSym command schemas / action policies into a grammar and
mask logits so the model can only emit valid, in-policy structures — guaranteed
valid output (no parse errors, no retries) that is *also* safety-checked by
construction. The literature establishes both that this is effective (Geng) and
that it is now cheap enough to be on the hot path (XGrammar/llguidance µs/token).

---

## 6. Theme D — Speculative decoding (→ *Echo Drafter*)

**Foundational.** *Fast Inference from Transformers via Speculative Decoding*
(Leviathan et al., ICML 2023, arXiv:2211.17192) and *Accelerating LLM Decoding with
Speculative Sampling* (Chen et al., 2023, arXiv:2302.01318) established the
**lossless** paradigm: a cheap draft proposes several tokens; the target verifies
them in one parallel forward pass and accepts a prefix — 2–3× latency reduction
with *identical* output distribution. The key economic insight (restated often
since): decode is **memory-bound**, so verifying many tokens in one pass is nearly
free — speculation is arbitrage on that.

**Model-based drafting.** *Medusa* (Cai et al., 2024, arXiv:2401.10774) adds
multiple MLP decoding heads to the base model (no separate draft model), ~3.6×.
*EAGLE* (Li et al., ICML 2024, arXiv:2401.15077) drafts at the *feature* level with
a single transformer layer + LM head; *EAGLE-2* (arXiv:2406.16858) adds dynamic
draft trees; *EAGLE-3* (2025) pushes SOTA via training-time test. These give the
best raw latency but require training/extra weights.

**Training-free drafting (Synapse's path).** A lighter family drafts by **retrieval
from recent text**: n-gram / prompt-lookup decoding builds drafts from a dictionary
or **suffix automaton / suffix tree** of recently generated tokens — no training,
ideal for repetitive workloads. *Cacheback* (arXiv:2511.21699, Nov 2025) speculates
from cache alone; *Goose* (arXiv:2604.02047) builds anisotropic speculation trees
training-free. A crucial empirical finding for our design: across large benchmarks,
end-to-end speedup correlates far more with the **draft's latency** than its
language-modeling accuracy (see also *Scaling Up, Speeding Up*, arXiv:2509.04474;
*Decoding Speculative Decoding*, NAACL 2025) — i.e., a cheap, often-right draft
beats an expensive, usually-right one. Spec-decode now interacts with quantization
(arXiv:2505.22179) and is a production standard (vLLM/TensorRT-LLM native;
2–3× latency, NVIDIA-reported 3.6× throughput on H200, late 2025).

**Relevance.** The **Echo Drafter**: build a suffix-automaton / n-gram draft store
from NORA's own ChromaDB command history and draft from it (cheap, training-free,
lossless). The literature both validates the mechanism (prompt-lookup / suffix
drafting) and tells us *why it should work here specifically* — NORA's command
traffic is highly repetitive, so a near-zero-cost retrieval draft hits the
"draft-latency-dominates" sweet spot.

---

## 7. Theme E — Structured-output safety (why fuse policy into the decoder)

Constrained decoding is usually framed as a *reliability* technique, but it is also
a *security control*. *Beyond Prompts: Space-Time Decoupling Control-Plane
Jailbreaks in LLM Structured Output* (arXiv:2503.24191) shows structured-output
machinery can itself be an attack surface — relevant because Synapse is *putting a
safety system (NeuroSym) on the decoding hot path* and must not introduce a new
control-plane vulnerability. *Draft-Conditioned Constrained Decoding*
(arXiv:2603.03305) is notable as an early sign that constrained decoding and
speculative decoding are beginning to be **co-designed** — precisely the
intersection Synapse occupies.

**Relevance.** Frames the Grammar Gate as safety-by-construction *and* flags the
threat model to respect (don't let the grammar/policy compiler become an injection
vector).

---

## 8. Synthesis & gap analysis

**What the literature already gives us, separately:**
- Fast paged KV memory and **automatic prefix caching** (PagedAttention, RadixAttention, vLLM/SGLang APC, KVFlow).
- **Cheap, production-grade grammar constraints** (Geng; Outlines; XGrammar; llguidance) — now <40–50 µs/token.
- **Lossless, training-free speculative decoding** from retrieval/suffix structures (prompt-lookup; Cacheback; Goose).
- A Rust runtime (mistral.rs/candle) that already implements the heavy primitives.

**The gap Synapse fills.** Each technique above is studied and shipped *in
isolation* and *for generic serving*. The literature does **not** yet present a
single **local, on-device** engine that is **co-designed with the agent and its
safety system**, specifically:
1. **Policy-as-grammar fusion** — compiling a *safety system's* (NeuroSym's) action
   policies into the decoding constraint, so "valid" and "in-policy" are the *same*
   guarantee (Theme C + E), rather than treating schema-validity and safety as two
   stages. The co-design trend (arXiv:2603.03305) is nascent; the *safety*-policy
   variant is open.
2. **Behavioral-memory drafting** — seeding the speculative draft from the agent's
   *own episodic command history* (NORA's ChromaDB) rather than a generic draft
   model or only the current context window (Theme D). The "draft-latency-dominates"
   result predicts this should pay off on repetitive agent traffic, but it is
   untested for a persistent personal-assistant corpus.
3. **Agent-aware prefix reuse** — Prefix Vault caches a *known, fixed,
   multi-component* preamble (system + guardrail), a narrower and more exploitable
   case than generic APC; KVFlow points at multi-agent reuse but not the
   single-user local-assistant setting.
4. **Vertical integration on-device** — combining all of the above in one *local*
   engine where the latency/safety wins compound, as opposed to cloud serving where
   most of this literature lives.

**Positioning.** Synapse is therefore less a novel *algorithm* and more a novel
*system*: an integration thesis — that co-locating safety policy, agent memory, and
a fixed agent preamble inside the decoder yields multiplicative gains a generic
engine cannot reach. The contribution is empirical (a working local engine + a
benchmark vs. stock Ollama on NORA's real command set) and architectural, which is
exactly the right shape for the portfolio goal in `PLAN.md`.

---

## 9. References

**Runtimes & memory**
- Kwon et al. *Efficient Memory Management for LLM Serving with PagedAttention.* SOSP 2023. https://arxiv.org/abs/2309.06180
- vLLM project. https://github.com/vllm-project/vllm · Automatic Prefix Caching docs: https://docs.vllm.ai/en/stable/design/v1/prefix_caching.html
- mistral.rs. https://github.com/EricLBuehler/mistral.rs
- candle. https://github.com/huggingface/candle

**KV reuse / prefix caching**
- Zheng et al. *SGLang: Efficient Execution of Structured LM Programs* (RadixAttention). NeurIPS 2024. https://arxiv.org/abs/2312.07104
- *KVFlow: Efficient Prefix Caching for LLM-Based Multi-Agent Workflows.* https://arxiv.org/html/2507.07400v1
- *Prefix Caching* (handbook). https://bentoml.com/llm/inference-optimization/prefix-caching
- Cross-Model KV-Cache Reuse with Activated LoRA. https://arxiv.org/pdf/2512.17910

**Constrained / grammar-guided decoding**
- Geng et al. *Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning.* EMNLP 2023. https://arxiv.org/abs/2305.13971
- Willard & Louf. *Efficient Guided Generation for LLMs* (Outlines). 2023. https://arxiv.org/abs/2307.09702
- Park et al. *Grammar-Aligned Decoding.* NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/file/2bdc2267c3d7d01523e2e17ac0a754f3-Paper-Conference.pdf
- *Guiding LLMs the Right Way: Fast, Non-Invasive Constrained Generation* (DOMINO). https://arxiv.org/html/2403.06988v1
- *XGrammar: Flexible and Efficient Structured Generation.* https://arxiv.org/pdf/2411.15100 · *XGrammar-2 (Agentic).* https://arxiv.org/pdf/2601.04426
- *Flexible and Efficient Grammar-Constrained Decoding.* https://arxiv.org/pdf/2502.05111
- *AdapTrack: Constrained Decoding without Distorting Output Intent.* https://arxiv.org/pdf/2510.17376
- llguidance. https://github.com/guidance-ai/llguidance
- Constrained-decoding overviews: https://zeroentropy.dev/concepts/constrained-decoding/ · https://tianpan.co/blog/2026-04-16-grammar-constrained-generation-output-reliability

**Speculative decoding**
- Leviathan et al. *Fast Inference from Transformers via Speculative Decoding.* ICML 2023. https://arxiv.org/abs/2211.17192
- Chen et al. *Accelerating LLM Decoding with Speculative Sampling.* 2023. https://arxiv.org/abs/2302.01318
- Cai et al. *Medusa.* 2024. https://arxiv.org/abs/2401.10774
- Li et al. *EAGLE.* ICML 2024. https://arxiv.org/abs/2401.15077 · *EAGLE-2.* https://arxiv.org/html/2406.16858v1
- *Cacheback: Speculative Decoding With Nothing But Cache.* 2025. https://arxiv.org/pdf/2511.21699
- *Goose: Anisotropic Speculation Trees for Training-Free Speculative Decoding.* https://arxiv.org/pdf/2604.02047
- *Scaling Up, Speeding Up: A Benchmark of Speculative Decoding.* https://arxiv.org/pdf/2509.04474
- *Decoding Speculative Decoding.* NAACL 2025. https://aclanthology.org/2025.naacl-long.328.pdf
- *Speculative Decoding Meets Quantization.* https://arxiv.org/pdf/2505.22179
- vLLM speculative decoding docs. https://docs.vllm.ai/en/latest/features/spec_decode/

**Structured-output safety / co-design**
- *Beyond Prompts: Space-Time Decoupling Control-Plane Jailbreaks in LLM Structured Output.* https://arxiv.org/pdf/2503.24191
- *Draft-Conditioned Constrained Decoding for Structured Generation.* https://arxiv.org/pdf/2603.03305

---

*Note on citations: arXiv IDs and venues were taken from search results and
canonical records; a few well-known identifiers (Outlines 2307.09702, Medusa
2401.10774, Chen 2302.01318) are cited from established memory and should be
spot-verified before formal publication. Items dated 2026 reflect the mid-2026
research baseline of this survey.*
