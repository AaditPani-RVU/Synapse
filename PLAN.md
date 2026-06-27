# Synapse — Plan

> The synapse between the brain and the body.
> **NeuroSym-AI** is the safety brain. **N.O.R.A** is the agent body.
> **Synapse** is the nervous system that connects them — a local inference layer
> that makes their on-device models run *faster* and *safer at the same time*,
> by exploiting things only a vertically-integrated engine can know.

---

## 0. The thesis (read this first)

NORA's local path today is Ollama (llama.cpp under the hood) + faster-whisper. A
generic engine like Ollama is *blind* to the system it serves — it doesn't know
NORA parses every utterance into a structured command, doesn't know NeuroSym
validates every action plan, doesn't know the user's command history, doesn't
know the system prompt is identical on every turn.

**Synapse is not a faster matmul.** We will *not* try to out-kernel llama.cpp — we
would lose, and a slower re-implementation is worse than nothing. Instead Synapse
wins on **integration-driven optimizations** a generic engine structurally cannot do:

| Pillar | What it does | Win |
| --- | --- | --- |
| **Grammar Gate** | Compiles NeuroSym command schemas / policies into a grammar and constrains decoding so the model can *only* emit valid, in-policy structures. | Safer **and** faster — invalid output is impossible, zero reparse/retry loops. |
| **Prefix Vault** | Radix-tree KV-cache reuse of the fixed NORA system prompt + NeuroSym guardrail preamble across turns. | Lower latency / time-to-first-token — stop recomputing the same 1–2k tokens every call. |
| **Echo Drafter** | Training-free speculative decoding whose draft comes from a suffix-automaton / n-gram store built from NORA's own ChromaDB command history. | Higher throughput on the repetitive command traffic NORA actually sees. Lossless. |
| **Router** | A tiny classifier sends trivial intents to a small model (or a cached answer), hard ones to the big model. | Most commands never touch the big model at all. |

The headline is **Grammar Gate**: it is the single mechanism that literally fuses
all three repos — NeuroSym's policies become decoding constraints inside NORA's
engine. "Valid and in-policy by construction" is both the safety story and the
speed story in one move.

---

## 1. Why this is the right flagship

- **Portfolio goal = career.** This proves the one dimension NeuroSym and NORA
  don't: **low-level systems / performance depth.** Both existing repos are
  high-level Python orchestration; Synapse goes down the stack (Rust, KV-cache,
  logits processors, sampling loops, quantized runtimes).
- **Coherence, not fragmentation.** Three loosely-related repos become one story:
  *"I built a vertically-integrated local agentic system, top to bottom —
  safety brain, agent body, and the inference nervous system between them."*
  That one sentence outperforms any single repo.
- **On the 2026 frontier.** Inference cost, on-device serving, constrained
  decoding, and speculative decoding are exactly where the field's attention is.
- **Provable.** The deliverable is a benchmark table + a side-by-side demo video,
  not a subjective "trust me." Numbers travel.

---

## 2. Architecture

```
                    ┌──────────────────────────────────────────────┐
   N.O.R.A  ───────▶│  Synapse server  (Rust, OpenAI/Ollama-compat) │
 (one config line)  │                                              │
                    │   Router ──▶ Prefix Vault ──▶ Echo Drafter   │
                    │                │                    │        │
                    │          Grammar Gate (logits mask) │        │
                    │                └────────┬───────────┘        │
                    │                  mistral.rs / candle         │
                    │              (forward pass, ISQ quant,       │
                    │               PagedAttention, Metal/CUDA)    │
                    └──────────────────────────────────────────────┘
                              ▲                         ▲
                    NeuroSym policies            NORA ChromaDB
                    → compiled grammars          → suffix-automaton draft store
```

**Reuse vs. hand-built (be honest in the README):**
- **Reused:** model forward pass, quantization (ISQ), PagedAttention, GPU/Metal
  backends — via [mistral.rs](https://github.com/EricLBuehler/mistral.rs)
  (itself built on [candle](https://github.com/huggingface/candle)). No point
  re-writing kernels.
- **Hand-built (the portfolio's substance):** the OpenAI/Ollama-compatible
  server, the Grammar Gate (NeuroSym-policy → grammar compiler + logits
  processor), the Prefix Vault (radix-tree KV reuse), the Echo Drafter
  (suffix-automaton draft + verification loop), the Router, and the benchmark
  harness. These are genuine systems work — logits, KV cache, sampling — without
  the suicidal scope of a from-scratch kernel.

---

## 3. Stack decision

**Rust + mistral.rs/candle.** Rationale (from research, §7):
- mistral.rs already ships speculative decoding, PagedAttention, in-situ
  quantization, continuous batching, and targets CUDA / Apple Metal / CPU
  (even Raspberry Pi) — so we build *on* a fast base and spend our effort on the
  four novel pillars.
- Rust *is* the altitude switch we're buying. Python-on-llama.cpp would integrate
  faster but undercut the whole point (proving systems depth).
- Integration with Python NORA is clean: Synapse exposes an HTTP API; NORA's
  provider config points at it. No FFI required.

**Target model (v0):** a small instruct model, int4-quantized, that runs on the
dev box (Windows + NVIDIA, since NORA is Windows-first) — e.g. a ~3B-class
instruct model. Keep it modest so it runs locally and benchmarks are honest.

For constrained decoding, evaluate **llguidance** (~50 µs/token, Rust Earley
parser, OpenAI-credited) and **XGrammar** (<40 µs/token, default backend for
vLLM/SGLang/TensorRT-LLM as of Mar 2026) — both are Rust-friendly. GBNF is the
fallback grammar format.

---

## 4. Phased build plan

Each phase ends with a **measurable** result so we fail fast before over-investing.

### Phase 0 — Foundation (proves integration)
- [x] Rust server exposing an Ollama-compatible API (`/api/chat` streaming NDJSON,
  `/api/tags`, `/api/version`) — verified end to end against a stub engine.
- [x] `Engine` trait seam + four pillar module stubs in place.
- [ ] Load the target quantized model (~1.5B int4) via mistral.rs; tokens end-to-end.
- [ ] Point NORA at Synapse (`http://127.0.0.1:11435`) with a one-line config change.
- **Exit check:** NORA runs a real command through Synapse. Record baseline
  tokens/sec, time-to-first-token (TTFT), end-to-end command latency, memory.
- **Env note:** toolchain is MSVC (GNU was a dead end — stale `C:\MinGW` shadows
  Rust's tools). VS C++ Build Tools 14.44 + CUDA 12.8 + nvcc installed. GPU is a
  GTX 1650 (4 GB) so the target model is ~1.5B int4; 31 GB RAM gives a CPU fallback.

### Phase 1 — Grammar Gate (the headline; do this early)
- Compiler: NeuroSym command schema / action-policy → grammar (start with JSON
  schema → grammar via llguidance/XGrammar; extend to a small command DSL).
- Logits processor that masks invalid tokens each step.
- **Exit check:** % malformed command outputs drops to **0**; NORA's reparse/retry
  rate drops; tokens-per-command drops. Demo: an unsafe/malformed command is now
  *structurally impossible to emit*.

### Phase 2 — Prefix Vault (latency)
- Radix-tree KV-cache keyed on token-prefix; reuse the fixed NORA system prompt +
  NeuroSym preamble across turns; LRU eviction + memory cap.
- **Exit check:** TTFT and end-to-end latency drop measurably on multi-turn
  sessions (the shared-prefix win).

### Phase 3 — Echo Drafter (throughput)
- Suffix-automaton / n-gram draft store built from NORA's ChromaDB command corpus;
  draft k tokens, verify in one target forward pass (lossless).
- **Exit check:** tokens/sec up on NORA's real command distribution; report draft
  acceptance rate. (Lossless: quality is provably unchanged.)

### Phase 4 — Router + polish
- Tiny intent classifier → route trivial commands to a small model or a cached
  answer; hard ones to the target model. Config, logging, graceful fallbacks.
- **Exit check:** measurable share of commands resolved without the big model.

### Phase 5 — Benchmark, write-up, demo
- Reproducible harness over NORA's real command set: Synapse vs stock Ollama.
- Metrics: tokens/sec, TTFT, end-to-end latency, % malformed (→0), retry rate, memory.
- A README benchmark table + a side-by-side screen recording (stock Ollama vs
  Synapse) + a short technical write-up / blog post.

**Fastest proof of the whole thesis (v0):** Phase 0 + a minimal Phase 1. NORA
running through Synapse, emitting structurally-valid commands via constrained
decoding, with a latency number next to it. That single demo proves "vertically
integrated, faster *and* safer" end to end. Everything after is amplitude.

---

## 5. What "done" / "great" looks like

- **Good:** Synapse serves NORA locally; Grammar Gate makes malformed commands
  impossible; at least one axis (latency *or* throughput *or* retry-elimination)
  is clearly better than stock Ollama, with reproducible numbers.
- **Great:** all four pillars live; a clean benchmark table where Synapse wins on
  end-to-end command latency *and* malformed-rate; a crisp demo video; a write-up
  that explains *why* a vertically-integrated engine beats a generic one.
- **The bar that matters:** competitive on at least one axis, with an honest
  explanation of where it wins and why. We don't need to win everywhere.

---

## 6. Risks & mitigations

| Risk | Mitigation |
| --- | --- |
| **Scope creep** (rewriting an engine). | Build *on* mistral.rs; only hand-write the four pillars + server + harness. |
| **Losing to llama.cpp on raw speed.** | Don't compete there. Compete on integration wins (constraints, prefix reuse, behavioral drafting, routing). |
| **Low draft acceptance** in Echo Drafter. | Lossless by design — worst case = no speedup, never wrong output. Report acceptance honestly. |
| **Windows + CUDA toolchain friction** (NORA is Windows-first). | mistral.rs supports CUDA on Windows; fall back to CPU/Metal dev path; document the build. |
| **Rust learning curve.** | mistral.rs/candle absorb the hardest parts; our code is server + logits/cache logic, learnable in the phase order above. |
| **"It's just glue."** | The substance is the hand-built pillars (logits processor, radix KV cache, suffix-automaton drafting). Keep reuse vs. hand-built explicit in the README. |

---

## 7. Research notes (current as of June 2026)

**Runtime base — mistral.rs / candle.** mistral.rs is a pure-Rust engine built on
candle; ships speculative decoding (draft model), In-Situ Quantization,
PagedAttention, continuous batching; targets CUDA / Apple Metal / CPU incl.
low-end devices; recent v0.8.2 added CUDA graphs and FlashInfer paged kernels.
→ Build on it; don't reinvent kernels.
[mistral.rs](https://github.com/EricLBuehler/mistral.rs) ·
[candle](https://github.com/huggingface/candle)

**Constrained decoding — the Grammar Gate basis.** Constraints are enforced
*during* generation, so output is guaranteed valid — no parse errors, no retries,
no fallback parser. **llguidance** (Microsoft): Rust Earley parser, ~50 µs/token,
credited by OpenAI for Structured Outputs (May 2025). **XGrammar**: <40 µs/token,
default structured-generation backend for vLLM/SGLang/TensorRT-LLM as of Mar 2026,
up to 14x faster on JSON / 80x on complex CFGs. **GBNF** (llama.cpp) expresses any
context-free grammar (useful for a command DSL, not just JSON).
[llguidance](https://github.com/guidance-ai/llguidance) ·
[constrained decoding overview](https://zeroentropy.dev/concepts/constrained-decoding/) ·
[grammar-constrained generation](https://tianpan.co/blog/2026-04-16-grammar-constrained-generation-output-reliability)

**Prefix caching — the Prefix Vault basis.** Cache KV blocks of processed
requests; reuse when a new request shares a prefix (system prompt / guardrail
preamble). **vLLM** = hash-based block matching; **SGLang** = RadixAttention
(radix-tree LRU of KV blocks). Massive win for repetitive multi-turn prompts;
needs eviction / memory tiering. KVFlow extends this to multi-agent workflows.
[vLLM automatic prefix caching](https://docs.vllm.ai/en/stable/design/v1/prefix_caching.html) ·
[prefix caching handbook](https://bentoml.com/llm/inference-optimization/prefix-caching) ·
[KVFlow (multi-agent)](https://arxiv.org/html/2507.07400v1)

**Speculative decoding — the Echo Drafter basis.** Lossless: a fast draft
predicts tokens, the target verifies in one parallel pass, accepting several at
once. Production standard as of Dec 2025 (2–3x latency cut; NVIDIA 3.6x throughput
on H200). **Training-free n-gram / prompt-lookup** methods draft by retrieving
recent token sequences via dictionary / **suffix automaton / suffix tree** — ideal
for NORA's repetitive command history. Note: draft *latency* matters more than
draft *accuracy* for end-to-end speedup. "Cacheback" (Nov 2025) shows speculation
from cache alone.
[speculative decoding guide 2025](https://introl.com/blog/speculative-decoding-llm-inference-speedup-guide-2025) ·
[vLLM spec decode](https://docs.vllm.ai/en/latest/features/spec_decode/) ·
[Cacheback](https://arxiv.org/pdf/2511.21699)

---

## 8. Immediate next steps

1. Confirm dev hardware (GPU/VRAM, OS) → pick the exact target model + quant.
2. Stand up the Rust project + mistral.rs; get one token out of the target model.
3. Build the Ollama-compatible endpoint; point NORA at it (Phase 0 exit check).
4. Wire llguidance/XGrammar + a NeuroSym schema → first Grammar Gate demo (v0).

---

*Research method used to build this plan: `skills/research/SKILL.md` (drafted for
this task — frame → decompose → gather → cross-check → distill → decide).*
