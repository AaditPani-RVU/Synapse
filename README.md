<div align="center">

# ⚡ Synapse

### The inference *nervous system* for local AI agents

*Make on-device models run **faster** and **safer** — at the same time —
by fusing the agent and its safety layer directly into the decoder.*

<br/>

[![Rust](https://img.shields.io/badge/Rust-000000?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![Status](https://img.shields.io/badge/status-Phase%200%20%C2%B7%20engine%20wired-blue?style=for-the-badge)](./PLAN.md)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900?style=for-the-badge&logo=nvidia&logoColor=white)]()
[![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)](#-license)
[![Local first](https://img.shields.io/badge/100%25-local%20%C2%B7%20offline-orange?style=for-the-badge)]()

<br/>

**[Brain 🧠 NeuroSym-AI](https://github.com/AaditPani-RVU/NeuroSym-AI)** &nbsp;•&nbsp;
**[Body 🤖 N.O.R.A](https://github.com/AaditPani-RVU/N.O.R.A)** &nbsp;•&nbsp;
**⚡ Synapse** *(you are here)*

</div>

---

## 💡 The idea

A generic engine (Ollama, llama.cpp) is **blind to the system it serves**. It doesn't
know that [N.O.R.A](https://github.com/AaditPani-RVU/N.O.R.A) turns every utterance into
a structured command, that [NeuroSym-AI](https://github.com/AaditPani-RVU/NeuroSym-AI)
validates every action, that the system prompt is identical on every turn, or that the
user's command history is wildly repetitive.

**Synapse is not a faster matmul.** It will never out-kernel llama.cpp — and it doesn't
try to. It wins on optimizations that are *only possible* when the engine knows exactly
who it's serving:

<table>
<tr>
<th align="left">Pillar</th>
<th align="left">What it does</th>
<th align="left">Why it wins</th>
</tr>
<tr>
<td><b>🛡️ Grammar Gate</b><br/><sub>the headline</sub></td>
<td>Compiles NeuroSym policies into a decoding grammar and masks logits so the model can <i>only</i> emit valid, in-policy output.</td>
<td>Safer <b>and</b> faster — malformed/unsafe output is <i>structurally impossible</i>. Zero retries.</td>
</tr>
<tr>
<td><b>🗄️ Prefix Vault</b></td>
<td>Radix-tree KV-cache reuse of the fixed system + guardrail preamble across turns.</td>
<td>Lower latency / time-to-first-token — stop recomputing the same 1–2k tokens every call.</td>
</tr>
<tr>
<td><b>🔮 Echo Drafter</b></td>
<td>Speculative decoding drafted from a suffix-automaton over NORA's own command history.</td>
<td>Higher throughput on repetitive commands. Lossless — output is provably unchanged.</td>
</tr>
<tr>
<td><b>🔀 Router</b></td>
<td>A tiny classifier sends trivial intents to a small model or a cached answer.</td>
<td>Most commands never touch the big model at all.</td>
</tr>
</table>

> The headline is **Grammar Gate**: it's the single mechanism that literally *fuses all
> three projects* — NeuroSym's safety policies become decoding constraints inside NORA's
> engine. *Valid* and *in-policy* become the same guarantee.

---

## 🏗️ Architecture

```
                       ┌───────────────────────────────────────────────┐
   N.O.R.A  ──────────▶│   Synapse   (Rust · Ollama-compatible API)     │
  one config line      │                                               │
                       │   Router ─▶ Prefix Vault ─▶ Echo Drafter       │
                       │                  │                  │          │
                       │            Grammar Gate (logits mask)│          │
                       │                  └────────┬──────────┘          │
                       │              mistral.rs / candle               │
                       │         (forward pass · quant · paged KV)      │
                       └───────────────────────────────────────────────┘
                              ▲                              ▲
                     NeuroSym policies               NORA command history
                     → compiled grammars             → suffix-automaton drafts
```

**Reused vs. hand-built** (and we keep it honest): the model forward pass, quantization,
and paged attention come from [`mistral.rs`](https://github.com/EricLBuehler/mistral.rs)
/ [`candle`](https://github.com/huggingface/candle). The **server, the four pillars, and
the benchmark harness** are hand-built — that's where the substance is.

---

## 🚦 Status &nbsp;·&nbsp; Roadmap

| Phase | Focus | State |
|:---:|:---|:---:|
| **0** | **Foundation** — Ollama-compatible server, real `mistral.rs` engine (GGUF + HF/ISQ, CUDA) | 🟢 **inference live** |
| **1** | **Grammar Gate** — constrained decoding from NeuroSym policies | ⚪ next |
| **2** | **Prefix Vault** — KV-cache prefix reuse | ⚪ planned |
| **3** | **Echo Drafter** — behavioral speculative decoding | ⚪ planned |
| **4** | **Router** — intent-based model routing | ⚪ planned |
| **5** | **Benchmark & demo** — A/B vs stock Ollama, write-up | ⚪ planned |

<sub>Phase 0 today: an Ollama-compatible server streaming real tokens from
`mistral.rs` (GGUF or HuggingFace+ISQ, CUDA-accelerated) — the four pillars are
declared as seams (`src/pillars/`) and land one phase at a time starting with
Grammar Gate. Full plan → **[`PLAN.md`](./PLAN.md)** · research grounding →
**[`LITERATURE_SURVEY.md`](./LITERATURE_SURVEY.md)**.</sub>

---

## 🚀 Quickstart

```sh
# no model configured -> stub engine, just proves the wiring
cargo run                                    # serves http://127.0.0.1:11435

# point at a local GGUF file -> real mistral.rs inference (CUDA if available)
SYNAPSE_GGUF_DIR=/models/qwen2.5-1.5b-instruct \
SYNAPSE_MODEL=qwen2.5-1.5b \
cargo run --release
```

```sh
# discover the model (Ollama /api/tags)
curl http://127.0.0.1:11435/api/tags

# chat — streaming NDJSON, exactly like Ollama
curl http://127.0.0.1:11435/api/chat -d '{
  "model": "qwen2.5-1.5b",
  "messages": [{ "role": "user", "content": "hello synapse" }]
}'
```

<details>
<summary><b>Configuration</b></summary>

| Env var | Default | Notes |
|---|---|---|
| `SYNAPSE_HOST` | `127.0.0.1` | bind address |
| `SYNAPSE_PORT` | `11435` | beside Ollama's `11434`, for honest A/B benchmarking |
| `SYNAPSE_MODEL` | `synapse-stub` | advertised model name |
| `SYNAPSE_GGUF_DIR` | *(unset)* | dir with a local GGUF file → loads the real engine |
| `SYNAPSE_GGUF_FILE` | `model.gguf` | filename inside `SYNAPSE_GGUF_DIR` |
| `SYNAPSE_HF_MODEL` | *(unset)* | HuggingFace model ID, downloaded + ISQ-quantized (used when `SYNAPSE_GGUF_DIR` is unset) |
| `SYNAPSE_FORCE_CPU` | *(unset)* | force CPU inference — set this if your GPU's compute capability is <8.0 and hits F16/BF16 NaNs on CUDA |

If neither `SYNAPSE_GGUF_DIR` nor `SYNAPSE_HF_MODEL` is set, Synapse falls back to
the `StubEngine` (echoes input) so the server + N.O.R.A wiring can be tested without
a GPU or a model download.

</details>

### Point N.O.R.A at it

Set NORA's Ollama provider base URL to **`http://127.0.0.1:11435`**. That's the whole change.

---

## 🗂️ Layout

```
src/
├── main.rs           server bootstrap, engine selection (stub / GGUF / HF)
├── config.rs         env-based config
├── api.rs            Ollama-compatible endpoints (/api/chat, /api/tags, …)
├── engine.rs         Engine trait (the seam) + StubEngine
├── mistral_engine.rs MistralEngine — real inference via mistral.rs (GGUF + HF/ISQ)
└── pillars/
    ├── grammar_gate.rs   🛡️ Pillar 1 — constrained decoding
    ├── prefix_vault.rs   🗄️ Pillar 2 — KV prefix reuse
    ├── echo_drafter.rs   🔮 Pillar 3 — speculative decoding
    └── router.rs         🔀 Pillar 4 — model routing
```

---

## 🧩 The trilogy

Synapse is the third piece of a vertically-integrated, fully-local agentic stack:

| | Project | Role |
|:---:|:---|:---|
| 🧠 | **[NeuroSym-AI](https://github.com/AaditPani-RVU/NeuroSym-AI)** | Neuro-symbolic safety guardrails — the brain |
| 🤖 | **[N.O.R.A](https://github.com/AaditPani-RVU/N.O.R.A)** | Local voice agent with episodic memory — the body |
| ⚡ | **Synapse** | The inference layer that connects and accelerates them |

---

## 🔧 Toolchain notes

Builds with stable Rust on `x86_64-unknown-linux-gnu`. Dev machine is a **4 GB GTX
1650 (sm_75) + CUDA 12.4** — that compute capability predates hardware BF16, so
`mistral.rs`'s `Auto` dtype selection can silently pick an F16 CUDA path that
overflows into NaN logits. Synapse works around this by driving `mistral.rs`'s
lower-level loader directly with an explicit dtype (see `mistral_engine.rs`), and
exposes `SYNAPSE_FORCE_CPU` as a real fallback lever, not dead code. Dev target is a
**~1.5B int4/GGUF** model on a 4 GB GPU, with CPU fallback for larger models.

---

## 📄 License

[MIT](LICENSE) © Aadit Pani

<div align="center"><sub>Built as a flagship for trustworthy, fast, fully-local AI.</sub></div>
