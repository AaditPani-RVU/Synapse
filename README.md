# Synapse

> The inference nervous system between **NeuroSym-AI** (safety brain) and
> **N.O.R.A** (agent body) — a vertically-integrated **local** inference layer that
> makes on-device models run **faster *and* safer at the same time**.

Synapse is *not* a faster matmul. It does not try to out-kernel llama.cpp. It wins
on optimizations a *generic* engine structurally cannot do, because it knows it is
serving NORA + NeuroSym:

| Pillar | Idea | Win |
| --- | --- | --- |
| **Grammar Gate** | NeuroSym policies → decoding grammar; mask logits | Valid & in-policy *by construction* — no retries, safer |
| **Prefix Vault** | Radix-tree KV reuse of the fixed system+guardrail preamble | Lower latency / TTFT |
| **Echo Drafter** | Speculative decoding drafted from NORA's command history | Higher throughput, lossless |
| **Router** | Tiny classifier → small model / cache for trivial intents | Most commands skip the big model |

See [`PLAN.md`](./PLAN.md) for the full design and phased plan, and
[`LITERATURE_SURVEY.md`](./LITERATURE_SURVEY.md) for the research grounding.

## Status

**Phase 0 — Foundation.** A minimal, **Ollama-compatible** server backed by a stub
engine. Goal: prove the wiring (NORA connects with a one-line base-URL change) and
establish the benchmark baseline. The real model runtime (mistral.rs/candle) and the
four pillars land in later phases.

## Run

```sh
cargo run            # starts on http://127.0.0.1:11435
```

Config via env: `SYNAPSE_HOST`, `SYNAPSE_PORT` (default 11435 — beside Ollama's
11434 for A/B benchmarking), `SYNAPSE_MODEL`.

## Try it

```sh
# discover the advertised model (Ollama /api/tags)
curl http://127.0.0.1:11435/api/tags

# chat (Ollama /api/chat, streaming NDJSON)
curl http://127.0.0.1:11435/api/chat -d '{
  "model": "synapse-stub",
  "messages": [{ "role": "user", "content": "hello" }]
}'
```

## Point N.O.R.A at it

Set NORA's Ollama provider base URL to `http://127.0.0.1:11435`. No other change.

## Layout

```
src/
  main.rs              server bootstrap
  config.rs            env-based config
  api.rs               Ollama-compatible endpoints (/api/chat, /api/tags, ...)
  engine.rs            Engine trait (the seam) + Phase 0 StubEngine
  pillars/
    grammar_gate.rs    Pillar 1 (Phase 1) — constrained decoding
    prefix_vault.rs    Pillar 2 (Phase 2) — KV prefix reuse
    echo_drafter.rs    Pillar 3 (Phase 3) — speculative decoding
    router.rs          Pillar 4 (Phase 4) — model routing
```

## Toolchain

Phase 0 builds with the Rust **GNU** toolchain (self-contained linker — no Visual
Studio required). When the mistral.rs CUDA backend lands we switch
`rust-toolchain.toml` to MSVC, which CUDA on Windows requires.

## License

MIT
