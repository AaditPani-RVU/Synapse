//! Runtime configuration. Phase 0 reads a few env vars with sane defaults so
//! N.O.R.A can point at Synapse with zero code changes.

use std::env;

#[derive(Debug, Clone)]
pub struct Config {
    pub host: String,
    pub port: u16,
    /// Model name Synapse advertises to Ollama-compatible clients.
    pub model: String,
    /// Directory containing a local GGUF file (e.g. "/models/qwen/").
    /// When set, Synapse loads the real model instead of the stub.
    pub gguf_dir: Option<String>,
    /// GGUF filename inside `gguf_dir`. Defaults to "model.gguf".
    pub gguf_file: String,
    /// HuggingFace model ID to download + ISQ-quantize (used when `gguf_dir` is unset).
    /// Example: "Qwen/Qwen2.5-1.5B-Instruct"
    pub hf_model: Option<String>,
    /// Force CPU inference even when CUDA is available.
    /// Useful when the GPU lacks ops the model needs (e.g. bf16 on sm_75 / GTX 1650).
    pub force_cpu: bool,
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            host: env::var("SYNAPSE_HOST").unwrap_or_else(|_| "127.0.0.1".to_string()),
            // 11434 is Ollama's default; we use 11435 so both can run side by side
            // (Synapse vs stock Ollama) for honest A/B benchmarking.
            port: env::var("SYNAPSE_PORT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(11435),
            model: env::var("SYNAPSE_MODEL").unwrap_or_else(|_| "synapse-stub".to_string()),
            gguf_dir: env::var("SYNAPSE_GGUF_DIR").ok(),
            gguf_file: env::var("SYNAPSE_GGUF_FILE")
                .unwrap_or_else(|_| "model.gguf".to_string()),
            hf_model: env::var("SYNAPSE_HF_MODEL").ok(),
            force_cpu: env::var("SYNAPSE_FORCE_CPU").is_ok(),
        }
    }
}
