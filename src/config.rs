//! Runtime configuration. Phase 0 reads a few env vars with sane defaults so
//! N.O.R.A can point at Synapse with zero code changes.

use std::env;

#[derive(Debug, Clone)]
pub struct Config {
    pub host: String,
    pub port: u16,
    /// Model name Synapse advertises to Ollama-compatible clients.
    pub model: String,
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
        }
    }
}
