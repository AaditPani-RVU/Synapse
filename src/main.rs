//! Synapse -- the inference nervous system between NeuroSym-AI and N.O.R.A.

mod api;
mod config;
mod engine;
mod mistral_engine;
mod pillars;

use std::sync::Arc;

use axum::{
    routing::{get, post},
    Router,
};
use tracing_subscriber::EnvFilter;

use crate::api::AppState;
use crate::config::Config;
use crate::engine::{Engine, StubEngine};
use crate::mistral_engine::MistralEngine;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("synapse=info")),
        )
        .init();

    let cfg = Config::from_env();

    let engine: Arc<dyn Engine> = if let Some(dir) = &cfg.gguf_dir {
        tracing::info!(model = %cfg.model, force_cpu = cfg.force_cpu, "engine: GGUF");
        Arc::new(
            MistralEngine::from_gguf(dir, &cfg.gguf_file, cfg.model.clone(), cfg.force_cpu).await?,
        )
    } else if let Some(hf_id) = &cfg.hf_model {
        tracing::info!(model = %cfg.model, "engine: HuggingFace ISQ");
        Arc::new(MistralEngine::from_hf(hf_id, cfg.model.clone()).await?)
    } else {
        tracing::info!(model = %cfg.model, "engine: stub (set SYNAPSE_GGUF_DIR or SYNAPSE_HF_MODEL to use a real model)");
        Arc::new(StubEngine::new(cfg.model.clone()))
    };

    tracing::info!(
        host = %cfg.host,
        port = cfg.port,
        model = %engine.model_name(),
        "Synapse ready"
    );

    let state = AppState { engine };

    let app = Router::new()
        .route("/", get(api::root))
        .route("/api/version", get(api::version))
        .route("/api/tags", get(api::tags))
        .route("/api/chat", post(api::chat))
        .with_state(state);

    let addr = format!("{}:{}", cfg.host, cfg.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("listening on http://{addr}  (point N.O.R.A's Ollama base URL here)");
    axum::serve(listener, app).await?;

    Ok(())
}
