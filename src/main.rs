//! Synapse -- the inference nervous system between NeuroSym-AI and N.O.R.A.
//!
//! Phase 0: a minimal, Ollama-compatible server backed by a stub engine. Its only
//! job is to prove the wiring -- NORA can talk to Synapse with a one-line config
//! change -- and to give us a baseline to benchmark every later pillar against.

mod api;
mod config;
mod engine;
mod pillars;

use std::sync::Arc;

use axum::{
    routing::{get, post},
    Router,
};
use tracing_subscriber::EnvFilter;

use crate::api::AppState;
use crate::config::Config;
use crate::engine::StubEngine;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("synapse=info")),
        )
        .init();

    let cfg = Config::from_env();
    tracing::info!(
        host = %cfg.host,
        port = cfg.port,
        model = %cfg.model,
        "starting Synapse (Phase 0: stub engine)"
    );

    let engine = Arc::new(StubEngine::new(cfg.model.clone()));
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
