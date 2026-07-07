//! Ollama-compatible HTTP API.
//!
//! N.O.R.A already speaks the Ollama protocol, so by implementing `/api/chat`,
//! `/api/tags`, and `/api/version` here, swapping NORA onto Synapse is a one-line
//! base-URL change (point it at this server's port instead of 11434).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use axum::{
    body::Body,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use bytes::Bytes;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::engine::{ChatMessage, Engine, GenerateRequest};

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<dyn Engine>,
}

#[derive(Debug, Deserialize)]
pub struct OllamaChatRequest {
    pub model: String,
    #[serde(default)]
    pub messages: Vec<OllamaMessage>,
    #[serde(default)]
    pub stream: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct OllamaMessage {
    pub role: String,
    pub content: String,
}

pub async fn root() -> &'static str {
    "Synapse - local inference nervous system (NeuroSym + N.O.R.A). Phase 0: stub engine.\n"
}

pub async fn version() -> Json<Value> {
    Json(json!({ "version": env!("CARGO_PKG_VERSION") }))
}

/// Advertise the single configured model so Ollama-style clients discover it.
pub async fn tags(State(state): State<AppState>) -> Json<Value> {
    let model = state.engine.model_name();
    Json(json!({
        "models": [{
            "name": model,
            "model": model,
            "modified_at": now(),
            "size": 0,
            "digest": "",
            "details": {
                "family": "synapse",
                "parameter_size": "stub",
                "quantization_level": "none"
            }
        }]
    }))
}

/// Ollama `/api/chat`. Supports streaming (NDJSON, the default) and non-streaming.
pub async fn chat(State(state): State<AppState>, Json(req): Json<OllamaChatRequest>) -> Response {
    let stream_mode = req.stream.unwrap_or(true);
    let model = req.model.clone();

    let gen_req = GenerateRequest {
        model: req.model,
        prompt: String::new(),
        messages: req
            .messages
            .into_iter()
            .map(|m| ChatMessage { role: m.role, content: m.content })
            .collect(),
    };

    let token_stream = match state.engine.generate(gen_req).await {
        Ok(s) => s,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    };

    if stream_mode {
        // Stop right after the first Err (inclusive) so a mid-stream failure is
        // reported instead of the connection just going quiet.
        let stopped = Arc::new(AtomicBool::new(false));
        let stopped_for_filter = Arc::clone(&stopped);
        let raw = token_stream.take_while(move |item| {
            let already_stopped = stopped_for_filter.load(Ordering::SeqCst);
            if item.is_err() {
                stopped_for_filter.store(true, Ordering::SeqCst);
            }
            futures::future::ready(!already_stopped)
        });

        let m = model.clone();
        let chunks = raw.map(move |item| match item {
            Ok(text) => to_ndjson(&json!({
                "model": m,
                "created_at": now(),
                "message": { "role": "assistant", "content": text },
                "done": false
            })),
            Err(e) => to_ndjson(&json!({
                "model": m,
                "created_at": now(),
                "message": { "role": "assistant", "content": "" },
                "done": true,
                "done_reason": "error",
                "error": e
            })),
        });

        let m2 = model;
        let tail = futures::stream::once(async move {
            if stopped.load(Ordering::SeqCst) {
                None
            } else {
                Some(to_ndjson(&json!({
                    "model": m2,
                    "created_at": now(),
                    "message": { "role": "assistant", "content": "" },
                    "done": true,
                    "done_reason": "stop"
                })))
            }
        })
        .filter_map(futures::future::ready);

        Response::builder()
            .header("content-type", "application/x-ndjson")
            .body(Body::from_stream(chunks.chain(tail)))
            .unwrap()
    } else {
        let results: Vec<Result<String, String>> = token_stream.collect().await;
        if let Some(e) = results.iter().find_map(|r| r.as_ref().err()) {
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e }))).into_response();
        }
        let full: String = results.into_iter().map(Result::unwrap).collect();
        Json(json!({
            "model": model,
            "created_at": now(),
            "message": { "role": "assistant", "content": full },
            "done": true,
            "done_reason": "stop"
        }))
        .into_response()
    }
}

fn to_ndjson(v: &Value) -> Result<Bytes, std::io::Error> {
    let mut buf = serde_json::to_vec(v).unwrap_or_default();
    buf.push(b'\n');
    Ok(Bytes::from(buf))
}

fn now() -> String {
    chrono::Utc::now().to_rfc3339()
}
