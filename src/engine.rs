//! The inference engine seam.
//!
//! Everything in Synapse plugs in here. Phase 0 ships `StubEngine` (echoes input)
//! purely to prove the server + N.O.R.A wiring end to end. Later phases provide a
//! real engine that wraps mistral.rs and layers the four pillars around the decode
//! loop: Router -> Prefix Vault -> Echo Drafter -> (Grammar Gate on the logits).

use async_trait::async_trait;
use futures::stream::{Stream, StreamExt};
use std::pin::Pin;
use std::time::Duration;

/// A streamed sequence of generated text chunks.
pub type TokenStream = Pin<Box<dyn Stream<Item = String> + Send>>;

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub model: String,
    pub prompt: String,
    pub messages: Vec<ChatMessage>,
}

/// The contract every Synapse engine implements. Keeping this trait stable is
/// what lets us swap the stub for mistral.rs without touching the API layer.
#[async_trait]
pub trait Engine: Send + Sync {
    fn model_name(&self) -> String;
    async fn generate(&self, req: GenerateRequest) -> anyhow::Result<TokenStream>;
}

/// Phase 0 placeholder: echoes the last user message back, streamed word by word
/// with a small delay so streaming + latency plumbing is exercised realistically.
pub struct StubEngine {
    model: String,
}

impl StubEngine {
    pub fn new(model: impl Into<String>) -> Self {
        Self { model: model.into() }
    }
}

#[async_trait]
impl Engine for StubEngine {
    fn model_name(&self) -> String {
        self.model.clone()
    }

    async fn generate(&self, req: GenerateRequest) -> anyhow::Result<TokenStream> {
        let last_user = req
            .messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .map(|m| m.content.clone())
            .unwrap_or(req.prompt);

        let reply = format!("[synapse-stub] You said: {last_user}");
        let words: Vec<String> = reply.split_inclusive(' ').map(str::to_string).collect();

        let stream = futures::stream::iter(words).then(|w| async move {
            tokio::time::sleep(Duration::from_millis(15)).await;
            w
        });

        Ok(Box::pin(stream))
    }
}
