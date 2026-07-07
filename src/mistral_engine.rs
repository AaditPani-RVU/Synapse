//! Real inference engine backed by mistral.rs.
//!
//! Two load paths:
//!   - `from_gguf`  — load a quantized GGUF file from a local directory
//!   - `from_hf`    — download from HuggingFace and ISQ-quantize on first run
//!
//! Both expose the same `Engine` trait so the API layer never needs to know which
//! path was used.  CUDA support is compiled in when `features = ["cuda"]` is added
//! to the mistralrs dependency in Cargo.toml; no code changes needed.
//!
//! # Lifetime note
//! mistral.rs `Stream<'a>` borrows `&'a Model` (a phantom anchor to tie its
//! lifetime).  To return a `'static` TokenStream, we bridge via a futures mpsc
//! channel: a spawned task owns the `Arc<Model>` and forwards tokens; the caller
//! gets the `'static` receiver.

use anyhow::Result;
use async_trait::async_trait;
use futures::{channel::mpsc, SinkExt};
use mistralrs::core::{
    AddModelConfig, AutoDeviceMapParams, DefaultSchedulerMethod, DeviceMapSetting, EngineConfig,
    GGUFLoaderBuilder, GGUFSpecificConfig, ModelDType, SchedulerConfig,
};
use mistralrs::model_builder_trait::build_model_from_pipeline;
use mistralrs::{
    best_device, IsqBits, Model, Response, TextMessageRole, TextMessages, TextModelBuilder,
    TokenSource,
};
use std::num::NonZeroUsize;
use std::sync::Arc;

use crate::engine::{Engine, GenerateRequest, TokenStream};

pub struct MistralEngine {
    model: Arc<Model>,
    name: String,
}

/// mistral.rs's `ModelDType::Auto` falls back to F16 on GPUs without BF16
/// support (compute capability < 8.0, e.g. this project's dev GTX 1650 /
/// sm_75). On CUDA, F16 compute for Qwen2.5-1.5B has been observed to overflow
/// into NaN/Inf logits (mid-stream `Invalid sampling probability` errors) —
/// the same F16 path on the CPU backend does not, so this is CUDA-kernel
/// specific, not a general precision problem. Force F32 whenever we're
/// actually using the GPU; `Auto` is fine for the CPU fallback.
fn gpu_safe_dtype(force_cpu: bool) -> ModelDType {
    if force_cpu {
        ModelDType::Auto
    } else {
        ModelDType::F32
    }
}

impl MistralEngine {
    /// Load a pre-quantized GGUF file from `model_dir/filename`.
    /// `force_cpu` bypasses CUDA and runs on CPU (useful when the GPU's compute
    /// capability lacks operations the model needs, e.g. bf16 on sm_75).
    ///
    /// This bypasses the ergonomic `GgufModelBuilder` (which hardcodes
    /// `ModelDType::Auto` with no way to override it) and drives the
    /// lower-level `mistralrs_core` loader API directly so we can force a safe
    /// dtype on CUDA. See `gpu_safe_dtype`.
    pub async fn from_gguf(
        model_dir: &str,
        filename: &str,
        name: String,
        force_cpu: bool,
    ) -> Result<Self> {
        let dtype = gpu_safe_dtype(force_cpu);
        tracing::info!(model_dir, filename, force_cpu, ?dtype, "loading GGUF model");

        let loader = GGUFLoaderBuilder::new(
            None,
            None,
            model_dir.to_string(),
            vec![filename.to_string()],
            GGUFSpecificConfig { topology: None },
            false,
            None,
        )
        .build();

        let device = best_device(force_cpu)?;
        let pipeline = loader.load_model_from_hf(
            None,
            TokenSource::CacheToken,
            &dtype,
            &device,
            false,
            DeviceMapSetting::Auto(AutoDeviceMapParams::default_text()),
            None,
            None,
        )?;

        let scheduler_config = SchedulerConfig::DefaultScheduler {
            method: DefaultSchedulerMethod::Fixed(NonZeroUsize::new(32).unwrap()),
        };

        let model = build_model_from_pipeline(
            pipeline,
            scheduler_config,
            AddModelConfig::new(EngineConfig::default()),
        )
        .await;

        tracing::info!("GGUF model loaded");
        Ok(Self { model: Arc::new(model), name })
    }

    /// Download `model_id` from HuggingFace and apply 4-bit ISQ quantization.
    /// The quantized weights are cached by the HF hub on subsequent runs.
    pub async fn from_hf(model_id: &str, name: String) -> Result<Self> {
        tracing::info!(model_id, "loading model from HuggingFace (ISQ Q4)");
        let model = TextModelBuilder::new(model_id)
            .with_auto_isq(IsqBits::Four)
            .with_logging()
            .build()
            .await?;
        tracing::info!("HuggingFace model loaded");
        Ok(Self { model: Arc::new(model), name })
    }
}

#[async_trait]
impl Engine for MistralEngine {
    fn model_name(&self) -> String {
        self.name.clone()
    }

    async fn generate(&self, req: GenerateRequest) -> anyhow::Result<TokenStream> {
        let mut messages = TextMessages::new();
        for msg in &req.messages {
            let role = match msg.role.as_str() {
                "system"    => TextMessageRole::System,
                "assistant" => TextMessageRole::Assistant,
                _           => TextMessageRole::User,
            };
            messages = messages.add_message(role, &msg.content);
        }

        let model = Arc::clone(&self.model);
        let (mut tx, rx) = mpsc::channel::<Result<String, String>>(256);

        tokio::spawn(async move {
            match model.stream_chat_request(messages).await {
                Ok(mut stream) => {
                    while let Some(response) = stream.next().await {
                        match response {
                            Response::Chunk(c) => {
                                if let Some(text) = c
                                    .choices
                                    .into_iter()
                                    .next()
                                    .and_then(|ch| ch.delta.content)
                                {
                                    if tx.send(Ok(text)).await.is_err() {
                                        break;
                                    }
                                }
                            }
                            // GGUF models may send a single Done instead of chunks
                            Response::Done(r) => {
                                if let Some(text) = r
                                    .choices
                                    .into_iter()
                                    .next()
                                    .and_then(|ch| ch.message.content)
                                {
                                    let _ = tx.send(Ok(text)).await;
                                }
                                break;
                            }
                            Response::InternalError(e) | Response::ValidationError(e) => {
                                let msg = e.to_string();
                                tracing::error!("mistralrs error: {msg}");
                                let _ = tx.send(Err(msg)).await;
                                break;
                            }
                            _ => {}
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("mistralrs stream error: {e}");
                    let _ = tx.send(Err(e.to_string())).await;
                }
            }
        });

        Ok(Box::pin(rx))
    }
}
