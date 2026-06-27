//! Prefix Vault -- Pillar 2 (Phase 2).
//!
//! Radix-tree KV-cache reuse of the fixed N.O.R.A system prompt + NeuroSym guardrail
//! preamble across turns, so we stop recomputing the same 1-2k tokens every call.
//! Lower TTFT / latency. LRU eviction under a memory cap. (cf. RadixAttention/SGLang,
//! vLLM automatic prefix caching -- see LITERATURE_SURVEY.md S4.)
//!
//! Status: placeholder seam (not yet wired).

/// Reusable KV state for a cached token prefix. Phase 2 fills this with the
/// engine's actual paged KV blocks; here it marks the integration point.
#[allow(dead_code)]
pub struct PrefixVault {
    // TODO(phase-2): radix tree keyed on token-prefix -> KV block handles + LRU.
}

#[allow(dead_code)]
impl PrefixVault {
    pub fn new() -> Self {
        Self {}
    }

    /// Longest cached prefix (in tokens) reusable for `tokens`. 0 == cold start.
    pub fn longest_match(&self, _tokens: &[u32]) -> usize {
        0
    }
}
