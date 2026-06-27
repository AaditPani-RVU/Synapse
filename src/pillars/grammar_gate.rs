//! Grammar Gate -- Pillar 1 (Phase 1, the headline).
//!
//! Compiles NeuroSym command schemas / action policies into a decoding grammar and
//! masks logits each step so the model can *only* emit valid, in-policy structures.
//! Result: output is valid-by-construction (no parse errors, no retries) AND
//! safety-checked by construction -- the single mechanism that fuses NeuroSym + NORA
//! inside the decoder. Phase 1 backs this with llguidance / XGrammar; GBNF fallback.
//!
//! Status: placeholder seam (not yet wired).

/// A constraint consulted at every decode step to mask invalid next tokens.
#[allow(dead_code)]
pub trait LogitsConstraint: Send + Sync {
    /// Given the token ids produced so far, return the allowed next-token ids.
    /// Phase 1 replaces this with a compiled grammar (Earley/FSM) over the vocab.
    fn allowed_next(&self, produced: &[u32]) -> Vec<u32>;

    /// Whether the constraint has reached an accepting (complete) state.
    fn is_complete(&self, produced: &[u32]) -> bool;
}
