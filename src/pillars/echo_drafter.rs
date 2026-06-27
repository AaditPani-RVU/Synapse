//! Echo Drafter -- Pillar 3 (Phase 3).
//!
//! Training-free speculative decoding whose draft comes from a suffix-automaton /
//! n-gram store built from N.O.R.A's own ChromaDB command history. Draft k tokens,
//! verify in one target forward pass (lossless -- output distribution unchanged).
//! NORA's command traffic is highly repetitive, which is exactly the regime where
//! a near-zero-cost retrieval draft wins (cf. prompt-lookup / Cacheback / Goose --
//! see LITERATURE_SURVEY.md S6).
//!
//! Status: placeholder seam (not yet wired).

/// Produces speculative draft tokens from recent/episodic command history.
#[allow(dead_code)]
pub trait Drafter: Send + Sync {
    /// Propose up to `k` likely continuation tokens given the context so far.
    /// Empty result == no confident draft; fall back to normal decoding.
    fn draft(&self, context: &[u32], k: usize) -> Vec<u32>;
}
