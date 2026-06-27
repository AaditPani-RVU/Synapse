//! Router -- Pillar 4 (Phase 4).
//!
//! A tiny intent classifier routes trivial commands to a small model (or a cached
//! answer) and hard ones to the target model, so most commands never touch the big
//! model at all. Status: placeholder seam (not yet wired).

/// Where a given request should be served.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    /// Trivial / known intent -> small model or cached response.
    Small,
    /// Everything else -> the primary target model.
    Target,
}

/// Decides the route for an incoming request. Phase 4 backs this with a real
/// lightweight classifier; the default is "always Target" (a safe no-op).
#[allow(dead_code)]
pub trait Router: Send + Sync {
    fn route(&self, prompt: &str) -> Route;
}
