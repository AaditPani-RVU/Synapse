//! The four Synapse pillars.
//!
//! Phase 0 ships them as documented seams; each is wired into the decode loop in
//! its own phase (see PLAN.md §4). They are declared now so the architecture is
//! legible from day one and the integration points are explicit.

pub mod grammar_gate;
pub mod prefix_vault;
pub mod echo_drafter;
pub mod router;
