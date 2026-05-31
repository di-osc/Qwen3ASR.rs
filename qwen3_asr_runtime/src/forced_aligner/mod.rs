//! Forced aligner (timestamps) support.
//!
//! This module is feature-gated because the aligner includes extra model logic and
//! language-specific tokenization.
//!
//! The official reference implementation lives at:
//! `../../../Qwen3-ASR/qwen_asr/inference/qwen3_forced_aligner.py`.

pub mod model;
pub mod processor;

pub use model::{ForcedAlignItem, ForcedAlignResult, Qwen3ForcedAligner};
pub use processor::{ForcedAlignProcessor, ItemMs, fix_timestamp};
