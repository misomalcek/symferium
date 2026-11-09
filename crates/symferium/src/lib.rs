//! Symferium: a local multi-model AI runtime for Apple Silicon.
//!
//! One Rust daemon supervises the best engine for each model (llama.cpp, MLX, whisper.cpp, a TTS
//! sidecar), serves each model on its own OpenAI-compatible localhost port, and keeps every load
//! inside a memory budget measured on the machine. See docs/PLAN.md.

pub mod api;
pub mod bench;
pub mod catalog;
pub mod config;
pub mod download;
pub mod engines;
pub mod estimate;
pub mod events;
pub mod gguf;
pub mod governor;
pub mod launchd;
pub mod library;
pub mod presets;
pub mod probe;
pub mod proxy;
pub mod runtime;
pub mod state;
pub mod symbiosis;
pub mod tools;

pub use runtime::Runtime;
