//! # coderag
//!
//! Retrieval engine for codebases: **syntax-aware chunking → pluggable
//! embeddings → persistent hybrid search**.
//!
//! The crate is deliberately boring where it can be (SQLite, brute-force
//! cosine) and clever only where it pays off (tree-sitter chunking that
//! respects definition boundaries, incremental indexing keyed by content
//! hashes, a hybrid lexical/semantic scorer).
//!
//! ## Library usage
//!
//! ```no_run
//! use coderag::config::CodeRagConfig;
//! use coderag::engine::{Engine, IndexOptions, SearchOptions};
//!
//! # async fn demo() -> coderag::error::Result<()> {
//! let config = CodeRagConfig::default(); // local offline embeddings
//! let mut engine = Engine::open(std::path::Path::new("."), config)?;
//!
//! engine.index(IndexOptions::default()).await?;
//! let hits = engine.search("cosine similarity scoring", SearchOptions::default()).await?;
//! for hit in hits {
//!     println!("{:.3} {}:{}-{}", hit.score, hit.chunk.path, hit.chunk.start_line, hit.chunk.end_line);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Layout
//!
//! | Module | Responsibility |
//! |---|---|
//! | [`chunking`] | tree-sitter definition-aware chunker |
//! | [`langs`] | grammar registry, feature-gated per language |
//! | [`embedding`] | `EmbeddingProvider` trait, offline + OpenAI-compatible backends |
//! | [`store`] | SQLite persistence for files, chunks, vectors |
//! | [`engine`] | incremental indexing pipeline + hybrid search |
//! | [`config`] | `.coderag/config.toml` |
//! | [`tokens`] | identifier-aware tokenizer shared by scorer + local backend |

// `deny` (not `forbid`) so the single audited FFI block in
// `store/ann.rs` can carry an explicit `#[allow(unsafe_code)]` plus a
// SAFETY comment; everything else stays unsafe-free.
#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod chunking;
pub mod config;
pub mod embedding;
pub mod engine;
pub mod error;
pub mod langs;
pub mod store;
pub mod tokens;

pub use chunking::{Chunk, ChunkOptions, Chunker};
pub use config::{CodeRagConfig, EmbeddingConfig, IndexConfig, ProviderKind, SearchConfig};
pub use engine::{
    Engine, IndexOptions, IndexReport, Progress, ProgressCallback, SearchHit, SearchOptions,
    StatusReport,
};
pub use error::{Error, Result};

/// Names of the languages compiled into this build.
pub fn supported_languages() -> Vec<&'static str> {
    langs::supported_names()
}

/// Crate version, recorded in the index metadata.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
