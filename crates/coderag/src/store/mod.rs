//! Persistent vector storage.

pub mod sqlite;

#[cfg(feature = "ann")]
pub mod ann;

pub use sqlite::{SqliteStore, StoreStats};

/// A chunk together with its stored embedding, as returned for search.
pub struct SearchRecord {
    /// The chunk row.
    pub chunk: crate::chunking::Chunk,
    /// Unit vector read back from disk.
    pub embedding: Vec<f32>,
}
