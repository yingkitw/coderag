//! Error type shared across the crate.

use std::path::PathBuf;

/// Errors returned by the coderag engine.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Underlying I/O failure.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// SQLite storage failure.
    #[error("database error: {0}")]
    Database(#[from] rusqlite::Error),

    /// HTTP transport failure while calling an embedding API.
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),

    /// Malformed JSON while talking to an embedding API.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    /// The configuration file could not be parsed or is inconsistent.
    #[error("configuration error: {0}")]
    Config(String),

    /// The embedding provider failed or returned an unusable response.
    #[error("embedding provider error: {0}")]
    Embed(String),

    /// The file's language has no parser compiled in.
    #[error("unsupported language for {0}")]
    UnsupportedLanguage(PathBuf),

    /// A source file could not be parsed.
    #[error("failed to parse {path}: {message}")]
    Parse {
        /// Path of the offending file.
        path: PathBuf,
        /// Description of the failure.
        message: String,
    },

    /// The project has not been initialized (`coderag init`).
    #[error("project at {0} is not initialized; run `coderag init` first")]
    NotInitialized(PathBuf),

    /// The index was built with a different provider/model/dimension combo.
    #[error(
        "index was built with a different embedding setup (indexed: {indexed}, requested: {requested}); \
         run `coderag index --rebuild` to re-index from scratch"
    )]
    IndexMismatch {
        /// Signature recorded in the index metadata.
        indexed: String,
        /// Signature derived from the current configuration.
        requested: String,
    },
}

/// Convenient result alias used throughout the crate.
pub type Result<T, E = Error> = std::result::Result<T, E>;
