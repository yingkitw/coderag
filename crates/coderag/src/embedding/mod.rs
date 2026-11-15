//! Pluggable embedding backends.
//!
//! An [`EmbeddingProvider`] turns texts into fixed-size unit vectors. Two
//! implementations ship with the crate:
//!
//! - [`LocalEmbeddings`] — a deterministic, dependency-free hashed
//!   bag-of-words embedding. Runs offline, makes tests hermetic, and gives
//!   decent lexical retrieval. Not a neural model; quality is modest.
//! - [`OpenAiEmbeddings`] — any OpenAI-compatible `/v1/embeddings` endpoint
//!   (OpenAI, Azure gateways, Ollama, vLLM, LM Studio) with batching and
//!   exponential-backoff retries.

pub mod local;
pub mod openai;

use std::sync::Arc;

use async_trait::async_trait;

use crate::config::CodeRagConfig;
use crate::error::{Error, Result};

pub use local::LocalEmbeddings;
pub use openai::OpenAiEmbeddings;

/// Model name advertised by the local provider.
pub const LOCAL_MODEL: &str = "hashed-bow-v1";
/// Default OpenAI model.
pub const OPENAI_DEFAULT_MODEL: &str = "text-embedding-3-small";
/// Default OpenAI-compatible base URL.
pub const OPENAI_DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// A source of embeddings for texts.
///
/// Implementations must return vectors of length [`EmbeddingProvider::dimensions`]
/// in the same order as the input texts, and should be L2-normalized so plain
/// dot products approximate cosine similarity.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Short provider id (`local`, `openai`, ...) used in index signatures.
    fn id(&self) -> &'static str;

    /// Model identifier used in index signatures.
    fn model(&self) -> &str;

    /// Vector dimensionality.
    fn dimensions(&self) -> usize;

    /// Embeds a batch of texts, preserving input order.
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>>;

    /// Embeds a single text (convenience wrapper with a provided default).
    async fn embed_one(&self, text: String) -> Result<Vec<f32>> {
        let mut out = self.embed(vec![text]).await?;
        if out.len() != 1 {
            return Err(Error::Embed("provider returned wrong batch size".into()));
        }
        Ok(out.remove(0))
    }
}

/// Builds the provider described by `config`, reading API keys and base URLs
/// from the environment when the config defers to them.
pub fn provider_from_config(config: &CodeRagConfig) -> Result<Arc<dyn EmbeddingProvider>> {
    let cfg = &config.embedding;
    let provider: Arc<dyn EmbeddingProvider> = match cfg.provider {
        crate::config::ProviderKind::Local => Arc::new(LocalEmbeddings::new(cfg.dimensions)),
        crate::config::ProviderKind::OpenAi => {
            let base_url = cfg
                .base_url
                .clone()
                .or_else(|| std::env::var("OPENAI_BASE_URL").ok())
                .unwrap_or_else(|| OPENAI_DEFAULT_BASE_URL.to_string());
            let key_env = cfg
                .api_key_env
                .clone()
                .unwrap_or_else(|| "OPENAI_API_KEY".to_string());
            let api_key = std::env::var(&key_env).map_err(|_| {
                Error::Config(format!(
                    "environment variable {key_env} is required for the openai provider"
                ))
            })?;
            Arc::new(OpenAiEmbeddings::new(
                base_url,
                api_key,
                cfg.effective_model().to_string(),
                Some(cfg.dimensions),
            ))
        }
    };
    Ok(provider)
}
