//! Project configuration (`<project>/.coderag/config.toml`).

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Root of the on-disk configuration.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CodeRagConfig {
    /// Embedding backend selection and tuning.
    pub embedding: EmbeddingConfig,
    /// Indexing / chunking tuning.
    pub index: IndexConfig,
    /// Search-time tuning.
    pub search: SearchConfig,
}

/// Which embedding backend to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderKind {
    /// Deterministic offline hashed bag-of-words embeddings. Zero network,
    /// zero cost, modest quality — the safe default and what the test suite
    /// runs against.
    Local,
    /// Any OpenAI-compatible `/v1/embeddings` endpoint (OpenAI, Azure
    /// gateways, Ollama, vLLM, LM Studio, ...).
    OpenAi,
}

impl ProviderKind {
    /// Machine-readable name used in the index signature.
    pub fn as_str(self) -> &'static str {
        match self {
            ProviderKind::Local => "local",
            ProviderKind::OpenAi => "openai",
        }
    }
}

/// Embedding backend configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EmbeddingConfig {
    /// Provider selection.
    pub provider: ProviderKind,
    /// Model name. Empty string means "provider default".
    pub model: String,
    /// Vector dimensionality. Must stay fixed for the life of an index.
    pub dimensions: usize,
    /// How many texts to send per embedding request.
    pub batch_size: usize,
    /// Override API base URL (OpenAI-compatible). Falls back to
    /// `OPENAI_BASE_URL`, then `https://api.openai.com/v1`.
    pub base_url: Option<String>,
    /// Environment variable holding the API key. Defaults to
    /// `OPENAI_API_KEY`. The key itself is never persisted.
    pub api_key_env: Option<String>,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: ProviderKind::Local,
            model: String::new(),
            dimensions: 256,
            batch_size: 64,
            base_url: None,
            api_key_env: None,
        }
    }
}

impl EmbeddingConfig {
    /// Effective model name after applying the provider default.
    pub fn effective_model(&self) -> &str {
        if self.model.is_empty() {
            match self.provider {
                ProviderKind::Local => crate::embedding::LOCAL_MODEL,
                ProviderKind::OpenAi => crate::embedding::OPENAI_DEFAULT_MODEL,
            }
        } else {
            &self.model
        }
    }

    /// Default configuration for the OpenAI provider.
    pub fn openai(model: Option<String>, dimensions: usize) -> Self {
        Self {
            provider: ProviderKind::OpenAi,
            model: model.unwrap_or_default(),
            dimensions,
            batch_size: 64,
            base_url: None,
            api_key_env: None,
        }
    }
}

/// Indexing / chunking configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IndexConfig {
    /// Maximum size of a chunk in bytes (~4 bytes/token). Oversized
    /// definitions are recursively split.
    pub max_chunk_chars: usize,
    /// Chunks whose content is smaller than this are dropped.
    pub min_chunk_chars: usize,
    /// How many bytes of file preamble (imports, module docs) to attach as
    /// context to every chunk of that file.
    pub preamble_chars: usize,
    /// Files larger than this are skipped entirely.
    pub max_file_bytes: u64,
    /// Directory names that are never indexed, in addition to `.git` and
    /// `.coderag` (which are always skipped).
    pub skip_dirs: Vec<String>,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            max_chunk_chars: 1600,
            min_chunk_chars: 24,
            preamble_chars: 512,
            max_file_bytes: 1_500_000,
            skip_dirs: [
                "target",
                "node_modules",
                "dist",
                "build",
                "out",
                ".venv",
                "venv",
                "__pycache__",
                ".next",
                ".hg",
                ".svn",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
        }
    }
}

/// Search-time configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SearchConfig {
    /// Number of results returned by default.
    pub top_k: usize,
    /// Cap on consecutive hits from the same file, to keep results diverse.
    pub max_per_file: usize,
    /// Weight of the lexical (token overlap) signal vs. cosine similarity.
    /// `0.15` means `score = 0.85 * cosine + 0.15 * lexical`.
    pub lexical_weight: f32,
    /// Discard hits below this final score.
    pub min_score: f32,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            top_k: 8,
            max_per_file: 3,
            lexical_weight: 0.15,
            min_score: 0.0,
        }
    }
}

impl CodeRagConfig {
    /// Validates ranges that would otherwise produce nonsense at runtime.
    pub fn validate(&self) -> Result<()> {
        if !(8..=4096).contains(&self.embedding.dimensions) {
            return Err(Error::Config(format!(
                "embedding.dimensions must be in 8..=4096, got {}",
                self.embedding.dimensions
            )));
        }
        if self.embedding.batch_size == 0 || self.embedding.batch_size > 2048 {
            return Err(Error::Config(
                "embedding.batch_size must be in 1..=2048".to_string(),
            ));
        }
        if self.index.min_chunk_chars >= self.index.max_chunk_chars {
            return Err(Error::Config(
                "index.min_chunk_chars must be smaller than index.max_chunk_chars".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.search.lexical_weight) {
            return Err(Error::Config(
                "search.lexical_weight must be in 0.0..=1.0".to_string(),
            ));
        }
        if self.search.top_k == 0 {
            return Err(Error::Config("search.top_k must be >= 1".to_string()));
        }
        Ok(())
    }

    /// Serializes to TOML for persistence in `.coderag/config.toml`.
    pub fn to_toml(&self) -> Result<String> {
        toml::to_string_pretty(self).map_err(|e| Error::Config(e.to_string()))
    }

    /// Parses `.coderag/config.toml` content.
    pub fn from_toml(input: &str) -> Result<Self> {
        let cfg: CodeRagConfig = toml::from_str(input).map_err(|e| Error::Config(e.to_string()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// String that uniquely identifies the embedding space of an index.
    /// Switching provider, model, or dimensionality requires a rebuild.
    pub fn embedding_signature(&self) -> String {
        format!(
            "{}:{}:{}",
            self.embedding.provider.as_str(),
            self.embedding.effective_model(),
            self.embedding.dimensions
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_roundtrips_through_toml() {
        let cfg = CodeRagConfig::default();
        let text = cfg.to_toml().unwrap();
        let parsed = CodeRagConfig::from_toml(&text).unwrap();
        assert_eq!(cfg, parsed);
    }

    #[test]
    fn rejects_out_of_range_dimensions() {
        let mut cfg = CodeRagConfig::default();
        cfg.embedding.dimensions = 4;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn signature_includes_model_and_dims() {
        let mut cfg = CodeRagConfig::default();
        cfg.embedding.provider = ProviderKind::OpenAi;
        cfg.embedding.model = "text-embedding-3-small".into();
        cfg.embedding.dimensions = 1536;
        assert_eq!(
            cfg.embedding_signature(),
            "openai:text-embedding-3-small:1536"
        );
    }
}
