//! OpenAI-compatible embeddings API client.
//!
//! Talks to any endpoint implementing `POST {base_url}/embeddings`
//! (`OpenAI`, Azure AI gateways, Ollama, vLLM, LM Studio, ...). Requests are
//! batched, retried with exponential backoff on 429/5xx/network errors, and
//! responses are re-ordered by the API's own `index` field before validation.

use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::time::sleep;

use crate::error::{Error, Result};

use super::EmbeddingProvider;

/// Hard cap matching the OpenAI batch limit; smaller configured batch sizes
/// win.
const API_MAX_BATCH: usize = 2048;
const MAX_ATTEMPTS: u32 = 5;

/// Client for an OpenAI-compatible embeddings endpoint.
#[derive(Debug, Clone)]
pub struct OpenAiEmbeddings {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    model: String,
    dimensions: Option<usize>,
    batch_size: usize,
}

impl OpenAiEmbeddings {
    /// Creates a client. `dimensions` is forwarded as the API's Matryoshka
    /// truncation parameter when set.
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
        dimensions: Option<usize>,
    ) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(90))
                .build()
                .expect("reqwest client"),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key: api_key.into(),
            model: model.into(),
            dimensions,
            batch_size: API_MAX_BATCH,
        }
    }

    /// Overrides the batch size (≤ 2048).
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size.clamp(1, API_MAX_BATCH);
        self
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        #[derive(Serialize)]
        struct Request<'a> {
            model: &'a str,
            input: &'a [String],
            #[serde(skip_serializing_if = "Option::is_none")]
            dimensions: Option<usize>,
        }
        #[derive(Deserialize)]
        struct Response {
            data: Vec<Datum>,
        }
        #[derive(Deserialize)]
        struct Datum {
            index: usize,
            embedding: Vec<f32>,
        }

        let body = Request {
            model: &self.model,
            input: texts,
            dimensions: self.dimensions,
        };
        let url = format!("{}/embeddings", self.base_url);

        let mut attempt = 0u32;
        loop {
            attempt += 1;
            let result = self
                .client
                .post(&url)
                .bearer_auth(&self.api_key)
                .json(&body)
                .send()
                .await;

            match result {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        let parsed: Response = resp.json().await.map_err(Error::Http)?;
                        let mut data = parsed.data;
                        data.sort_by_key(|d| d.index);
                        let vectors: Vec<Vec<f32>> =
                            data.into_iter().map(|d| d.embedding).collect();
                        if vectors.len() != texts.len() {
                            return Err(Error::Embed(format!(
                                "provider returned {} vectors for {} inputs",
                                vectors.len(),
                                texts.len()
                            )));
                        }
                        if let Some(dim) = self.dimensions {
                            if vectors.iter().any(|v| v.len() != dim) {
                                return Err(Error::Embed(format!(
                                    "provider returned vectors of unexpected dimensionality (expected {dim})"
                                )));
                            }
                        }
                        return Ok(vectors);
                    }
                    let detail = resp.text().await.unwrap_or_default();
                    let retryable = status.as_u16() == 429 || status.is_server_error();
                    if !retryable || attempt >= MAX_ATTEMPTS {
                        return Err(Error::Embed(format!(
                            "embeddings API returned {status}: {detail}"
                        )));
                    }
                    tracing::warn!(
                        attempt,
                        status = %status,
                        "embeddings API error, retrying with backoff"
                    );
                }
                Err(err) => {
                    if attempt >= MAX_ATTEMPTS {
                        return Err(Error::Embed(format!(
                            "embeddings API unreachable after {MAX_ATTEMPTS} attempts: {err}"
                        )));
                    }
                    tracing::warn!(attempt, error = %err, "embeddings request failed, retrying");
                }
            }
            let backoff = Duration::from_millis(500u64.saturating_mul(1 << (attempt - 1)));
            sleep(backoff.min(Duration::from_secs(10_000))).await;
        }
    }
}

#[async_trait]
impl EmbeddingProvider for OpenAiEmbeddings {
    fn id(&self) -> &'static str {
        "openai"
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn dimensions(&self) -> usize {
        self.dimensions.unwrap_or(1536)
    }

    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(self.batch_size.max(1)) {
            out.extend(self.embed_batch(chunk).await?);
        }
        Ok(out)
    }
}
