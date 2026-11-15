//! Deterministic offline embeddings via feature hashing.
//!
//! Each token (see [`crate::tokens`]) is hashed with FNV-1a; the hash picks a
//! bucket and a sign, and the token's sublinear term frequency is accumulated
//! there. The result is L2-normalized. This is *not* a neural embedding — it
//! has no notion of synonymy — but it is fully deterministic across runs and
//! platforms, which keeps stored vectors valid forever and makes the test
//! suite hermetic.

use async_trait::async_trait;

use crate::error::Result;
use crate::tokens::{fnv1a64, tokenize};

use super::{EmbeddingProvider, LOCAL_MODEL};

/// Hashed bag-of-words embedding provider.
#[derive(Debug, Clone)]
pub struct LocalEmbeddings {
    dim: usize,
}

impl LocalEmbeddings {
    /// Creates a provider producing `dim`-dimensional vectors.
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// Embeds a single text into a unit vector (zero vector for empty input).
    pub fn embed_one_sync(text: &str, dim: usize) -> Vec<f32> {
        let mut vec = vec![0f32; dim];
        let mut counts = std::collections::HashMap::new();
        for token in tokenize(text) {
            *counts.entry(token).or_insert(0u32) += 1;
        }
        for (token, count) in counts {
            let hash = fnv1a64(token.as_bytes());
            let bucket = (hash as usize) % dim;
            let sign = if (hash >> 63) & 1 == 0 { 1.0 } else { -1.0 };
            let weight = 1.0 + (count as f32).ln();
            vec[bucket] += sign * weight;
        }
        let norm = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > f32::EPSILON {
            for v in &mut vec {
                *v /= norm;
            }
        }
        vec
    }
}

#[async_trait]
impl EmbeddingProvider for LocalEmbeddings {
    fn id(&self) -> &'static str {
        "local"
    }

    fn model(&self) -> &str {
        LOCAL_MODEL
    }

    fn dimensions(&self) -> usize {
        self.dim
    }

    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| Self::embed_one_sync(t, self.dim))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_normalized() {
        let a = LocalEmbeddings::embed_one_sync("fn binary_tree_insert(node: &mut Node)", 64);
        let b = LocalEmbeddings::embed_one_sync("fn binary_tree_insert(node: &mut Node)", 64);
        assert_eq!(a, b);
        let norm: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm was {norm}");
        assert_eq!(a.len(), 64);
    }

    #[test]
    fn related_texts_score_higher_than_unrelated() {
        let q = LocalEmbeddings::embed_one_sync("binary tree insert node", 128);
        let related =
            LocalEmbeddings::embed_one_sync("fn binary_tree_insert(node: &mut Node) {", 128);
        let other = LocalEmbeddings::embed_one_sync("impl HttpRetryPolicy for Client {", 128);
        let cos = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        assert!(
            cos(&q, &related) > cos(&q, &other),
            "related={} other={}",
            cos(&q, &related),
            cos(&q, &other)
        );
    }

    #[test]
    fn empty_text_gives_zero_vector() {
        let v = LocalEmbeddings::embed_one_sync("", 32);
        assert!(v.iter().all(|x| *x == 0.0));
    }
}
