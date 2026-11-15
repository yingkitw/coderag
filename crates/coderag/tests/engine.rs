//! End-to-end tests exercising the engine against a temp project using the
//! offline local embedding provider (no network, fully hermetic).

use std::fs;
use std::path::Path;

use coderag::config::{CodeRagConfig, ProviderKind};
use coderag::engine::{Engine, IndexOptions, SearchOptions};
use tempfile::TempDir;

const FIB_RS: &str = r#"use std::collections::Vec;

/// Computes the n-th Fibonacci number iteratively.
pub fn fibonacci(n: u64) -> u64 {
    let (mut a, mut b) = (0u64, 1u64);
    for _ in 0..n {
        let next = a + b;
        a = b;
        b = next;
    }
    a
}
"#;

const HTTP_RS: &str = r#"/// Retry policy for HTTP requests with exponential backoff.
pub struct HttpRetryPolicy {
    pub max_attempts: u32,
}

impl HttpRetryPolicy {
    pub fn backoff_delay(&self, attempt: u32) -> u64 {
        1u64 << attempt.min(self.max_attempts)
    }
}
"#;

fn setup_project() -> (TempDir, CodeRagConfig) {
    let dir = TempDir::new().unwrap();
    fs::create_dir_all(dir.path().join("src")).unwrap();
    fs::write(dir.path().join("src/fib.rs"), FIB_RS).unwrap();
    fs::write(dir.path().join("src/http.rs"), HTTP_RS).unwrap();
    (dir, CodeRagConfig::default())
}

#[tokio::test]
async fn indexes_and_retrieves_relevant_code() {
    let (dir, config) = setup_project();
    let mut engine = Engine::open(dir.path(), config).unwrap();

    let report = engine.index(IndexOptions::default()).await.unwrap();
    assert_eq!(report.files_new, 2);
    assert_eq!(report.files_skipped_unknown, 0);
    assert!(report.chunks_embedded > 0);
    assert!(report.errors.is_empty(), "{:?}", report.errors);

    let hits = engine
        .search(
            "fibonacci number sequence computation",
            SearchOptions::default(),
        )
        .await
        .unwrap();
    assert!(!hits.is_empty());
    assert_eq!(hits[0].chunk.path, "src/fib.rs");
    assert_eq!(hits[0].chunk.symbol.as_deref(), Some("fibonacci"));

    let hits = engine
        .search("http retry backoff delay", SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(hits[0].chunk.path, "src/http.rs");
}

#[tokio::test]
async fn incremental_index_only_touches_changes() {
    let (dir, config) = setup_project();
    let mut engine = Engine::open(dir.path(), config).unwrap();

    let first = engine.index(IndexOptions::default()).await.unwrap();
    let total_first = first.chunks_embedded;

    // Touch one file with new content.
    let fib_path = dir.path().join("src/fib.rs");
    let mut updated = String::from(FIB_RS);
    updated.push_str("\npub fn lucas(n: u64) -> u64 {\n    fibonacci(n) + fibonacci(n + 1)\n}\n");
    fs::write(&fib_path, updated).unwrap();

    let second = engine.index(IndexOptions::default()).await.unwrap();
    assert_eq!(second.files_changed, 1);
    assert_eq!(second.files_unchanged, 1);
    assert_eq!(second.files_new, 0);
    assert!(
        second.chunks_embedded < total_first,
        "expected a partial re-embed: {} vs {total_first}",
        second.chunks_embedded
    );
    assert!(second.chunks_embedded > 0);

    // Delete a file.
    fs::remove_file(dir.path().join("src/http.rs")).unwrap();
    let third = engine.index(IndexOptions::default()).await.unwrap();
    assert_eq!(third.files_removed, 1);
    assert_eq!(third.files_unchanged, 1);
    assert_eq!(third.chunks_embedded, 0);

    // Deleted code is no longer retrievable.
    let hits = engine
        .search("http retry backoff delay", SearchOptions::default())
        .await
        .unwrap();
    assert!(hits.is_empty() || !hits.iter().any(|h| h.chunk.path == "src/http.rs"));

    let status = engine.status().unwrap();
    assert_eq!(status.files, 1);
}

#[tokio::test]
async fn switching_embedding_setup_requires_rebuild() {
    let (dir, config) = setup_project();
    {
        let mut engine = Engine::open(dir.path(), config.clone()).unwrap();
        engine.index(IndexOptions::default()).await.unwrap();
    }

    let mut other = config;
    other.embedding.dimensions = 128;

    let mut engine = Engine::open(dir.path(), other).unwrap();
    let err = engine.index(IndexOptions::default()).await.unwrap_err();
    assert!(matches!(err, coderag::Error::IndexMismatch { .. }), "{err}");

    let report = engine
        .index(IndexOptions {
            rebuild: true,
            ..Default::default()
        })
        .await
        .unwrap();
    assert!(report.rebuilt);
    assert_eq!(report.files_new, 2);
}

#[tokio::test]
async fn search_respects_top_k_and_max_per_file() {
    let dir = TempDir::new().unwrap();
    fs::create_dir_all(dir.path().join("src")).unwrap();
    for i in 0..6 {
        let src = format!(
            "/// Widget number {i} for the widget bench.\npub fn widget_{i}_refresh(x: u32) -> u32 {{\n    x + {i}\n}}\n"
        );
        fs::write(dir.path().join(format!("src/widget{i}.rs")), src).unwrap();
    }

    let mut engine = Engine::open(dir.path(), CodeRagConfig::default()).unwrap();
    engine.index(IndexOptions::default()).await.unwrap();

    let hits = engine
        .search(
            "widget refresh",
            SearchOptions {
                top_k: Some(4),
                max_per_file: Some(1),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert!(hits.len() <= 4);
    let mut paths = std::collections::HashSet::new();
    for hit in &hits {
        assert!(
            paths.insert(hit.chunk.path.clone()),
            "duplicate file in results"
        );
    }
}

#[tokio::test]
async fn openai_provider_without_key_fails_clearly() {
    // Ensure the var is not set for this test process.
    std::env::remove_var("OPENAI_API_KEY");
    let mut config = CodeRagConfig::default();
    config.embedding.provider = ProviderKind::OpenAi;
    let dir = TempDir::new().unwrap();
    let err = match Engine::open(Path::new(dir.path()), config) {
        Err(e) => e,
        Ok(_) => panic!("expected open without OPENAI_API_KEY to fail"),
    };
    let msg = err.to_string();
    assert!(msg.contains("OPENAI_API_KEY"), "{msg}");
}
