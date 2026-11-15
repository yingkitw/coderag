//! Golden retrieval benchmark — the project's retrieval-fidelity harness.
//!
//! A fixed multi-language corpus is indexed with the offline `local`
//! provider, then a set of golden queries asserts that expected files (and,
//! where the vocabulary is distinctive, expected symbols) rank at the top.
//! This is the retrieval equivalent of float-tolerance tests: every change
//! to chunking, embedding, or scoring must keep it green — or consciously
//! update it with justification.
//!
//! Conventions (see MEMORY.md §6):
//! - The local provider is lexical, so queries deliberately share vocabulary
//!   with their targets; the benchmark measures the *pipeline* (chunking,
//!   hybrid scoring, diversity caps), not the embedder's semantics.
//! - `strict` goldens demand top-1; `loose` goldens demand top-3. Prefer
//!   adding strict cases; loosen only with an inspected, written reason.
//! - The OpenAI variant is `#[ignore]`d (needs `OPENAI_API_KEY`); it reports
//!   hits@1/hits@3 and requires top-3 containment.

use std::path::Path;

use coderag::config::{CodeRagConfig, EmbeddingConfig};
use coderag::engine::{Engine, IndexOptions, SearchOptions};
use tempfile::TempDir;

mod corpus;

/// One golden expectation.
struct Golden {
    query: &'static str,
    expect_path: &'static str,
    /// Symbol that must appear on the top-1 hit (strict mode only).
    expect_symbol: Option<&'static str>,
    /// Strict: expected path at rank 1. Loose: within the top 3.
    strict: bool,
}

fn goldens() -> Vec<Golden> {
    use Golden as G;
    vec![
        // --- strict: distinctive vocabulary, stable top-1 -------------------
        G {
            query: "http retry exponential backoff delay",
            expect_path: "src/retry.rs",
            expect_symbol: Some("compute_backoff_delay"),
            strict: true,
        },
        G {
            query: "binary search tree insert node",
            expect_path: "src/btree.rs",
            expect_symbol: Some("binary_search_tree_insert"),
            strict: true,
        },
        G {
            query: "inorder traversal collect sorted values",
            expect_path: "src/btree.rs",
            expect_symbol: Some("inorder_traversal"),
            strict: true,
        },
        G {
            query: "parse toml configuration file",
            expect_path: "src/config_loader.rs",
            expect_symbol: Some("parse_toml_config"),
            strict: true,
        },
        G {
            query: "event bus subscribe publish listeners",
            expect_path: "src/event_bus.rs",
            expect_symbol: Some("EventBus"),
            strict: true,
        },
        G {
            query: "median absolute deviation outliers",
            expect_path: "scripts/stats.py",
            expect_symbol: Some("median_absolute_deviation"),
            strict: true,
        },
        G {
            query: "polygon area vertices",
            expect_path: "src/geometry.rs",
            expect_symbol: Some("polygon_area"),
            strict: true,
        },
        // --- loose: shared vocabulary, top-3 containment -------------------
        G {
            query: "retry policy attempts exhausted give up",
            expect_path: "src/retry.rs",
            expect_symbol: None,
            strict: true,
        },
        G {
            query: "merge default config values missing fields",
            expect_path: "src/config_loader.rs",
            expect_symbol: None,
            strict: false,
        },
        G {
            query: "triangle centroid average of vertices",
            expect_path: "src/geometry.rs",
            expect_symbol: None,
            strict: true,
        },
        G {
            query: "data pipeline transform records stages",
            expect_path: "scripts/pipeline.py",
            expect_symbol: None,
            strict: true,
        },
        G {
            query: "api client fetch json request timeout",
            expect_path: "web/api-client.ts",
            expect_symbol: None,
            strict: true,
        },
        G {
            query: "refresh session token expired renewal",
            expect_path: "web/api-client.ts",
            expect_symbol: None,
            strict: true,
        },
        G {
            query: "process job queue worker loop",
            expect_path: "queue/worker.go",
            expect_symbol: None,
            strict: true,
        },
        G {
            query: "drain pending jobs batch",
            expect_path: "queue/worker.go",
            expect_symbol: None,
            strict: true,
        },
    ]
}

fn ranked(hits: &[coderag::engine::SearchHit]) -> String {
    hits.iter()
        .take(5)
        .map(|h| {
            format!(
                "\n  {:.3} {}:{} {}",
                h.score,
                h.chunk.path,
                h.chunk.start_line,
                h.chunk.symbol.as_deref().unwrap_or("-")
            )
        })
        .collect()
}

fn open_engine(dir: &Path, config: CodeRagConfig) -> Engine {
    Engine::open(dir, config).expect("engine opens")
}

async fn index_corpus(dir: &Path, config: CodeRagConfig) -> Engine {
    corpus::write(dir);
    let mut engine = open_engine(dir, config);
    let report = engine
        .index(IndexOptions::default())
        .await
        .expect("index ok");
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    assert!(report.chunks_embedded > 0);
    engine
}

#[tokio::test]
async fn golden_queries_rank_expected_chunks() {
    let dir = TempDir::new().unwrap();
    let engine = index_corpus(dir.path(), CodeRagConfig::default()).await;

    let opts = SearchOptions {
        top_k: Some(10),
        ..Default::default()
    };
    let mut strict_passed = 0usize;
    for golden in goldens() {
        let hits = engine.search(golden.query, opts.clone()).await.unwrap();
        let window = if golden.strict { 1 } else { 3 };
        let in_window = hits
            .iter()
            .take(window)
            .any(|h| h.chunk.path == golden.expect_path);
        assert!(
            in_window,
            "query {:?} expected {} in top-{}; got:{}",
            golden.query,
            golden.expect_path,
            window,
            ranked(&hits)
        );
        if let Some(symbol) = golden.expect_symbol {
            // Symbols get a fixed top-3 tolerance regardless of path
            // strictness: hashed-BoW cosine favors short chunks (see
            // MEMORY.md §5), so a struct sharing vocabulary can
            // legitimately outrank the method a query names — the method
            // must still be *retrieved* nearby.
            const SYMBOL_WINDOW: usize = 3;
            let symbol_in_window = hits
                .iter()
                .take(SYMBOL_WINDOW)
                .any(|h| h.chunk.symbol.as_deref() == Some(symbol));
            assert!(
                symbol_in_window,
                "query {:?} expected symbol {symbol} within top-{SYMBOL_WINDOW}; got:{}",
                golden.query,
                ranked(&hits)
            );
        }
        if golden.strict {
            strict_passed += 1;
        }
    }
    // Keep the benchmark honest: most goldens must be strict.
    let total = goldens().len();
    assert!(
        strict_passed * 2 >= total,
        "at least half the goldens should be strict (got {strict_passed}/{total})"
    );
}

/// Filters narrow results before ranking; each is exercised end to end
/// against the golden corpus.
#[tokio::test]
async fn filters_restrict_results() {
    let dir = TempDir::new().unwrap();
    let engine = index_corpus(dir.path(), CodeRagConfig::default()).await;
    let base = SearchOptions {
        top_k: Some(20),
        ..Default::default()
    };

    // Language filter: everything comes from the Go worker.
    let hits = engine
        .search(
            "queue job worker",
            SearchOptions {
                language: Some("go".into()),
                ..base.clone()
            },
        )
        .await
        .unwrap();
    assert!(!hits.is_empty(), "language filter found nothing");
    assert!(
        hits.iter()
            .all(|h| h.chunk.language == "go" && h.chunk.path == "queue/worker.go"),
        "non-go results leaked: {:?}",
        hits.iter().map(|h| &h.chunk.path).collect::<Vec<_>>()
    );

    // Path glob: only the web client.
    let hits = engine
        .search(
            "client session token",
            SearchOptions {
                path_glob: Some("web/*".into()),
                ..base.clone()
            },
        )
        .await
        .unwrap();
    assert!(!hits.is_empty(), "path filter found nothing");
    assert!(
        hits.iter().all(|h| h.chunk.path == "web/api-client.ts"),
        "glob leaked results: {:?}",
        hits.iter().map(|h| &h.chunk.path).collect::<Vec<_>>()
    );

    // Symbols only: no file/block chunks survive.
    let hits = engine
        .search(
            "config",
            SearchOptions {
                symbols_only: true,
                ..base.clone()
            },
        )
        .await
        .unwrap();
    assert!(!hits.is_empty(), "symbols_only found nothing");
    assert!(
        hits.iter().all(|h| h.chunk.symbol.is_some()),
        "symbols_only leaked synthetic chunks: {:?}",
        hits.iter()
            .map(|h| (h.chunk.kind.clone(), h.chunk.symbol.clone()))
            .collect::<Vec<_>>()
    );

    // An impossible glob filters to nothing rather than erroring.
    let hits = engine
        .search(
            "config",
            SearchOptions {
                path_glob: Some("no/such/dir/**".into()),
                ..base
            },
        )
        .await
        .unwrap();
    assert!(hits.is_empty());

    // Invalid globs are configuration errors, not panics.
    let err = engine
        .search(
            "config",
            SearchOptions {
                path_glob: Some("{invalid".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap_err();
    assert!(err.to_string().contains("glob"), "{err}");
}

/// ANN parity (feature `ann` only): with the candidate pool capped at the
/// corpus size, sqlite-vec candidate retrieval must return **identical**
/// results to the brute-force scan — this is the TODO acceptance criterion
/// ("same golden results as brute force at recall@10 ≥ 0.95").
#[cfg(feature = "ann")]
#[tokio::test]
async fn ann_matches_brute_force_on_golden_queries() {
    let dir = TempDir::new().unwrap();
    let engine = index_corpus(dir.path(), CodeRagConfig::default()).await;

    for golden in goldens() {
        // A tiny min_score drops the zero-score tail where table-order tie
        // breaks are meaningless; everything above must match exactly.
        let options = |use_ann: bool| SearchOptions {
            top_k: Some(10),
            min_score: Some(f32::EPSILON),
            use_ann: Some(use_ann),
            ..Default::default()
        };
        let ann = engine.search(golden.query, options(true)).await.unwrap();
        let brute = engine.search(golden.query, options(false)).await.unwrap();
        let keys = |hits: &[coderag::engine::SearchHit]| {
            hits.iter()
                .map(|h| (h.chunk.path.clone(), h.chunk.start_line, h.score.to_bits()))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            keys(&ann),
            keys(&brute),
            "ANN and brute-force disagree for {:?}",
            golden.query
        );
    }
}

/// Corpus chunks must carry per-language metadata — filters and status
/// depend on it, and it proves multi-language chunking end to end.
#[tokio::test]
async fn corpus_is_indexed_across_all_languages() {
    let dir = TempDir::new().unwrap();
    let engine = index_corpus(dir.path(), CodeRagConfig::default()).await;
    let status = engine.status().unwrap();
    let languages: std::collections::HashSet<&str> = status
        .languages
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    for expected in ["rust", "python", "typescript", "go"] {
        assert!(
            languages.contains(expected),
            "missing language {expected}: {languages:?}"
        );
    }
}

/// OpenAI-provider variant: same goldens, hits@1 / hits@3 report.
/// Opt-in (network + key): `cargo test -p coderag --test retrieval_fixtures -- --ignored`
#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and network"]
async fn openai_hits_at_k() {
    if std::env::var("OPENAI_API_KEY").is_err() {
        eprintln!("skipping: OPENAI_API_KEY not set");
        return;
    }
    let dir = TempDir::new().unwrap();
    let config = CodeRagConfig {
        embedding: EmbeddingConfig::openai(None, 1536),
        ..Default::default()
    };
    let engine = index_corpus(dir.path(), config).await;

    let opts = SearchOptions {
        top_k: Some(10),
        ..Default::default()
    };
    let mut hits1 = 0usize;
    let mut hits3 = 0usize;
    let total = goldens().len();
    for golden in goldens() {
        let hits = engine.search(golden.query, opts.clone()).await.unwrap();
        let rank = hits
            .iter()
            .position(|h| h.chunk.path == golden.expect_path)
            .map(|i| i + 1);
        if rank.is_some_and(|r| r <= 1) {
            hits1 += 1;
        }
        if rank.is_some_and(|r| r <= 3) {
            hits3 += 1;
        } else {
            eprintln!(
                "miss: {:?} → {} not in top 3:{}",
                golden.query,
                golden.expect_path,
                ranked(&hits)
            );
        }
    }
    eprintln!("openai golden results: hits@1 = {hits1}/{total}, hits@3 = {hits3}/{total}");
    assert!(
        hits3 * 2 >= total,
        "hits@3 = {hits3}/{total} is below the 50% floor"
    );
}
