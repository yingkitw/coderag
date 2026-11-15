//! ANN recall benchmark — feature-gated to `ann` builds
//! (`cargo test -p coderag --features ann --test ann_recall`).
//!
//! Builds a synthetic multi-file corpus (deterministic LCG, no `rand`
//! dependency), indexes it once, and compares candidate strategies on the
//! same queries: sqlite-vec KNN + hybrid rerank vs. the brute-force scan.
//! sqlite-vec 0.1.x is an *exact* SIMD scanner, so this validates the
//! plumbing (mirror sync, id hydration, rerank) more than approximation
//! quality — the recall floor stays meaningful once real ANN lands.

#![cfg(feature = "ann")]

use std::fs;
use std::path::Path;

use coderag::config::CodeRagConfig;
use coderag::engine::{Engine, IndexOptions, SearchOptions};
use tempfile::TempDir;

/// Tiny deterministic LCG so the corpus is reproducible without deps.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[(self.next() as usize) % items.len()]
    }
}

const VOCAB: &[&str] = &[
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet",
    "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango",
    "uniform", "victor", "whiskey", "xray", "yankee", "zulu", "anchor", "beacon", "cinder",
    "dovetail", "ember", "fathom", "girder", "harbor", "inlet", "jasper", "kelp", "lantern",
    "marble", "nimbus", "onyx", "pinnacle",
];

const FILES: usize = 120;
const FUNCS_PER_FILE: usize = 5;
const TOP_K: usize = 10;
const QUERIES: usize = 20;

fn write_corpus(root: &Path) {
    let mut rng = Lcg(0x5eed_cafe);
    for file_index in 0..FILES {
        let mut src = String::from("//! Synthetic module.\n\n");
        for fn_index in 0..FUNCS_PER_FILE {
            let words: Vec<&str> = (0..8).map(|_| *rng.pick(VOCAB)).collect();
            src.push_str(&format!(
                "/// Handles {} {} payloads.\npub fn task_{file_index}_{fn_index}(input: u32) -> u32 {{\n    let tag = \"{}\";\n    input.wrapping_add({file_index} as u32 ^ {fn_index} as u32 ^ tag.len() as u32)\n}}\n\n",
                words[0], words[1], words[2]
            ));
            // sprinkle the remaining words in the body for lexical spread
            src.push_str(&format!(
                "pub fn audit_{file_index}_{fn_index}(x: u32) -> u32 {{\n    // {} {} {} markers\n    x.rotate_left({fn_index} as u32)\n}}\n\n",
                words[3], words[4], words[5]
            ));
        }
        fs::write(root.join(format!("synth/mod_{file_index:03}.rs")), src).unwrap();
    }
}

fn query_set() -> Vec<String> {
    let mut rng = Lcg(0x0a11_ce5e);
    (0..QUERIES)
        .map(|_| {
            let a = *rng.pick(VOCAB);
            let b = *rng.pick(VOCAB);
            format!("{a} {b} payload")
        })
        .collect()
}

fn hit_keys(hits: &[coderag::engine::SearchHit]) -> std::collections::HashSet<(String, u32)> {
    hits.iter()
        .map(|h| (h.chunk.path.clone(), h.chunk.start_line))
        .collect()
}

#[tokio::test]
async fn ann_recall_matches_brute_force() {
    let dir = TempDir::new().unwrap();
    fs::create_dir_all(dir.path().join("synth")).unwrap();
    write_corpus(dir.path());

    let mut engine = Engine::open(dir.path(), CodeRagConfig::default()).unwrap();
    let report = engine.index(IndexOptions::default()).await.unwrap();
    assert!(report.chunks_embedded >= FILES * FUNCS_PER_FILE * 2 - 20);

    let brute = SearchOptions {
        top_k: Some(TOP_K),
        use_ann: Some(false),
        ..Default::default()
    };
    let ann = SearchOptions {
        top_k: Some(TOP_K),
        use_ann: Some(true),
        ..Default::default()
    };

    let mut total_recall = 0.0f32;
    let mut top1_agreements = 0usize;
    for query in query_set() {
        let brute_hits = engine.search(&query, brute.clone()).await.unwrap();
        let ann_hits = engine.search(&query, ann.clone()).await.unwrap();
        assert!(
            !brute_hits.is_empty(),
            "brute force found nothing for {query}"
        );
        assert_eq!(
            ann_hits.len(),
            brute_hits.len(),
            "pool exhausted for {query}"
        );

        let brute_keys = hit_keys(&brute_hits);
        let ann_keys = hit_keys(&ann_hits);
        let overlap = brute_keys.intersection(&ann_keys).count();
        total_recall += overlap as f32 / TOP_K as f32;
        if brute_hits[0].chunk.id == ann_hits[0].chunk.id {
            top1_agreements += 1;
        }
    }

    let avg_recall = total_recall / QUERIES as f32;
    let agreement = top1_agreements as f32 / QUERIES as f32;
    // sqlite-vec 0.1.x is an exact SIMD scanner, so the residual gap is
    // *two-stage* divergence, not index corruption: the hybrid rerank
    // (lexical + symbol bonus) promotes chunks that pure cosine ranks
    // below the candidate pool, and vice versa on the scan side. Measured
    // ~0.94 at pool = 500/1200. Tighter parity is asserted by the golden
    // benchmark (identical results when pool ≥ corpus); this test guards
    // the plumbing and catches mirror/hydration regressions.
    eprintln!("ann vs brute: recall@{TOP_K} = {avg_recall:.3}, top-1 agreement = {agreement:.3}");
    assert!(
        avg_recall >= 0.90,
        "recall@{TOP_K} = {avg_recall:.3} below the 0.90 floor"
    );
    assert!(
        agreement >= 0.8,
        "top-1 agreement = {agreement:.3} below 0.8"
    );
}

/// Incremental re-index must keep the vec0 mirror in lockstep: replaced
/// chunks' old vectors must disappear.
#[tokio::test]
async fn ann_mirror_follows_incremental_updates() {
    let dir = TempDir::new().unwrap();
    fs::create_dir_all(dir.path().join("src")).unwrap();
    fs::write(
        dir.path().join("src/target.rs"),
        "/// locate the ancient rune.\npub fn locate_rune(x: u32) -> u32 { x + 1 }\n",
    )
    .unwrap();

    let mut engine = Engine::open(dir.path(), CodeRagConfig::default()).unwrap();
    engine.index(IndexOptions::default()).await.unwrap();

    let ann = SearchOptions {
        use_ann: Some(true),
        ..Default::default()
    };
    let hits = engine
        .search("ancient rune locate", ann.clone())
        .await
        .unwrap();
    assert!(!hits.is_empty());

    // Rewrite the file with entirely different vocabulary.
    fs::write(
        dir.path().join("src/target.rs"),
        "/// bake the modern cookie.\npub fn bake_cookie(x: u32) -> u32 { x + 2 }\n",
    )
    .unwrap();
    engine.index(IndexOptions::default()).await.unwrap();

    let hits = engine.search("ancient rune locate", ann).await.unwrap();
    assert!(
        hits.is_empty()
            || hits
                .iter()
                .all(|h| !h.chunk.content.contains("locate_rune")),
        "stale vector rows survived the re-index: {:?}",
        hits.iter().map(|h| &h.chunk.symbol).collect::<Vec<_>>()
    );
    let hits = engine
        .search(
            "modern cookie bake",
            SearchOptions {
                use_ann: Some(true),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert!(!hits.is_empty());
}
