# coderag — TODO

Prioritized backlog. Work items through the loop in `AGENTS.md`
(consult `MEMORY.md` first; harvest into it after). Dated scorecards live at
the bottom.

## Next

1. **Index migrations** — `format_version` bump path: schema diff runner so
   old `index.db` files upgrade in place instead of demanding `--rebuild`
   for benign changes (e.g. adding an indexed column, the `vchunks`
   companion table).
   *Success criterion:* migration test: v1 db opens and serves correct results after upgrade.
2. **More grammars** — ruby, c#, php, kotlin, zig, swift, lua (each behind
   `lang-*`; update `langs.rs` + chunker tests + `coderag languages`).
   *Success criterion:* per-language chunker test asserting symbols found; docs_sync green.

## Brainstorm (not scheduled)

- **IDF-weighted lexical scoring** — weight query tokens by inverse
  document frequency computed over indexed chunks at search time (df is
  free; the provider can't do this because embeddings are context-free).
  Motivated by the short-chunk cosine bias found by the golden benchmark
  (MEMORY.md §5); would also damp ubiquitous tokens (`pub`, `fn`, `file`).
- **Length-normalized cosine correction** — alternative/complement to IDF
  for the same bias.

- **fastembed provider** — local ONNX embeddings (bge-small/all-MiniLM) as a
  third `EmbeddingProvider`; would make "offline" mean *neural* offline.
  Heavy dep → cargo feature.
- **MCP server** — expose `index`/`search`/`status` as MCP tools (stdio
  JSON-RPC) so editor agents can query the index directly.
- **Cross-encoder reranker** — rerank top-50 hybrid hits (feature-gated).
- **Per-provider embedding cache** — key embeddings by blake3(chunk_id +
  provider:model:dim) so switching providers back and forth doesn't re-bill.
- **Repo-map summaries** — one LLM-or-heuristic summary chunk per file/dir as
  extra retrieval signal for "where does X live" queries.
- **Query language** — quoted phrases, `symbol:foo`, negation in the query box.
- **Incremental tree editing** — reuse previous tree-sitter trees via edit
  scripts for very large files (only worth it if profiling says so).
- **Windows CI leg** — notify backends differ (ReadDirectoryChangesW); needs
  a smoke test.
- **Benchmark suite** — `cargo bench`/criterion for chunking and search
  latency on a pinned corpus; catches perf regressions, feeds the Footprint
  scorecard.

## Done

- 2026-11-15 — Release: `coderag 0.1.0` published to crates.io (Apache-2.0,
  verified from a scratch consumer project: add → index → search). The CLI
  crate is repo-only (`publish = false`); its 0.1.0 upload was removed by
  crates.io itself (index entry carries a delete-marker), so the registry
  carries the library alone. Docs updated (install paths, license).
- 2026-11-15 — Housekeeping: license switched MIT → Apache-2.0 (LICENSE
  added, manifests + README updated, repository URL pointed at
  yingkitw/coderag); GitHub Actions workflow removed — the Step 3 gate is
  now local-only and must run before every push; AGENTS.md and MEMORY.md
  are local-only (untracked + gitignored) — TODO.md stays published as the
  public roadmap; .gitignore extended (build artifacts, index DBs, env
  files, editor dirs).
- 2026-11-15 — ANN backend (feature `ann`, off by default): sqlite-vec 0.1.9
  vec0 mirror transactionally synced with `chunks` (write/delete in the
  same tx), hex TEXT PKs, meta-tracked dimensionality with drop/recreate on
  change, KNN candidate retrieval + hybrid rerank with recomputed exact
  cosine. Research: chose sqlite-vec over usearch (same-file transactions,
  pure C, no separate index to sync). Acceptance: golden queries return
  identical results to brute force (parity test); synthetic 1.2k-chunk
  corpus: recall@10 0.94 / top-1 0.85 vs brute force — documented as
  two-stage rerank divergence (MEMORY §5), floor 0.90. CI runs the ann
  feature matrix. Two real bugs found by its own tests: extension
  registration race, vec0 DDL dimension pinning.
- 2026-11-15 — Search filters: `--lang` (exact language, SQL-level
  pushdown), `--path <glob>` (globset, matched against relative paths,
  invalid globs are config errors), `--symbols-only` (SQL `symbol IS NOT
  NULL`). Library surface: `SearchOptions { language, path_glob,
  symbols_only }`; store gained `all_records_filtered` via a private
  `records_query` (extend-don't-edit). Filter integration tests against the
  golden corpus; dogfooded on this repo.
- 2026-11-15 — Golden retrieval benchmark: multi-language corpus (9 files,
  rust/python/ts/go) with 15 golden queries (13 strict top-1 path), fixed
  top-3 symbol window, ≥50%-strict honesty gate, multi-language index
  check, and a key-gated `#[ignore]` OpenAI hits@1/hits@3 report. Found and
  documented the short-chunk cosine bias (MEMORY.md §5) → IDF brainstorm
  item. `examples/dump_chunks.rs` now takes a real file argument.
- 2026-11-15 — Initial release: workspace (lib + CLI), tree-sitter chunker for
  9 languages with oversized-definition decomposition and no-byte-loss gap
  blocks; local (offline, deterministic) + OpenAI-compatible embedding
  providers with batching/retries; SQLite store (WAL, FK cascade, signature
  guard); incremental indexing (blake3 diff, honest accounting); hybrid
  search (cosine + lexical + symbol bonus, per-file diversity cap); watch
  mode; 24 tests; CI (fmt/clippy/test, ubuntu+macos); README/AGENTS/TODO/
  MEMORY + docs_sync enforcement.

## Scorecards

| Date | Correctness | Retrieval | Maintainability | Docs | Wiring | Footprint | Total /30 | Weakest → action |
|---|---|---|---|---|---|---|---|---|
| 2026-11-15 | 4 (24 tests green, coverage unmeasured) | 3 (dogfood good; no golden fixture yet) | 4 (clippy-clean, chunker is the hot spot) | 5 (docs_sync enforced) | 4 (json everywhere; no search filters) | 4 (dep-light; 9 grammars default-on) | 24 | Retrieval → item Next#1 |
| 2026-11-15 (b) | 4 (26 tests green, coverage unmeasured) | 4 (golden harness: 13/15 strict top-1, dogfood good; openai hits@k unmeasured — needs a key) | 4 (clippy-clean, chunker is the hot spot) | 5 (docs_sync enforced) | 4 (json everywhere; no search filters) | 4 (dep-light; 9 grammars default-on) | 25 | Wiring → item Next#1 (search filters) |
| 2026-11-15 (c) | 4 (28 tests green, coverage unmeasured) | 4 (golden harness: 13/15 strict top-1, dogfood good; openai hits@k unmeasured — needs a key) | 4 (clippy-clean, chunker is the hot spot) | 5 (docs_sync enforced) | 5 (filters + json everywhere; library options mirror CLI flags) | 4 (dep-light; globset reused via ignore's tree; 9 grammars default-on) | 26 | Correctness/Maintainability/Retrieval tie at 4 → coverage run (llvm-cov) next, then IDF lexical (feeds both Retrieval and Correctness) |
| 2026-11-15 (d) | 4 (27 default + 30 ann tests green, coverage unmeasured) | 4 (same as (c); ANN parity proven on goldens, 0.94 recall on synthetic documented) | 4 (clippy-clean on both feature sets; one audited unsafe site) | 5 (docs_sync enforced; CI covers feature matrix) | 5 (filters, --no-ann, ann line in status) | 4 (default build unchanged: sqlite-vec fully optional; ann build +1 C dep) | 26 | Coverage measurement (llvm-cov) is the un-run evidence → next: run it, then act on the weakest measured dimension |
| 2026-11-15 (e) | 4 (unchanged; gate re-run green post-housekeeping) | 4 (unchanged) | 4 (unchanged) | 5 (docs_sync enforced locally; NOTE: hosted CI removed — gate is a pre-push manual step now, slight regression in enforcement) | 5 (unchanged) | 4 (unchanged) | 26 | Docs enforcement weakened by CI removal → compensate: run the full gate (Step 3) before every push; revisit hosted CI if lapses occur |
| 2026-11-15 (f) | 4 (unchanged; 27 default + 30 ann tests green) | 4 (unchanged) | 4 (unchanged) | 5 (license/install docs accurate; library published and consumer-verified) | 5 (unchanged) | 4 (crates.io carries the library only; CLI build-from-repo) | 26 | Coverage still unmeasured → next action remains a coverage run, then the weakest measured dimension |
