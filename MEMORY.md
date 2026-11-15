# coderag — MEMORY

Institutional knowledge: patterns, anti-patterns, and compatibility
invariants. Consult before starting work (AGENTS.md Step 0); harvest into
after completing it (Step 4). Entries are dated.

---

## 1. Chunking Patterns

### Grammar wrapper nodes must be containers (2026-11-15)
Definitions in most grammars are **not** direct children of the node you're
scanning; they sit inside a body wrapper. If the wrapper isn't in
`LangDef::containers`, oversized parents fall through to statement-splitting
and symbols are lost.

| Language | Wrapper kind(s) to declare as containers |
|---|---|
| Rust | `declaration_list` (impl + mod bodies) |
| Python | `block` (class/fn bodies), `decorated_definition` |
| JS/TS/TSX | `class_body`, `export_statement` |
| Java | `class_body`, `interface_body`, `enum_body_declarations` |
| C/C++ | `declaration_list`, `field_declaration_list` |

Symptom when missing: chunks with kind `impl_item`/parent kind and the
parent's symbol repeated, no method symbols.
See `langs.rs` + the `chunks_rust_definitions` test in `chunking.rs`.

### Attach leading comments by extending the span, never by re-rooting (2026-11-15)
**Anti-pattern (actual bug here):** `attach_leading` originally *returned the
comment node itself*, so the "definition chunk" became a lone comment and the
definition body dropped into a gap block — silently degrading retrieval.
**Pattern:** keep the definition node authoritative; walk `prev_sibling()`
backwards over attachable kinds (`comment`, `line_comment`, `decorator`, ...)
while contiguous (no blank line, size cap) and record `start =
prev.start_byte()`. Attachment is span arithmetic, not tree surgery.

### The no-byte-loss invariant (2026-11-15)
After section collection, a gap pass (`gap_blocks`) emits `block` chunks for
any span ≥ `min_chunk_chars` not covered by a definition. Exception: the
*leading* preamble gap is only emitted when it exceeds `preamble_chars`
(else it's carried as context on every chunk — don't double-index it).
Tests assert both directions: symbols found **and** no standalone preamble
chunk. Any chunker change must keep this true.

### Small containers stay whole (2026-11-15)
A class/impl below `max_chunk_chars` is one chunk containing its methods —
that's correct, not a bug. Test expectations must be written against the
*size threshold*, not against "methods should always be separate" (a test
written the other way was wrong twice).

### Symbol extraction fallback (2026-11-15)
Field names first (`name_fields`, e.g. Rust also needs `"type"` for
`impl_item`), then depth-≤3 BFS for identifier-ish kinds — this is what
rescues Go `type_declaration > type_spec > name` and C declarators.

## 2. Embedding & Vector-Space Patterns

### The signature guard (2026-11-15)
`provider:model:dimensions` is stored in the `meta` table at first index.
A mismatch fails fast with a `--rebuild` hint (`Error::IndexMismatch`)
instead of mixing vector spaces. Never bypass `ensure_signature` when adding
index-writing paths.

### The LOCAL_MODEL bump rule (2026-11-15) — COMPATIBILITY INVARIANT
The local provider's vectors are a pure function of `tokens::tokenize` +
FNV-1a + dimension. The signature only records `local:hashed-bow-v1:256` —
so changing tokenization or hashing **silently invalidates every offline
index** without tripping the guard. Any semantic change to `tokens.rs` or
`embedding/local.rs` must bump `LOCAL_MODEL` (→ `hashed-bow-v2`) so old
indexes demand a rebuild. This is why those two files are day-one mature
modules ([[Mature Module Policy]]).

### FNV-1a over std hashing (2026-11-15)
`DefaultHasher` stability is not guaranteed across Rust versions — never use
it for anything stored. The hand-rolled `fnv1a64` has reference test vectors
(`fnv_matches_reference_vectors`).

### OpenAI-compatible clients (2026-11-15)
- Sort response `data` by its `index` field before use; the API does not
  promise order.
- Retry only 429/5xx/transport errors, exponential backoff; hard-fail on
  other 4xx with the body included.
- Batch ≤ 2048 (API cap); validate returned dimensionality against config.
- Works with Ollama/vLLM/LM Studio via `OPENAI_BASE_URL` — never hardcode
  `api.openai.com` in code paths.

### async trait objects (2026-11-15)
`#[async_trait]` on `EmbeddingProvider`; **no generic methods** or the trait
stops being dyn-compatible (learned: `impl Into<String>` on `embed_one`
broke `Arc<dyn ...>`). Owned `Vec<String>` params avoid lifetime pain.

## 3. Storage Patterns

- Pragmas per connection: `journal_mode=WAL`, `synchronous=NORMAL`,
  `foreign_keys=ON`. FK `chunks.path → files.path ON DELETE CASCADE` makes
  "delete file = delete chunks" atomic.
- File re-index = one transaction: upsert `files`, `DELETE chunks WHERE
  path`, insert new chunks. Readers never see half a file.
- Vectors: little-endian `f32` BLOBs (`to_le_bytes`/`from_le_bytes`);
  decode validates length against expected dims and points at `--rebuild`
  in the error text.
- `meta` table is the version store (`format_version`,
  `embedding_signature`, `created_at`). Schema changes bump
  `INDEX_FORMAT_VERSION` and need a migration path (TODO item).
- rusqlite: reads take `&self`, writes take `&mut self` (transactions need
  `&mut Connection`) — that's why `Engine::index` is `&mut self` while
  `search` is `&self`.
- Dynamic SQL predicates: build `Vec<&dyn rusqlite::ToSql>` — but note the
  borrow gotcha: an `if let Some(x) = param` binding is scoped to the block,
  so pushing `&x` into a longer-lived params vec fails borrowck. Bind to a
  function-scoped local first (see `all_records_filtered`). For owned
  params (e.g. id blobs) use `Vec<Box<dyn ToSql>>` and borrow for the call
  (see `records_by_ids`). (2026-11-15)
- Store extension pattern (extend-don't-edit, 2026-11-15): new query shapes
  go through new public methods over a shared private `records_query(sql,
  params, dimensions)`; existing public signatures stay untouched so mature
  callers never break.

### sqlite-vec mirror gotchas (2026-11-15, feature `ann`)

- **Registration race**: `sqlite3_auto_extension` only affects connections
  opened *after* registration. Register in `SqliteStore::open` **before**
  `Connection::open` — registering in `Engine::open` after the store exists
  breaks concurrently-opened engines with `no such module: vec0`.
- **Dimension is baked into the vec0 DDL** (`float[N]` literal) and
  survives `clear()`; switching embedding dims corrupts inserts. Track the
  dim in `meta` (`ann_dimensions`), drop+recreate the table on change, and
  backfill only rows of the right byte width
  (`length(embedding) = 4·dim`).
- The vec0 mirror is a **derived table**: every write/delete happens inside
  the same transaction as its `chunks` siblings (`replace_file_chunks`,
  `remove_file`, `clear`). Never allow silent divergence — an orphaned
  vector row merely hydrates to nothing; a missing one silently shrinks
  recall.
- Unit tests that touch `replace_file_chunks` in `ann` builds must call
  `enable_ann(dims)` first — mirror the engine's open sequence.
- KNN form: `SELECT chunk_id FROM vchunks WHERE embedding MATCH ? AND k =
  ?` with the query as a compact LE-f32 BLOB; ids are hex-encoded TEXT
  primary keys (blob PKs are not supported by vec0 in 0.1.x). Use ANN
  distance **only for candidate ordering** — the engine recomputes exact
  cosine from stored embeddings during rerank, so the L2/L2² question never
  affects results.
- clippy demands explicit `transmute` annotations; the SQLite extension
  entry-point type is spelled via `rusqlite::ffi::{sqlite3,
  sqlite3_api_routines}` (`rusqlite::libsqlite3_sys` is not a public path).

## 4. Indexing Pipeline Patterns

### Honest report accounting (2026-11-15)
**Anti-pattern (actual bug here):** unsupported-language files were counted
as "new" on *every* run because they never get a `files` row. **Pattern:**
classify at diff time — unsupported/large files are counted as skipped in
the same pass that decides new/changed, and if such a file previously *was*
indexed (extension changed / grammar feature removed) remove it and count
`files_removed`. Verified behavior: no-op run reports
`0 new, 0 changed, N unchanged, 0 removed, 0 embedded`.

### Double-read is fine (2026-11-15)
The diff phase reads a file to hash it, the chunk phase reads it again. OS
page cache makes the second read ~free; passing bytes through would bloat
peak memory on full re-indexes. Keep it simple.

### Walker policy (2026-11-15)
`ignore::WalkBuilder` with gitignore+global+exclude; `filter_entry` rejects
any path component in `skip_dirs ∪ {".git", ".coderag"}`. Always skip
`.coderag` or the watcher re-indexes its own database (feedback loop).

## 5. Search & Ranking Patterns

- Hybrid score: `(1−w)·cosine + w·lexical + symbol_bonus`, clamped to 1.0;
  `w = search.lexical_weight` (0.15 default). Lexical = unique query-token
  coverage of the chunk's tokens; bonus = 0.1 × symbol-token coverage.
- Per-file diversity cap (`max_per_file`, default 3) applied while walking
  the sorted list — cheap and effective against file flooding.
- Score with rayon `par_iter`, then sort. Zero-vector guard in
  `cosine_similarity` (empty files → score 0, never NaN).
- `local` provider is lexical, not neural — ranking assertions should
  compare *relevant vs irrelevant* documents, not absolute scores.
- **Filter pushdown** (2026-11-15): cheap predicates (language, symbol
  presence) belong in SQL (`all_records_filtered`) so non-matching
  embeddings are never loaded; path globs are applied in memory after load
  (SQL `LIKE` can't express real globs). Invalid globs surface as
  `Error::Config`, not panics; impossible-but-valid globs return empty
  results. Reused `globset` (already in the tree via `ignore`) instead of
  adding a matcher.
- **Two-stage divergence** (2026-11-15, feature `ann`): reranking only a
  cosine-ordered candidate pool (ANN path) loses hits that the hybrid
  signal (lexical + symbol bonus) would promote from low cosine rank.
  Measured recall@10 ≈ 0.94 vs brute force at pool 500/1200 on the
  synthetic corpus — a property of two-stage retrieval, not mirror
  corruption. Golden queries get exact parity because pool ≥ corpus there.
  Parity assertions must be **tie-aware**: equal scores order by input
  sequence, so compare with `min_score = f32::EPSILON` to drop the
  zero-score tail where ordering is table-order noise.

### Short-chunk cosine bias (2026-11-15, found by the golden benchmark)
Raw hashed-BoW cosine favors **short chunks**: a 14-token struct sharing
3/5 query tokens (cos ≈ 0.80) beats a 60-token method chunk containing
5/5 (cos ≈ 0.64), because doc-vector norm grows with every extra token.
Observed: `HttpRetryPolicy` struct outranked `compute_backoff_delay` for
its own query. Consequences:
- Golden symbol assertions use a fixed top-3 window while path assertions
  stay strict top-1 (see `tests/retrieval_fixtures/main.rs`).
- Real fix candidate: **IDF-weight the lexical signal at search time**
  (df computable from indexed chunks) and/or length-normalized cosine
  correction — in TODO brainstorm; don't tweak `search.lexical_weight`
  per-query to paper over it.

## 6. Testing Patterns

- **Hermetic**: everything runs offline via the local provider; temp-dir
  projects (`tempfile`); the only key-related test *asserts failure* when
  `OPENAI_API_KEY` is unset.
- **Ranking assertions**: fixed corpus (fib.rs + http.rs style), assert
  top-1 path and symbol — see `tests/engine.rs`
  (`indexes_and_retrieves_relevant_code`).
- **Pitfall**: `Result::unwrap_err()` needs `T: Debug` — `Engine` isn't
  Debug; `match` on the result instead (bit us once).
- **Multi-file integration tests**: `mod corpus;` inside a flat
  `tests/foo.rs` does **not** resolve to `tests/foo/corpus.rs` — use the
  `tests/foo/main.rs` + sibling-module layout (as
  `tests/retrieval_fixtures/main.rs` does).
- **Golden benchmark conventions** (`tests/retrieval_fixtures/`): path
  assertions strict top-1 / loose top-3; symbol assertions get a fixed
  top-3 window (short-chunk cosine bias, §5); keep ≥50% of goldens strict
  (enforced by assertion) so the harness can't rot into all-loose.
  Corpus files are sized to exercise oversized-decomposition (impl >1600
  bytes) — check with `examples/dump_chunks.rs <file>` when a symbol
  golden misses.
- `tests/docs_sync.rs` enforces the AGENTS.md file listing both directions;
  when adding/removing files update the tree, not the test.
- Incremental tests assert the *counts* (`files_changed`,
  `chunks_embedded < first_run`) — cheap and they catch diff-logic
  regressions exactly.

## 7. CLI Patterns

- `tokio::runtime` built explicitly in `main`; clap args parsed before.
- Watch mode bridge: `notify::recommended_watcher(std mpsc)` → debounce
  thread (`recv_timeout` quiet window) → `tokio::sync::mpsc` → async loop
  with `select!` on `ctrl_c()`. The watcher thread owns the watcher and an
  owned `PathBuf` root (borrowed `&Path` can't escape into `'static`
  threads).
- Human output to stdout, progress to stderr (`\r`-overwrite for embedding
  progress); `--json` on every data command serializes library types
  (they all derive `Serialize` — keep it that way).
- `--project` is canonicalized once in `main`; everything downstream deals
  in absolute paths + `/`-separated rel paths for storage keys.

## 8. Environment Notes

- tree-sitter 0.25 + grammar crates 0.23–0.25: register grammars with
  `LANGUAGE.into()` (`tree_sitter::Language` from `LanguageFn`); TSX is
  `LANGUAGE_TSX`, not the TS grammar. (2026-11-15)
- `ignore`'s `DirEntry::file_type()` returns `Option`, not `Result`. (2026-11-15)
- Grammar crates are matched to the runtime via the `tree-sitter-language`
  ABI; when bumping the runtime, check grammar compat (see Step 8 research).
- Chunker edge case: a top-level `const X: &str = r#"..."#` (raw string
  blob) produces a header-only `const_item` chunk plus line-window chunks
  of the literal — harmless (no bytes lost) but noisy for fixture-style
  files; candidate refinement if it ever hurts retrieval. (2026-11-15)
