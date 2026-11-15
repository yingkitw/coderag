# Agent Development Loop

This document defines the continuous improvement cycle for **coderag** — a Rust workspace providing a code-retrieval library (`crates/coderag`) and CLI (`crates/coderag-cli`): syntax-aware chunking → pluggable embeddings → persistent hybrid search.

## Project Structure

```
.
├── Cargo.toml                # workspace: members, shared dependency versions, release profile
├── README.md                 # user-facing docs: quickstart, config reference, scoring formula
├── AGENTS.md                 # this file: the development loop and doc contracts
├── TODO.md                   # roadmap, done list, brainstorm, dated scorecards
├── MEMORY.md                 # institutional knowledge: patterns, anti-patterns, invariants
├── .github/workflows/ci.yml  # fmt + clippy -D warnings + test on ubuntu/macos
└── crates/
    ├── coderag/                  # the library
    │   ├── Cargo.toml            # deps + lang-* feature gates (one per grammar)
    │   ├── src/lib.rs            # crate root: module docs, re-exports, supported_languages()
    │   ├── src/error.rs          # Error (thiserror) + Result alias
    │   ├── src/config.rs         # CodeRagConfig (.coderag/config.toml), validation, signature
    │   ├── src/tokens.rs         # identifier-aware tokenizer + FNV-1a (stability-critical)
    │   ├── src/langs.rs          # tree-sitter registry: definitions/containers/name fields
    │   ├── src/chunking.rs       # definition-aware chunker: attach, split, gap blocks
    │   ├── src/engine.rs         # incremental index pipeline + hybrid search + status
    │   ├── src/embedding/mod.rs      # EmbeddingProvider trait + provider_from_config
    │   ├── src/embedding/local.rs    # deterministic hashed-BoW offline provider
    │   ├── src/embedding/openai.rs   # OpenAI-compatible client: batching, retries
    │   ├── src/store/mod.rs      # SearchRecord type, module re-exports
    │   ├── src/store/ann.rs      # sqlite-vec vec0 mirror (feature ann): KNN, sync, hex ids
    │   ├── src/store/sqlite.rs   # SQLite schema, chunk/vector persistence, stats
    │   ├── examples/dump_chunks.rs   # debug utility: print chunks for one file
    │   ├── tests/engine.rs       # end-to-end: index/search/incremental/rebuild/top_k
    │   ├── tests/ann_recall.rs  # ANN vs brute-force recall + vec0 mirror sync (feature ann)
    │   ├── tests/retrieval_fixtures/main.rs  # golden retrieval benchmark (local + #[ignore] openai)
    │   ├── tests/retrieval_fixtures/corpus.rs  # multi-language golden corpus
    │   └── tests/docs_sync.rs    # doc-drift enforcement: AGENTS.md listing must match code
    └── coderag-cli/              # the `coderag` binary
        ├── Cargo.toml
        └── src/main.rs           # clap CLI: init/index/search/status/watch/config/languages,
                                  # human+json output, debounced notify watcher
```

`tests/docs_sync.rs` (run by `cargo test`) fails when this listing drifts from the real files. If it fails, update **this document**, not the test.

## The Loop

### 0. Consult MEMORY.md Before Starting
**CRITICAL**: Always begin each task by reading relevant sections of `MEMORY.md`. This prevents:
- Reinventing solutions to already-solved problems
- Repeating known mistakes and anti-patterns (see the chunking bug patterns)
- Missing established conventions for similar features
- Ignoring domain-specific invariants (vector-space compatibility, byte-coverage, determinism)

**How to consult MEMORY.md**:
1. Search for keywords related to your task (e.g. "containers", "signature", "cascade", "hermetic", "watch")
2. Read relevant pattern sections for context and proven approaches
3. Follow established conventions unless there's a clear reason to diverge
4. If MEMORY.md lacks relevant patterns, note this during the harvest step (Step 4)

### 1. Complete Remaining TODO Items
Pick the next highest-priority item from `TODO.md`. If no high-priority items remain, run the competitive intelligence step to seed new work. Implement with minimal, focused changes. Do not add speculative features.

### 2. Create Tests and Examples
For every new capability:
- Add inline `#[cfg(test)] mod tests` in the relevant source file
- Integration tests belong in `crates/coderag/tests/` — they must be **hermetic**: temp-dir projects, the offline `local` provider, zero network. Never make `cargo test` require an API key
- Retrieval changes need **ranking assertions** (expected top-1 path/symbol for a fixed corpus) — this project's equivalent of float-tolerance tests
- Any change that can alter stored vectors or the DB schema must respect the compatibility rules in MEMORY.md §2 (signature bump / `LOCAL_MODEL` bump / format version)
- New file? Add it to the structure tree above — `tests/docs_sync.rs` enforces it
- Dogfood: `cargo run -p coderag-cli -- index && cargo run -p coderag-cli -- search "..."` on this repo

### 3. Ensure `cargo test` Passes
Run the full gate — all three must be clean:
```bash
cargo test --workspace        # unit + integration + doc + docs_sync
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
```
Fix any failures before proceeding. CI runs the same gate on ubuntu + macos.

### 4. Harvest to MEMORY.md
After each completed feature, extract patterns and best practices:
- **Success patterns**: What worked well and should be repeated
- **Anti-patterns**: What to avoid in future implementations
- **Retrieval domain knowledge**: chunking heuristics, grammar quirks per language, embedding/vector-store pitfalls, ranking trade-offs
- **Rust patterns**: coderag-specific conventions (Error taxonomy, config serde defaults, feature-gated grammars, async trait objects)
- **Testing patterns**: how to assert rankings, how to build temp projects, what must stay hermetic

**Harvest Quality Checklist**:
- [ ] Added success patterns with code examples where useful
- [ ] Documented anti-patterns with what to avoid instead
- [ ] Included domain gotchas (grammar wrappers, vector-space mixing, watcher feedback loops)
- [ ] Referenced specific files/sections for future lookup
- [ ] Used existing MEMORY.md categories or created new ones if needed
- [ ] Made entries searchable with relevant keywords
- [ ] Noted any gaps found in MEMORY.md during Step 0 consultation

### 5. Loop Back to Step 1
Return to `TODO.md` and pick the next item. Repeat until the backlog is clear.

### 6. Audit and Optimize
After each batch of features, perform a quality pass:
- **Maintainability**: Are functions small and well-named? Is the module structure logical?
- **Docs alignment**: README/AGENTS/TODO/MEMORY match the actual implementation (no drift). Enforced by `tests/docs_sync.rs`: the AGENTS.md module listing is checked by `cargo test` — if it fails, update the DOC, not the test
- **Leanness**: Remove dead code, unused imports, and speculative abstractions
- **Wiring**: Ensure all new capabilities are exported from `lib.rs`, reachable from the CLI, and reflected in `coderag languages` (for grammars) or `--json` output (for data)
- **Small footprint**: Avoid unnecessary dependencies; prefer standard library or lightweight crates; new heavy deps (ANN indexes, ONNX runtimes) must go behind cargo features
- **Consistency**: Match existing style (Rust 2021, `thiserror` errors, serde-defaulted config, `//`-comment doc tone)
- **Retrieval fidelity**: Verify chunker/embedding changes against the ranking fixtures — coverage of the no-byte-loss invariant must stay intact
- **CRAP score**: Evaluate change risk with the CRAP metric (Change Risk Anti-Patterns):
  `CRAP = comp² × (1 − coverage)³ + comp`, where `comp` is cyclomatic complexity and coverage is the fraction of the code exercised by tests.
  - CRAP ≈ `comp`: code is well-covered — safe to change
  - CRAP ≫ `comp`: high complexity + low coverage — refactor by **adding tests first**, then simplifying
  - Keep new code under CRAP ≈ 30 (roughly `comp` ≤ 6 with full coverage); treat CRAP > 100 as a red flag
  - Tools: `cargo-tarpaulin` or `cargo-llvm-cov` for coverage, manual inspection (or `cargo-mutants`) for complexity
  - Note: CRAP identifies *where risk lives*; it does not mandate refactoring. For mature modules, always prefer adding coverage over rewriting (see Mature Module Policy below)
- **Honest reporting**: IndexReport/status numbers must stay truthful (see MEMORY.md §4 — the skipped-file accounting lesson)

### 7. Score the Codebase and Improve the Weakest Area
After the audit pass (Step 6) — or when the backlog runs dry — quantify overall health, then fix the single weakest dimension instead of guessing what to work on next.

**Scorecard** — score each dimension 0–5, evidence-based (never from vibes):

| Dimension | Evidence to check |
|---|---|
| Correctness & coverage | `cargo test` green; coverage % via `cargo-llvm-cov`/`cargo-tarpaulin`; no ignored/known-failing cases |
| Retrieval quality | Ranking fixtures pass (golden queries → expected top hits); dogfood searches on this repo return sensible results; hybrid weights not regressed |
| Maintainability | CRAP scores (Step 6), dead code, clippy warning count, function sizes |
| Docs alignment | README/AGENTS/TODO/MEMORY match the implementation; `docs_sync` green |
| Wiring & ergonomics | lib re-exports complete, every feature reachable from CLI, `--json` output for all data commands, examples compile |
| Footprint | Dependency count/weight, compile time, binary size; grammars stay feature-gated |

Total is /30. Record the dated scorecard in `TODO.md` so progress is visible across iterations.

**Improve the weakness**:
1. Pick the single lowest-scoring dimension (tie-break: the one with the highest user impact)
2. Turn the fix into a concrete `TODO.md` item with a success criterion and test plan
3. Execute it through the normal loop (Steps 1–4) — tests and memory harvesting are not optional
4. Respect the Mature Module Policy: weak scores on mature modules are fixed by **adding tests/coverage first**, not rewrites
5. Re-score that dimension after the fix; a step that doesn't move the score didn't happen

### 8. Competitive Intelligence
Research adjacent tools and identify capabilities this project lacks. Current watchlist:
- **Code search/RAG**: Sourcegraph Cody, GitHub code search, tabby-ml/tabby, continue-dev, Sourcegraph's chunker heuristics
- **Chunking**: `tree-sitter-chunker`-style crates, ast-grep's pattern matching, Cody's `--mast` tree-sitter chunker
- **Vector search**: `sqlite-vec` (loadable extension — natural fit for our store seam), `usearch`, `hnswlib` bindings, `instant-distance`, Lance
- **Embeddings**: `fastembed` (local ONNX models — candidate third provider), OpenAI v3 Matryoshka truncation behavior

Add the most valuable gaps to the `TODO.md` brainstorming section with a note on fit (e.g. sqlite-vec slots in behind `store/sqlite.rs` without touching the engine).

#### Web Research with the `web2md` MCP Server
All web research in this loop (competitive intelligence, docs lookup, crates.io version checks) should go through the **`web2md`** MCP server (`https://web2md.net/api/mcp`) instead of ad-hoc HTML fetching.

**Why**: web2md's `fetch` tool is deterministic (no LLM in the loop), strips boilerplate, and returns token-efficient Markdown plus structured metadata — much cheaper and more reliable for agents than raw HTML.

**Tool**: `fetch` — one tool, one required argument:

```
fetch(url, main_content=false, max_length=null, format=null, ...)
```

**Conventions**:
- **Docs/API pages** (docs.rs, crates.io, GitHub READMEs): `fetch(url, main_content=true)` to skip nav/footers
- **Long pages**: pass `max_length` (bytes); output is cleanly `[truncated]` instead of blowing the context window
- **Link discovery** (finding related crates/benchmarks): `fetch(url, format="links")` returns a structured link list instead of prose
- **Tables** (benchmark comparisons, version matrices): kept as GFM by default (`include_tables=true`)
- Use `only_with_metadata=true` when a page must have a `title` + `published_date` (e.g. citing a dated source)
- If web2md is unavailable in the current environment, fall back to any built-in fetch tool — but note it in `MEMORY.md` only if the fetched content produced a durable pattern

### 9. Update Documentation
Keep all project docs aligned with the current implementation. Root docs (required):

- **`README.md`**: Quick start, CLI usage, config reference, scoring formula, honest limits
- **`TODO.md`**: Mark completed items, move them to Done, keep brainstorming current, record dated scorecards
- **`MEMORY.md`**: Harvested patterns, domain knowledge, compatibility invariants
- **`AGENTS.md`** (this file): The loop itself and the structure listing

Update **`AGENTS.md`** if the loop evolves, and whenever files are added or removed (`docs_sync` will remind you).

## Memory System (MEMORY.md)

### Purpose
`MEMORY.md` is the institutional knowledge repository that accelerates development by:
- **Preventing wheel reinvention**: Reuse proven patterns instead of guessing
- **Domain knowledge preservation**: Capture retrieval-engineering rules that are counter-intuitive (grammar wrapper nodes, vector-space compatibility, watcher feedback loops)
- **Onboarding acceleration**: New contributors (human or AI) can understand patterns quickly
- **Quality consistency**: Ensure all features follow established conventions

### Structure
Organize `MEMORY.md` into these sections:

#### 1. Chunking Patterns
- Tree-sitter node-kind selection per language (definitions vs. containers)
- Grammar wrapper nodes (`declaration_list`, `class_body`, `block`, `export_statement`) that must be descended into
- Comment/decorator attachment rules and the span-extension approach
- Oversized-definition decomposition and statement packing
- The no-byte-loss invariant and gap blocks

#### 2. Embedding & Vector-Space Patterns
- Provider trait conventions (unit vectors, order preservation, dimension contracts)
- Determinism requirements: FNV-1a and the `LOCAL_MODEL` bump rule
- Index signature (provider:model:dimensions) and rebuild semantics
- OpenAI-compatible API handling: batching, retryable statuses, index re-ordering

#### 3. Storage Patterns
- SQLite schema, WAL/FK-cascade setup, transactional file replacement
- f32 little-endian BLOB encoding and dimension guards on decode
- Meta table as version/signature store; migration policy

#### 4. Indexing Pipeline Patterns
- Walk/ignore configuration and skip-dir policy
- Content-hash diffing and honest report accounting
- Progress callbacks; where async meets blocking (SQLite, notify)

#### 5. Search & Ranking Patterns
- Hybrid scoring formula and its knobs
- Per-file diversity caps, symbol bonus, min-score filtering
- Parallel scoring with rayon

#### 6. Testing Patterns
- Hermetic offline tests via the local provider
- Temp-dir project fixtures; ranking assertions
- `docs_sync` enforcement; pitfalls (e.g. `unwrap_err` needs `Debug`)

#### 7. CLI Patterns
- clap structure, human vs. `--json` output conventions
- Watch mode: std-mpsc → debounce → tokio-mpsc bridge; feedback-loop avoidance

### Maintaining MEMORY.md Quality

**When to Update MEMORY.md**:
- After completing any non-trivial feature or bug fix
- When discovering a new pattern or anti-pattern
- After resolving a tricky debugging session (grammar quirks, async wiring, SQLite locking)
- When establishing new conventions
- When a compatibility rule is discovered or tightened

**How to Write Good MEMORY.md Entries**:
1. **Be specific**: Reference actual files and sections where patterns occur
2. **Show examples**: Include minimal code snippets demonstrating the pattern
3. **Explain why**: Don't just say what—explain the reasoning behind the pattern
4. **Link related patterns**: Cross-reference related entries with `[[PatternName]]`
5. **Keep it searchable**: Use keywords that future developers will search for
6. **Date entries**: Add dates so readers know how current the information is

**Signs MEMORY.md Needs Attention**:
- Same questions or patterns coming up repeatedly in development
- Agents solving problems that should be documented
- Frequent bugs in similar areas (indicates missing anti-pattern documentation)
- Index invalidation surprises (indicates missing compatibility-rule documentation)

**MEMORY.md Hygiene Routine** (run monthly):
- [ ] Review for outdated information (deps, grammar versions, API changes)
- [ ] Consolidate redundant entries
- [ ] Add missing patterns from recent work
- [ ] Update cross-references if structure changed
- [ ] Verify all file references still exist
- [ ] Check for learnings that weren't captured

## Principles

- **Simplicity over flexibility**: Solve the problem at hand, not every hypothetical future problem
- **CRAP score**: see Step 6 — keep change risk low via complexity + coverage
- **Surgical changes**: Touch only what you must; clean up only your own mess
- **Goal-driven**: Every change should have a verifiable success criterion
- **Test before ship**: No feature is complete until it has passing tests
- **Docs are code**: Documentation drift is a bug (and `docs_sync` makes it a *failing* one)
- **Retrieval fidelity**: Never silently lose bytes when chunking; never silently mix vector spaces; never lie in reports
- **Hermetic by default**: `cargo test` must never need network, keys, or fixtures outside the repo
- **Memory first**: Consult `MEMORY.md` before starting work. If it lacks relevant information, note the gap and fill it during harvest
- **Pattern harvesting**: After success, update `MEMORY.md` so others benefit from your experience

## Mature Module Policy

**Mature modules are effectively frozen.** Do not modify them as part of feature work, cleanup, or "drive-by" refactoring — even if the code looks improvable.

**A module is considered mature when ALL of the following hold:**
- It is stable and correct: no open bugs, no known retrieval regressions
- It is well-covered by tests that have not needed changes for several iterations
- Its behavior is a **compatibility surface** (existing on-disk indexes depend on it)
- It has documented, validated behavior (MEMORY.md entries, ranking fixtures)

**Current compatibility surfaces** (treat as mature from day one):
- `tokens.rs` — tokenizer output feeds stored local-provider vectors; any change silently invalidates every existing offline index (see the `LOCAL_MODEL` bump rule)
- `embedding/local.rs` — same class of risk (hashing/normalization)
- `store/sqlite.rs` — schema and BLOB encoding; existing `index.db` files must keep opening

**Rules**:
1. **Extend, don't edit**: Add new capabilities in new modules or new functions; never reshape a mature module's public API to fit a new feature
2. **Wrap, don't rewrite**: Prefer adapters/new types over altering mature internals
3. **Bug fixes only**: A change to a mature module requires a concrete, reproducible bug or test failure — fix minimally and add a regression test; no opportunistic restructuring in the same commit
4. **Bump, don't break**: Changes that intentionally alter stored representations must escape the signature guard deliberately: bump `LOCAL_MODEL` (e.g. `hashed-bow-v2`), the index `format_version`, or the config signature components — so old indexes demand `--rebuild` instead of degrading silently
5. **Prefer coverage over rewriting**: If a mature module has high complexity/low coverage (high CRAP), respond by adding tests, not by rewriting
6. **Update docs, not code**: If behavior is confusing, document it in MEMORY.md rather than "fixing" it

## File Positioning and Value

### README.md
- **Value**: User-facing documentation and project overview
- **Audience**: Users, contributors, stakeholders
- **Position**: Entry point for anyone discovering the project
- **Focus**: Features, quick start, CLI usage, config reference, honest limits

### TODO.md
- **Value**: Feature roadmap and backlog management
- **Audience**: Development team (human and AI agents)
- **Position**: Development planning and prioritization
- **Focus**: What to build next, what's done, brainstorm, dated scorecards

### MEMORY.md
- **Value**: Institutional knowledge and pattern library
- **Audience**: Development team (accelerates onboarding and consistency)
- **Position**: Development acceleration and quality consistency
- **Focus**: Proven patterns, domain knowledge, compatibility invariants
- **Update**: Must be updated after each completed feature to capture patterns and lessons learned

### AGENTS.md (this file)
- **Value**: Development process and workflow definition
- **Audience**: AI agents and human developers following the development loop
- **Position**: Process automation and continuous improvement
- **Focus**: How we work, the loop, memory system, structure contract
- **Update**: When the loop evolves, or when files are added/removed (enforced by `docs_sync`)

## How These Files Work Together

1. **README.md** tells stakeholders what the project is and how to use it
2. **TODO.md** tells developers what to build next (driven by scorecards and competitive intelligence)
3. **AGENTS.md** tells agents how to work through the TODO items with quality and memory
4. **MEMORY.md** captures what we learned so we don't repeat mistakes and provides proven patterns to accelerate development

The loop reinforces these files:
Consult MEMORY.md → Complete TODO → Test → Harvest to MEMORY → Optimize → Score & improve weakest → Research → Update TODO

This creates a flywheel of continuous improvement with institutional knowledge preservation. **MEMORY.md is both the starting point (consult before work) and the destination (harvest after work), creating a virtuous cycle of learning and improvement.**
