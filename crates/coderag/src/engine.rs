//! The indexing + retrieval engine.
//!
//! [`Engine`] ties the pieces together:
//!
//! ```text
//!  walk (gitignore-aware) ──► blake3 hash ──► diff vs store
//!        │ unchanged → skip                    │ new / changed
//!        ▼                                     ▼
//!   remove deleted files            tree-sitter chunking
//!                                           ▼
//!                              embed (batched, retried)
//!                                           ▼
//!                            SQLite: replace file's chunks
//! ```
//!
//! Search embeds the query with the same provider and scores every chunk with
//! a hybrid signal: `(1-w) * cosine + w * lexical` plus a small exact-symbol
//! bonus, with a per-file diversity cap.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;
use serde::Serialize;

use crate::chunking::{Chunk, ChunkOptions, Chunker};
use crate::config::CodeRagConfig;
use crate::embedding::{provider_from_config, EmbeddingProvider};
use crate::error::{Error, Result};
use crate::store::SearchRecord;
use crate::store::SqliteStore;
use crate::tokens;

/// Version of the on-disk index format.
pub const INDEX_FORMAT_VERSION: &str = "1";

/// Live progress events emitted during [`Engine::index`].
#[derive(Debug, Clone)]
pub enum Progress {
    /// File discovery finished with this many candidates.
    Discovered {
        /// Candidate file count.
        files: usize,
    },
    /// The incremental diff is known.
    Plan {
        /// Chunks queued for embedding.
        to_embed: usize,
        /// Files dropped from the index.
        removed: usize,
    },
    /// An embedding batch completed.
    Embedding {
        /// Chunks embedded so far.
        done: usize,
        /// Total chunks to embed.
        total: usize,
    },
}

/// Callback invoked on progress events.
pub type ProgressCallback = Arc<dyn Fn(Progress) + Send + Sync>;

/// Options for [`Engine::index`].
#[derive(Default, Clone)]
pub struct IndexOptions {
    /// Wipe the index before indexing (required when the embedding setup
    /// changed).
    pub rebuild: bool,
    /// Optional progress sink.
    pub progress: Option<ProgressCallback>,
}

/// Result of an indexing run.
#[derive(Debug, Clone, Default, Serialize)]
pub struct IndexReport {
    /// Candidate files found after ignore rules.
    pub files_seen: usize,
    /// Files indexed for the first time.
    pub files_new: usize,
    /// Files whose content hash changed.
    pub files_changed: usize,
    /// Files removed from the index (deleted on disk).
    pub files_removed: usize,
    /// Files whose hash matched the index — zero work done.
    pub files_unchanged: usize,
    /// Files skipped because no grammar is compiled in.
    pub files_skipped_unknown: usize,
    /// Files skipped for exceeding `index.max_file_bytes`.
    pub files_skipped_large: usize,
    /// Files skipped for not being valid UTF-8.
    pub files_skipped_binary: usize,
    /// Chunks currently stored for the re-indexed files.
    pub chunks_indexed: usize,
    /// Chunks sent through the embedding provider this run.
    pub chunks_embedded: usize,
    /// Per-file error messages (non-fatal).
    pub errors: Vec<String>,
    /// Whether the index was wiped first.
    pub rebuilt: bool,
    /// Wall-clock duration in milliseconds.
    pub elapsed_ms: u64,
}

/// Options for [`Engine::search`]; `None` falls back to config.
#[derive(Default, Clone)]
pub struct SearchOptions {
    /// Overrides `search.top_k`.
    pub top_k: Option<usize>,
    /// Overrides `search.max_per_file`.
    pub max_per_file: Option<usize>,
    /// Overrides `search.min_score`.
    pub min_score: Option<f32>,
    /// Restrict results to one language id (`rust`, `go`, ...).
    pub language: Option<String>,
    /// Restrict results to chunks whose path matches this glob
    /// (e.g. `src/**/*.rs`). Matched against `/`-separated relative paths.
    pub path_glob: Option<String>,
    /// Only return chunks with a symbol (definitions), skipping `file`
    /// and `block` chunks.
    pub symbols_only: bool,
    /// Force the candidate strategy for this query: `Some(true)` requires
    /// the sqlite-vec mirror (feature `ann`), `Some(false)` forces the
    /// brute-force scan, `None` uses ANN when compiled in. SQL filters
    /// (`language`, `symbols_only`) always use the scan path regardless.
    pub use_ann: Option<bool>,
}

/// A scored retrieval hit.
#[derive(Debug, Clone, Serialize)]
pub struct SearchHit {
    /// Final blended score in `0.0..=1.0` (approximately).
    pub score: f32,
    /// Cosine similarity between query and chunk embeddings.
    pub cosine: f32,
    /// Fraction of query tokens present in the chunk.
    pub lexical: f32,
    /// The matched chunk.
    pub chunk: Chunk,
}

/// Snapshot of index state for `coderag status`.
#[derive(Debug, Clone, Serialize)]
pub struct StatusReport {
    /// Project root.
    pub project: PathBuf,
    /// Path of the SQLite index.
    pub db_path: PathBuf,
    /// Database size in bytes.
    pub db_size_bytes: u64,
    /// Indexed file count.
    pub files: usize,
    /// Stored chunk count.
    pub chunks: usize,
    /// Chunk counts per language.
    pub languages: Vec<(String, usize)>,
    /// Provider:model:dimensions signature recorded in the index.
    pub embedding_signature: Option<String>,
    /// Unix timestamp (seconds) of index creation, if known.
    pub created_at: Option<i64>,
    /// Index format version.
    pub format_version: String,
    /// Whether the sqlite-vec candidate mirror is compiled in.
    pub ann_available: bool,
}

/// The code retrieval engine for one project directory.
pub struct Engine {
    root: PathBuf,
    config: CodeRagConfig,
    store: SqliteStore,
    provider: Arc<dyn EmbeddingProvider>,
    chunker: Chunker,
}

impl Engine {
    /// Creates (and opens) an engine with the given configuration, creating
    /// the `.coderag` directory and database if needed.
    pub fn open(root: &Path, config: CodeRagConfig) -> Result<Self> {
        config.validate()?;
        let dir = root.join(".coderag");
        std::fs::create_dir_all(&dir)?;
        let store = SqliteStore::open(&dir.join("index.db"))?;
        let provider = provider_from_config(&config)?;
        let chunker = Chunker::new(ChunkOptions {
            max_chunk_chars: config.index.max_chunk_chars,
            min_chunk_chars: config.index.min_chunk_chars,
            preamble_chars: config.index.preamble_chars,
        });
        #[cfg(feature = "ann")]
        store
            .enable_ann(provider.dimensions())
            .map_err(|e| Error::Config(format!("sqlite-vec setup failed: {e}")))?;
        Ok(Self {
            root: root.to_path_buf(),
            config,
            store,
            provider,
            chunker,
        })
    }

    /// Loads `.coderag/config.toml` and opens the engine.
    pub fn load(root: &Path) -> Result<Self> {
        let cfg_path = root.join(".coderag").join("config.toml");
        if !cfg_path.exists() {
            return Err(Error::NotInitialized(root.to_path_buf()));
        }
        let text = std::fs::read_to_string(&cfg_path)?;
        let config = CodeRagConfig::from_toml(&text)?;
        Self::open(root, config)
    }

    /// The active configuration.
    pub fn config(&self) -> &CodeRagConfig {
        &self.config
    }

    /// Project root.
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn progress(&self, opts: &IndexOptions, event: Progress) {
        if let Some(cb) = &opts.progress {
            cb(event);
        }
    }

    fn ensure_signature(&mut self, rebuild: bool) -> Result<bool> {
        let sig = self.config.embedding_signature();
        let existing = self.store.meta_get("embedding_signature")?;
        match existing {
            None => {
                self.store
                    .meta_set("format_version", INDEX_FORMAT_VERSION)?;
                self.store.meta_set("embedding_signature", &sig)?;
                self.store
                    .meta_set("created_at", &now_seconds().to_string())?;
                Ok(false)
            }
            Some(current) if current != sig => {
                if !rebuild {
                    return Err(Error::IndexMismatch {
                        indexed: current,
                        requested: sig,
                    });
                }
                self.store.clear()?;
                self.store
                    .meta_set("format_version", INDEX_FORMAT_VERSION)?;
                self.store.meta_set("embedding_signature", &sig)?;
                self.store
                    .meta_set("created_at", &now_seconds().to_string())?;
                Ok(true)
            }
            Some(_) => {
                if rebuild {
                    self.store.clear()?;
                    self.store
                        .meta_set("format_version", INDEX_FORMAT_VERSION)?;
                    self.store.meta_set("embedding_signature", &sig)?;
                    self.store
                        .meta_set("created_at", &now_seconds().to_string())?;
                    return Ok(true);
                }
                Ok(false)
            }
        }
    }

    /// Runs an incremental index pass.
    pub async fn index(&mut self, opts: IndexOptions) -> Result<IndexReport> {
        let started = Instant::now();
        let mut report = IndexReport {
            rebuilt: opts.rebuild,
            ..IndexReport::default()
        };
        if self.ensure_signature(opts.rebuild)? {
            report.rebuilt = true;
        }

        // ---- discover -------------------------------------------------
        let mut skip: HashSet<std::ffi::OsString> = self
            .config
            .index
            .skip_dirs
            .iter()
            .map(String::clone)
            .map(Into::into)
            .collect();
        skip.insert(".git".into());
        skip.insert(".coderag".into());

        let walker = ignore::WalkBuilder::new(&self.root)
            .hidden(true)
            .git_ignore(true)
            .git_global(true)
            .git_exclude(true)
            .filter_entry(move |entry| {
                !entry
                    .path()
                    .components()
                    .any(|c| skip.contains(c.as_os_str()))
            })
            .build();

        struct Candidate {
            rel: String,
            size: u64,
        }
        let mut candidates: Vec<Candidate> = Vec::new();
        for entry in walker.flatten() {
            let Some(file_type) = entry.file_type() else {
                continue;
            };
            if !file_type.is_file() {
                continue;
            }
            let Ok(rel_path) = entry.path().strip_prefix(&self.root) else {
                continue;
            };
            let size = entry.metadata().map(|m| m.len()).unwrap_or(u64::MAX);
            candidates.push(Candidate {
                rel: rel_path.to_string_lossy().replace('\\', "/"),
                size,
            });
        }
        report.files_seen = candidates.len();
        self.progress(
            &opts,
            Progress::Discovered {
                files: report.files_seen,
            },
        );

        // ---- diff against the store ------------------------------------
        let mut existing: HashMap<String, Vec<u8>> = self.store.files()?.into_iter().collect();
        let on_disk: HashSet<&str> = candidates.iter().map(|c| c.rel.as_str()).collect();

        for path in existing.keys() {
            if !on_disk.contains(path.as_str()) {
                self.store.remove_file(path)?;
                report.files_removed += 1;
            }
        }

        let mut pending: Vec<Candidate> = Vec::new();
        for candidate in candidates {
            if candidate.size > self.config.index.max_file_bytes {
                if existing.contains_key(&candidate.rel) {
                    self.store.remove_file(&candidate.rel)?;
                    report.files_removed += 1;
                }
                report.files_skipped_large += 1;
                continue;
            }
            if !self.chunker.supports(&candidate.rel) {
                // Language was dropped from the build (or extension changed):
                // drop any stale index entry and stop tracking the file.
                if existing.remove(&candidate.rel).is_some() {
                    self.store.remove_file(&candidate.rel)?;
                    report.files_removed += 1;
                }
                report.files_skipped_unknown += 1;
                continue;
            }
            match existing.get(&candidate.rel) {
                Some(hash) => {
                    let bytes = match std::fs::read(self.root.join(&candidate.rel)) {
                        Ok(bytes) => bytes,
                        Err(err) => {
                            report
                                .errors
                                .push(format!("{}: read failed: {err}", candidate.rel));
                            continue;
                        }
                    };
                    let new_hash = blake3::hash(&bytes);
                    if new_hash.as_bytes() == hash.as_slice() {
                        report.files_unchanged += 1;
                    } else {
                        report.files_changed += 1;
                        pending.push(candidate);
                    }
                }
                None => {
                    report.files_new += 1;
                    pending.push(candidate);
                }
            }
        }

        // ---- chunk ------------------------------------------------------
        struct PendingChunk {
            rel: String,
            size: u64,
            chunk: Chunk,
        }
        let mut flat: Vec<PendingChunk> = Vec::new();
        let mut new_hashes: HashMap<String, Vec<u8>> = HashMap::new();
        for candidate in pending {
            let path = self.root.join(&candidate.rel);
            let bytes = match std::fs::read(&path) {
                Ok(bytes) => bytes,
                Err(err) => {
                    report
                        .errors
                        .push(format!("{}: read failed: {err}", candidate.rel));
                    continue;
                }
            };
            new_hashes.insert(
                candidate.rel.clone(),
                blake3::hash(&bytes).as_bytes().to_vec(),
            );
            let src = match String::from_utf8(bytes) {
                Ok(src) => src,
                Err(_) => {
                    report.files_skipped_binary += 1;
                    continue;
                }
            };
            match self.chunker.chunk(&candidate.rel, &src) {
                Ok(chunks) => {
                    for chunk in chunks {
                        flat.push(PendingChunk {
                            rel: candidate.rel.clone(),
                            size: candidate.size,
                            chunk,
                        });
                    }
                }
                Err(err) => report.errors.push(format!("{}: {err}", candidate.rel)),
            }
        }

        let total = flat.len();
        self.progress(
            &opts,
            Progress::Plan {
                to_embed: total,
                removed: report.files_removed,
            },
        );

        // ---- embed ------------------------------------------------------
        let texts: Vec<String> = flat.iter().map(|p| p.chunk.embedding_text()).collect();
        let dimensions = self.provider.dimensions();
        let batch = self.config.embedding.batch_size.max(1);
        let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(total);
        for chunk_of_texts in texts.chunks(batch) {
            let mut vectors = self
                .provider
                .embed(chunk_of_texts.to_vec())
                .await
                .map_err(|e| Error::Embed(format!("batch failed: {e}")))?;
            for v in &vectors {
                if v.len() != dimensions {
                    return Err(Error::Embed(format!(
                        "provider returned {} dims, expected {dimensions}",
                        v.len()
                    )));
                }
            }
            embeddings.append(&mut vectors);
            self.progress(
                &opts,
                Progress::Embedding {
                    done: embeddings.len(),
                    total,
                },
            );
        }
        report.chunks_embedded = embeddings.len();

        // ---- persist ------------------------------------------------------
        /// File bucket used while grouping embedded chunks for persistence.
        type FileBucket = (u64, Vec<(Chunk, Vec<f32>)>);
        let mut order: Vec<String> = Vec::new();
        let mut by_file: HashMap<String, FileBucket> = HashMap::new();
        for (item, embedding) in flat.into_iter().zip(embeddings) {
            match by_file.entry(item.rel.clone()) {
                std::collections::hash_map::Entry::Occupied(mut e) => {
                    e.get_mut().1.push((item.chunk, embedding));
                }
                std::collections::hash_map::Entry::Vacant(e) => {
                    order.push(item.rel.clone());
                    e.insert((item.size, vec![(item.chunk, embedding)]));
                }
            }
        }
        for rel in order {
            let (size, entries) = by_file.remove(&rel).expect("just inserted");
            let hash = new_hashes.get(&rel).cloned().unwrap_or_default();
            self.store
                .replace_file_chunks(&rel, &hash, size, &entries)?;
            report.chunks_indexed += entries.len();
        }

        report.elapsed_ms = started.elapsed().as_millis() as u64;
        Ok(report)
    }

    /// Hybrid semantic + lexical search over the index.
    pub async fn search(&self, query: &str, opts: SearchOptions) -> Result<Vec<SearchHit>> {
        if self.store.stats()?.chunks == 0 {
            return Ok(Vec::new());
        }
        let query_vec = self.provider.embed_one(query.to_string()).await?;
        let dimensions = self.provider.dimensions();
        let total = self.store.stats()?.chunks;
        let top_k = opts.top_k.unwrap_or(self.config.search.top_k).max(1);

        // Candidate selection. The ANN mirror serves only unfiltered
        // queries (vec0 KNN can't join metadata predicates); filtered
        // queries fall back to the SQL-scan path, which applies the
        // predicates before embeddings are ever loaded.
        let sql_filters = opts.language.is_some() || opts.symbols_only;
        let ann_active = cfg!(feature = "ann")
            && opts.use_ann.unwrap_or(true)
            && !sql_filters
            && self.store.ann_available();
        let mut records = if ann_active && total > 0 {
            // Pool must be generous: the hybrid rerank can promote chunks
            // that pure cosine ranks low, so a tight pool trades recall.
            let pool = (top_k.saturating_mul(50)).max(500).min(total);
            let ids = self.store.ann_search(&query_vec, pool)?;
            self.store.records_by_ids(&ids, dimensions, None, false)?
        } else if sql_filters {
            self.store.all_records_filtered(
                dimensions,
                opts.language.as_deref(),
                opts.symbols_only,
            )?
        } else {
            self.store.all_records(dimensions)?
        };
        if let Some(pattern) = &opts.path_glob {
            let matcher = globset::Glob::new(pattern)
                .map_err(|e| Error::Config(format!("invalid path glob {pattern:?}: {e}")))?
                .compile_matcher();
            records.retain(|record| matcher.is_match(&record.chunk.path));
        }
        if records.is_empty() {
            return Ok(Vec::new());
        }

        let query_tokens: HashSet<String> = tokens::unique_tokens(query);
        let weight = self.config.search.lexical_weight;
        let max_per_file = opts
            .max_per_file
            .unwrap_or(self.config.search.max_per_file)
            .max(1);
        let min_score = opts.min_score.unwrap_or(self.config.search.min_score);

        let mut scored: Vec<(f32, f32, f32, SearchRecord)> = records
            .into_par_iter()
            .map(|record| {
                let cosine = cosine_similarity(&query_vec, &record.embedding);
                let doc_tokens = tokens::unique_tokens(&format!(
                    "{} {} {}",
                    record.chunk.symbol.as_deref().unwrap_or(""),
                    record.chunk.content,
                    record.chunk.context
                ));
                let lexical = if query_tokens.is_empty() {
                    0.0
                } else {
                    query_tokens.intersection(&doc_tokens).count() as f32
                        / query_tokens.len() as f32
                };
                // Small exact-symbol bonus: queries naming a symbol rank it up.
                let symbol_bonus = record
                    .chunk
                    .symbol
                    .as_deref()
                    .map(|symbol| {
                        let symbol_tokens = tokens::unique_tokens(symbol);
                        if symbol_tokens.is_empty() || query_tokens.is_empty() {
                            0.0
                        } else {
                            0.1 * symbol_tokens
                                .iter()
                                .filter(|t| query_tokens.contains(*t))
                                .count() as f32
                                / query_tokens.len() as f32
                        }
                    })
                    .unwrap_or(0.0);
                let score = ((1.0 - weight) * cosine + weight * lexical + symbol_bonus).min(1.0);
                (score, cosine, lexical, record)
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut hits = Vec::with_capacity(top_k);
        let mut per_file: HashMap<String, usize> = HashMap::new();
        for (score, cosine, lexical, record) in scored {
            if score < min_score {
                break;
            }
            let count = per_file.entry(record.chunk.path.clone()).or_insert(0);
            if *count >= max_per_file {
                continue;
            }
            *count += 1;
            hits.push(SearchHit {
                score,
                cosine,
                lexical,
                chunk: record.chunk,
            });
            if hits.len() >= top_k {
                break;
            }
        }
        Ok(hits)
    }

    /// Returns a snapshot of index state.
    pub fn status(&self) -> Result<StatusReport> {
        let stats = self.store.stats()?;
        Ok(StatusReport {
            project: self.root.clone(),
            db_path: self.store.path().to_path_buf(),
            db_size_bytes: self.store.size_bytes(),
            files: stats.files,
            chunks: stats.chunks,
            languages: stats.languages,
            embedding_signature: self.store.meta_get("embedding_signature")?,
            created_at: self
                .store
                .meta_get("created_at")?
                .and_then(|s| s.parse::<i64>().ok()),
            format_version: INDEX_FORMAT_VERSION.to_string(),
            ann_available: self.store.ann_available(),
        })
    }

    /// The embedding provider in use.
    pub fn provider(&self) -> &dyn EmbeddingProvider {
        self.provider.as_ref()
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let dot: f32 = a[..n].iter().zip(&b[..n]).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na <= f32::EPSILON || nb <= f32::EPSILON {
        return 0.0;
    }
    dot / (na * nb)
}

fn now_seconds() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}
