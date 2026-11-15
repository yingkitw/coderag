//! SQLite-backed vector store.
//!
//! Design notes:
//! - Vectors are stored as little-endian `f32` BLOBs next to their chunk.
//!   Brute-force cosine over an in-memory load is exact and fast enough for
//!   codebase-scale indexes (tens of thousands of chunks scan in
//!   milliseconds); swapping in an ANN index later only touches this module.
//! - `chunks.path` has a foreign key to `files` with `ON DELETE CASCADE`, so
//!   re-indexing or deleting a file atomically replaces its chunks.

use std::path::{Path, PathBuf};

use rusqlite::{params, Connection, OptionalExtension};

use crate::chunking::Chunk;
use crate::error::{Error, Result};

#[cfg(feature = "ann")]
use super::ann;
use super::SearchRecord;

/// Aggregate counters for [`SqliteStore::stats`].
#[derive(Debug, Clone, Default)]
pub struct StoreStats {
    /// Number of indexed files.
    pub files: usize,
    /// Number of stored chunks.
    pub chunks: usize,
    /// Chunk counts per language.
    pub languages: Vec<(String, usize)>,
}

/// Persistent store: `.coderag/index.db`.
pub struct SqliteStore {
    conn: Connection,
    db_path: PathBuf,
}

impl SqliteStore {
    /// Opens (creating if necessary) the store and applies pragmas + schema.
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // Register the vec extension *before* the connection exists:
        // sqlite3_auto_extension only applies to connections opened after
        // registration, and engines can be opened concurrently.
        #[cfg(feature = "ann")]
        ann::register_extension()?;
        let conn = Connection::open(path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS meta(
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS files(
                path       TEXT PRIMARY KEY,
                hash       BLOB NOT NULL,
                size       INTEGER NOT NULL,
                chunks     INTEGER NOT NULL,
                indexed_at INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS chunks(
                id        BLOB PRIMARY KEY,
                path      TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
                language  TEXT NOT NULL,
                symbol    TEXT,
                kind      TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line  INTEGER NOT NULL,
                content   TEXT NOT NULL,
                context   TEXT NOT NULL DEFAULT '',
                embedding BLOB NOT NULL
             );
             CREATE INDEX IF NOT EXISTS ix_chunks_path ON chunks(path);",
        )?;
        Ok(Self {
            conn,
            db_path: path.to_path_buf(),
        })
    }

    /// Path of the underlying database file.
    pub fn path(&self) -> &Path {
        &self.db_path
    }

    /// Size of the database file in bytes (0 if it vanished).
    pub fn size_bytes(&self) -> u64 {
        std::fs::metadata(&self.db_path)
            .map(|m| m.len())
            .unwrap_or(0)
    }

    /// Reads a metadata key.
    pub fn meta_get(&self, key: &str) -> Result<Option<String>> {
        let value = self
            .conn
            .query_row(
                "SELECT value FROM meta WHERE key = ?1",
                params![key],
                |row| row.get::<_, String>(0),
            )
            .optional()?;
        Ok(value)
    }

    /// Writes a metadata key.
    pub fn meta_set(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, value],
        )?;
        Ok(())
    }

    /// Returns `(path, hash)` for every indexed file.
    pub fn files(&self) -> Result<Vec<(String, Vec<u8>)>> {
        let mut stmt = self.conn.prepare("SELECT path, hash FROM files")?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?))
        })?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    /// Removes a file and (via cascade) all of its chunks.
    pub fn remove_file(&self, path: &str) -> Result<()> {
        #[cfg(feature = "ann")]
        {
            // Must run before the FK cascade removes the chunk rows the
            // vector table is keyed by. Autocommits are fine here: a
            // briefly orphaned vector row simply hydrates to nothing.
            let ids = self.chunk_ids_for_path(path)?;
            ann::delete_ids(&self.conn, &ids)?;
        }
        self.conn
            .execute("DELETE FROM files WHERE path = ?1", params![path])?;
        Ok(())
    }

    #[cfg(feature = "ann")]
    fn chunk_ids_for_path(&self, path: &str) -> Result<Vec<[u8; 16]>> {
        let mut stmt = self.conn.prepare("SELECT id FROM chunks WHERE path = ?1")?;
        let ids = stmt
            .query_map(params![path], |row| {
                Ok(row
                    .get::<_, Vec<u8>>(0)?
                    .as_slice()
                    .try_into()
                    .unwrap_or([0u8; 16]))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(ids)
    }

    /// Whether this build carries the sqlite-vec mirror.
    pub fn ann_available(&self) -> bool {
        cfg!(feature = "ann")
    }

    /// Creates/repairs the vec0 mirror table (feature `ann` builds only).
    #[cfg(feature = "ann")]
    pub fn enable_ann(&self, dimensions: usize) -> Result<()> {
        ann::ensure_table(&self.conn, dimensions)
    }

    /// KNN candidate ids via the vec0 mirror. Only functional in `ann`
    /// feature builds (returns a configuration error otherwise).
    pub fn ann_search(&self, query: &[f32], k: usize) -> Result<Vec<[u8; 16]>> {
        #[cfg(feature = "ann")]
        {
            ann::search(&self.conn, query, k)
        }
        #[cfg(not(feature = "ann"))]
        {
            let _ = (query, k);
            Err(Error::Config(
                "this binary was built without the `ann` feature".into(),
            ))
        }
    }

    /// Hydrates full records for a set of chunk ids, preserving the given
    /// (relevance) order. Cheap predicates apply at hydration time.
    pub fn records_by_ids(
        &self,
        ids: &[[u8; 16]],
        dimensions: usize,
        language: Option<&str>,
        symbols_only: bool,
    ) -> Result<Vec<SearchRecord>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let mut sql = String::from(
            "SELECT id, path, language, symbol, kind, start_line, end_line, content, context, embedding
             FROM chunks WHERE id IN (",
        );
        let mut params: Vec<Box<dyn rusqlite::ToSql>> = Vec::with_capacity(ids.len() + 1);
        for (index, id) in ids.iter().enumerate() {
            if index > 0 {
                sql.push(',');
            }
            sql.push('?');
            params.push(Box::new(id.to_vec()));
        }
        sql.push(')');
        let language_param = language.map(str::to_string);
        if let Some(language) = &language_param {
            sql.push_str(" AND language = ?");
            params.push(Box::new(language.clone()));
        }
        if symbols_only {
            sql.push_str(" AND symbol IS NOT NULL");
        }
        let borrowed: Vec<&dyn rusqlite::ToSql> = params.iter().map(|p| p.as_ref()).collect();
        let fetched = self.records_query(&sql, &borrowed, dimensions)?;
        let by_id: std::collections::HashMap<[u8; 16], SearchRecord> = fetched
            .into_iter()
            .map(|record| (record.chunk.id, record))
            .collect();
        let mut ordered = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(record) = by_id.get(id) {
                ordered.push(SearchRecord {
                    embedding: record.embedding.clone(),
                    chunk: record.chunk.clone(),
                });
            }
        }
        Ok(ordered)
    }

    /// Atomically replaces a file's chunks and updates its row.
    pub fn replace_file_chunks(
        &mut self,
        path: &str,
        hash: &[u8],
        size: u64,
        entries: &[(Chunk, Vec<f32>)],
    ) -> Result<()> {
        let tx = self.conn.transaction()?;
        tx.execute(
            "INSERT INTO files(path, hash, size, chunks, indexed_at)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(path) DO UPDATE SET
                hash = excluded.hash, size = excluded.size,
                chunks = excluded.chunks, indexed_at = excluded.indexed_at",
            params![
                path,
                hash,
                size as i64,
                entries.len() as i64,
                chrono_now_seconds()
            ],
        )?;
        // Keep the (optional) vec0 mirror inside the same transaction:
        // collect ids about to disappear, drop their vectors, then insert
        // new ones alongside the new chunk rows.
        #[cfg(feature = "ann")]
        {
            let mut stale = tx.prepare("SELECT id FROM chunks WHERE path = ?1")?;
            let old_ids: Vec<[u8; 16]> = stale
                .query_map(params![path], |row| {
                    Ok(row
                        .get::<_, Vec<u8>>(0)?
                        .as_slice()
                        .try_into()
                        .unwrap_or([0u8; 16]))
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            drop(stale);
            ann::delete_ids(&tx, &old_ids)?;
        }
        tx.execute("DELETE FROM chunks WHERE path = ?1", params![path])?;
        {
            let mut stmt = tx.prepare(
                "INSERT INTO chunks(id, path, language, symbol, kind, start_line, end_line, content, context, embedding)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            )?;
            for (chunk, embedding) in entries {
                if embedding.is_empty() {
                    return Err(Error::Embed(format!(
                        "refusing to store empty embedding for {path}"
                    )));
                }
                stmt.execute(params![
                    chunk.id.as_slice(),
                    path,
                    chunk.language,
                    chunk.symbol,
                    chunk.kind,
                    chunk.start_line as i64,
                    chunk.end_line as i64,
                    chunk.content,
                    chunk.context,
                    encode_embedding(embedding),
                ])?;
                #[cfg(feature = "ann")]
                ann::insert_entry(&tx, chunk.id, embedding)?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Loads every chunk with its embedding (search input).
    pub fn all_records(&self, dimensions: usize) -> Result<Vec<SearchRecord>> {
        self.records_query(
            "SELECT id, path, language, symbol, kind, start_line, end_line, content, context, embedding FROM chunks",
            &[],
            dimensions,
        )
    }

    /// Loads chunks filtered at the SQL level: by exact language and/or by
    /// having a symbol. Pushes cheap predicates into the database so
    /// non-matching embeddings are never loaded.
    pub fn all_records_filtered(
        &self,
        dimensions: usize,
        language: Option<&str>,
        symbols_only: bool,
    ) -> Result<Vec<SearchRecord>> {
        let mut sql = String::from(
            "SELECT id, path, language, symbol, kind, start_line, end_line, content, context, embedding FROM chunks WHERE 1=1",
        );
        let mut params: Vec<&dyn rusqlite::ToSql> = Vec::new();
        // Bind to a local that outlives `params` (an `if let` binding would
        // be scoped to the block).
        let language_param = language.map(str::to_string);
        if let Some(language) = &language_param {
            sql.push_str(" AND language = ?");
            params.push(language);
        }
        if symbols_only {
            sql.push_str(" AND symbol IS NOT NULL");
        }
        self.records_query(&sql, &params, dimensions)
    }

    fn records_query(
        &self,
        sql: &str,
        params: &[&dyn rusqlite::ToSql],
        dimensions: usize,
    ) -> Result<Vec<SearchRecord>> {
        let mut stmt = self.conn.prepare(sql)?;
        let rows = stmt.query_map(params, |row| {
            let id_blob: Vec<u8> = row.get(0)?;
            let path: String = row.get(1)?;
            let language: String = row.get(2)?;
            let symbol: Option<String> = row.get(3)?;
            let kind: String = row.get(4)?;
            let start_line: i64 = row.get(5)?;
            let end_line: i64 = row.get(6)?;
            let content: String = row.get(7)?;
            let context: String = row.get(8)?;
            let blob: Vec<u8> = row.get(9)?;
            Ok(SearchRecord {
                embedding: decode_embedding(&blob, dimensions)?,
                chunk: Chunk {
                    id: id_blob.as_slice().try_into().unwrap_or([0u8; 16]),
                    path,
                    language,
                    symbol,
                    kind,
                    start_line: start_line.max(1) as u32,
                    end_line: end_line.max(1) as u32,
                    content,
                    context,
                },
            })
        })?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    /// Aggregate counters.
    pub fn stats(&self) -> Result<StoreStats> {
        let files = self
            .conn
            .query_row("SELECT COUNT(*) FROM files", [], |r| r.get::<_, i64>(0))?
            as usize;
        let chunks = self
            .conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get::<_, i64>(0))?
            as usize;
        let mut stmt = self.conn.prepare(
            "SELECT language, COUNT(*) FROM chunks GROUP BY language ORDER BY COUNT(*) DESC",
        )?;
        let languages = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? as usize))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(StoreStats {
            files,
            chunks,
            languages,
        })
    }

    /// Drops all indexed data (metadata included) for a full rebuild.
    pub fn clear(&mut self) -> Result<()> {
        #[cfg(feature = "ann")]
        ann::clear(&self.conn)?;
        let tx = self.conn.transaction()?;
        tx.execute_batch("DELETE FROM chunks; DELETE FROM files; DELETE FROM meta;")?;
        tx.commit()?;
        Ok(())
    }
}

fn chrono_now_seconds() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn encode_embedding(v: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for f in v {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    bytes
}

fn decode_embedding(
    blob: &[u8],
    expected_dim: usize,
) -> std::result::Result<Vec<f32>, rusqlite::Error> {
    if blob.len() % 4 != 0 || blob.len() / 4 != expected_dim {
        return Err(rusqlite::Error::ToSqlConversionFailure(Box::new(
            crate::error::Error::Embed(format!(
                "stored embedding has {0} dims, expected {expected_dim} — the index was likely \
                 built with different settings; run `coderag index --rebuild`",
                blob.len() / 4
            )),
        )));
    }
    Ok(blob
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_chunk(path: &str, content: &str, id_byte: u8) -> Chunk {
        Chunk {
            id: [id_byte; 16],
            path: path.to_string(),
            language: "rust".to_string(),
            symbol: Some("demo".to_string()),
            kind: "function_item".to_string(),
            start_line: 1,
            end_line: 3,
            content: content.to_string(),
            context: String::new(),
        }
    }

    #[test]
    fn roundtrips_chunks_and_meta() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = SqliteStore::open(&dir.path().join("t.db")).unwrap();
        // Mirror the engine's open sequence in `ann` builds.
        #[cfg(feature = "ann")]
        store.enable_ann(2).unwrap();
        store.meta_set("k", "v").unwrap();
        assert_eq!(store.meta_get("k").unwrap().as_deref(), Some("v"));

        let entries = vec![
            (sample_chunk("a.rs", "fn one() {}", 1), vec![1.0f32, 0.0]),
            (sample_chunk("a.rs", "fn two() {}", 2), vec![0.0f32, 1.0]),
        ];
        store
            .replace_file_chunks("a.rs", &[1, 2, 3], 100, &entries)
            .unwrap();

        let files = store.files().unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].0, "a.rs");

        let records = store.all_records(2).unwrap();
        assert_eq!(records.len(), 2);
        assert!(records.iter().any(|r| r.chunk.content == "fn two() {}"));

        let stats = store.stats().unwrap();
        assert_eq!(stats.chunks, 2);
        assert_eq!(stats.languages, vec![("rust".to_string(), 2)]);

        store.remove_file("a.rs").unwrap();
        assert_eq!(store.stats().unwrap().chunks, 0);
    }

    #[test]
    fn dimension_mismatch_is_reported() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = SqliteStore::open(&dir.path().join("t.db")).unwrap();
        #[cfg(feature = "ann")]
        store.enable_ann(4).unwrap();
        store
            .replace_file_chunks(
                "a.rs",
                &[9],
                10,
                &[(sample_chunk("a.rs", "fn x() {}", 1), vec![0.5f32; 4])],
            )
            .unwrap();
        assert!(store.all_records(8).is_err());
    }
}
