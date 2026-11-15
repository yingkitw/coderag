//! sqlite-vec integration: a `vec0` virtual table for vector candidate
//! retrieval (cargo feature `ann`).
//!
//! Design contract with the rest of the store:
//! - `vchunks(chunk_id, embedding)` is a **derived table**: its rows are
//!   written and deleted inside the *same transactions* that write
//!   `chunks`, so it can never disagree with the source of truth.
//! - KNN `distance` (L2) is used **only to order candidates**; the engine
//!   recomputes exact cosine from the stored embeddings during rerank, so
//!   nothing downstream depends on the distance metric's exact scale.
//! - sqlite-vec 0.1.x performs an exact SIMD scan — this feature is about
//!   offloading the scan to C and establishing the ANN seam; swapping in
//!   approximate indexes later touches only this module.

use rusqlite::{params, Connection};

use crate::error::{Error, Result};

const TABLE: &str = "vchunks";

/// Registers the sqlite-vec extension for every connection this process
/// opens from now on. Idempotent and thread-safe.
#[allow(unsafe_code)] // single audited FFI site; see SAFETY below
pub fn register_extension() -> Result<()> {
    use std::sync::OnceLock;
    static REGISTER: OnceLock<()> = OnceLock::new();
    REGISTER.get_or_init(|| {
        // SAFETY: the pattern blessed by the sqlite-vec crate's own tests:
        // `sqlite3_auto_extension` stores the entry point and invokes it on
        // every future connection; the transmute adapts the extension's
        // real SQLite signature to the auto-extension callback type. The C
        // entry point is statically linked into this binary, and
        // registration happens exactly once (guarded by the OnceLock).
        unsafe {
            rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute::<
                *const (),
                unsafe extern "C" fn(
                    *mut rusqlite::ffi::sqlite3,
                    *mut *mut std::ffi::c_char,
                    *const rusqlite::ffi::sqlite3_api_routines,
                ) -> i32,
            >(
                sqlite_vec::sqlite3_vec_init as *const (),
            )));
        }
    });
    Ok(())
}

/// Creates the virtual table (if missing) and backfills it from `chunks`
/// whenever the row counts disagree (fresh enable, or a DB written by a
/// build without the feature).
///
/// The table's dimensionality is baked into its DDL, so it is recorded in
/// `meta` and the table is **dropped and recreated** when the embedding
/// dimension changes (the index itself demands a rebuild anyway via the
/// signature guard; stale rows of the old width are skipped below).
pub fn ensure_table(conn: &Connection, dimensions: usize) -> Result<()> {
    register_extension()?;
    let table_exists: bool = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?1",
            params![TABLE],
            |row| row.get::<_, i64>(0),
        )
        .map(|count| count > 0)?;
    let recorded: Option<String> = conn
        .query_row(
            "SELECT value FROM meta WHERE key = 'ann_dimensions'",
            [],
            |row| row.get(0),
        )
        .ok();
    let recreate = match &recorded {
        Some(previous) => previous.parse::<usize>() != Ok(dimensions),
        None => table_exists, // unknown provenance: rebuild defensively
    };
    if recreate && table_exists {
        conn.execute_batch(&format!("DROP TABLE {TABLE};"))?;
    }
    conn.execute_batch(&format!(
        "CREATE VIRTUAL TABLE IF NOT EXISTS {TABLE} USING \
         vec0(chunk_id TEXT PRIMARY KEY, embedding float[{dimensions}]);"
    ))?;
    conn.execute(
        "INSERT INTO meta(key, value) VALUES ('ann_dimensions', ?1)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        params![dimensions.to_string()],
    )?;
    // chunks.embedding blobs are little-endian f32, exactly what vec0
    // float[] columns accept as compact binary. Rows of the wrong width
    // belong to a superseded embedding setup; they are skipped here and
    // removed by the signature-guarded rebuild.
    let valid_width = (dimensions * 4) as i64;
    let indexed: i64 =
        conn.query_row(&format!("SELECT COUNT(*) FROM {TABLE}"), [], |r| r.get(0))?;
    let source: i64 = conn.query_row(
        "SELECT COUNT(*) FROM chunks WHERE length(embedding) = ?1",
        params![valid_width],
        |r| r.get(0),
    )?;
    if indexed != source {
        conn.execute_batch(&format!(
            "DELETE FROM {TABLE};
             INSERT INTO {TABLE}(chunk_id, embedding)
               SELECT lower(hex(id)), embedding FROM chunks
               WHERE length(embedding) = {valid_width};"
        ))?;
    }
    Ok(())
}

/// Runs a KNN query and returns chunk ids (16 raw bytes, decoded from the
/// hex primary key) ordered by ascending distance.
pub fn search(conn: &Connection, query: &[f32], k: usize) -> Result<Vec<[u8; 16]>> {
    let blob = encode_vector(query);
    let mut stmt = conn.prepare(&format!(
        "SELECT chunk_id FROM {TABLE} WHERE embedding MATCH ? AND k = ?"
    ))?;
    let rows = stmt.query_map(params![blob, k as i64], |row| {
        let hex: String = row.get(0)?;
        Ok(hex)
    })?;
    let mut ids = Vec::new();
    for hex in rows {
        let hex = hex?;
        let bytes = hex_to_bytes(&hex)?;
        if bytes.len() == 16 {
            let mut id = [0u8; 16];
            id.copy_from_slice(&bytes);
            ids.push(id);
        }
    }
    Ok(ids)
}

/// Deletes vector rows for the given chunk ids (inside the caller's
/// transaction that removes the chunks themselves).
pub fn delete_ids(conn: &Connection, ids: &[[u8; 16]]) -> Result<()> {
    let mut stmt = conn.prepare(&format!("DELETE FROM {TABLE} WHERE chunk_id = ?1"))?;
    for id in ids {
        stmt.execute(params![hex_encode(id)])?;
    }
    Ok(())
}

/// Inserts one vector row (inside the caller's transaction).
pub fn insert_entry(conn: &Connection, id: [u8; 16], embedding: &[f32]) -> Result<()> {
    conn.execute(
        &format!("INSERT INTO {TABLE}(chunk_id, embedding) VALUES (?1, ?2)"),
        params![hex_encode(&id), encode_vector(embedding)],
    )?;
    Ok(())
}

/// Empties the virtual table (companion to `SqliteStore::clear`).
pub fn clear(conn: &Connection) -> Result<()> {
    conn.execute_batch(&format!("DELETE FROM {TABLE};"))?;
    Ok(())
}

fn encode_vector(v: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for f in v {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    bytes
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

fn hex_to_bytes(hex: &str) -> Result<Vec<u8>> {
    fn value(c: u8) -> Result<u8> {
        match c {
            b'0'..=b'9' => Ok(c - b'0'),
            b'a'..=b'f' => Ok(c - b'a' + 10),
            b'A'..=b'F' => Ok(c - b'A' + 10),
            _ => Err(Error::Embed("invalid hex chunk id from vec table".into())),
        }
    }
    let bytes = hex.as_bytes();
    if bytes.len() % 2 != 0 {
        return Err(Error::Embed("odd-length hex chunk id".into()));
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for pair in bytes.chunks_exact(2) {
        out.push((value(pair[0])? << 4) | value(pair[1])?);
    }
    Ok(out)
}
