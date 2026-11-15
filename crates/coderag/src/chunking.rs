//! Syntax-aware chunking.
//!
//! The chunker turns a source file into a list of [`Chunk`]s that align with
//! the language's own definition structure (functions, classes, traits, ...)
//! rather than arbitrary line windows:
//!
//! 1. Definitions at the outermost nesting level become chunks as-is.
//! 2. A definition larger than `max_chunk_chars` is decomposed into its nested
//!    definitions (methods of a class, items of an `impl`), each carrying the
//!    parent's header line as context. Leftover code becomes `block` chunks.
//! 3. A definition with no useful nested structure is split along its child
//!    statements, packing as many as fit.
//! 4. Leading comments / decorators are attached to the definition they
//!    document; the file preamble (imports, module docs) is attached to every
//!    chunk as retrieval context.
//!
//! Nothing is silently dropped: any byte span not covered by a definition is
//! emitted as a `block` chunk when it exceeds `min_chunk_chars`.

use serde::Serialize;
use tree_sitter::{Node, Parser};

use crate::error::{Error, Result};
use crate::langs::{self, LangDef};

/// Tuning for the chunker.
#[derive(Debug, Clone)]
pub struct ChunkOptions {
    /// Maximum chunk content size in bytes.
    pub max_chunk_chars: usize,
    /// Chunks smaller than this are dropped as noise.
    pub min_chunk_chars: usize,
    /// Preamble (imports etc.) cap in bytes.
    pub preamble_chars: usize,
}

impl Default for ChunkOptions {
    fn default() -> Self {
        Self {
            max_chunk_chars: 1600,
            min_chunk_chars: 24,
            preamble_chars: 512,
        }
    }
}

/// One retrieval unit: a contiguous span of a source file with metadata.
#[derive(Debug, Clone, Serialize)]
pub struct Chunk {
    /// Deterministic content hash (blake3-128) — stable across re-indexes.
    #[serde(skip)]
    pub id: [u8; 16],
    /// Path relative to the project root, `/`-separated.
    pub path: String,
    /// Language id (`rust`, `python`, ...).
    pub language: String,
    /// Best-effort symbol name (`Engine::index`, `MyClass`, ...).
    pub symbol: Option<String>,
    /// Tree-sitter node kind, or `block` / `file` for synthetic chunks.
    pub kind: String,
    /// 1-based inclusive start line.
    pub start_line: u32,
    /// 1-based inclusive end line.
    pub end_line: u32,
    /// The chunk's source text.
    pub content: String,
    /// Shared retrieval context: file preamble plus enclosing headers.
    pub context: String,
}

impl Chunk {
    /// Text handed to the embedding provider: metadata header + context +
    /// content.
    pub fn embedding_text(&self) -> String {
        let mut text = String::with_capacity(self.content.len() + self.context.len() + 64);
        text.push_str("// file: ");
        text.push_str(&self.path);
        text.push('\n');
        if let Some(symbol) = &self.symbol {
            text.push_str("// symbol: ");
            text.push_str(symbol);
            text.push('\n');
        }
        if !self.context.is_empty() {
            text.push_str(&self.context);
            if !self.context.ends_with('\n') {
                text.push('\n');
            }
        }
        text.push_str(&self.content);
        text
    }
}

/// Computes the deterministic chunk id.
pub fn chunk_id(path: &str, start_line: u32, end_line: u32, content: &str) -> [u8; 16] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(path.as_bytes());
    hasher.update(b"\0");
    hasher.update(&start_line.to_le_bytes());
    hasher.update(&end_line.to_le_bytes());
    hasher.update(content.as_bytes());
    let digest = *hasher.finalize().as_bytes();
    let mut id = [0u8; 16];
    id.copy_from_slice(&digest[..16]);
    id
}

/// Internal intermediate: a byte span plus metadata, before `Chunk` assembly.
#[derive(Debug, Clone)]
struct Section {
    start: usize,
    end: usize,
    start_row: usize, // 0-based
    end_row: usize,   // 0-based, inclusive
    kind: &'static str,
    symbol: Option<String>,
    /// Enclosing header lines only; the file preamble is prepended later.
    headers: String,
}

/// Syntax-aware chunker over the compiled-in tree-sitter grammars.
pub struct Chunker {
    opts: ChunkOptions,
}

impl Chunker {
    /// Creates a chunker with the given options.
    pub fn new(opts: ChunkOptions) -> Self {
        Self { opts }
    }

    /// Whether the path's language is compiled in.
    pub fn supports(&self, path: &str) -> bool {
        langs::by_extension(std::path::Path::new(path)).is_some()
    }

    /// Chunks a source file.
    ///
    /// Returns `Err(Error::UnsupportedLanguage)` when the extension has no
    /// grammar compiled in — callers decide whether that is a skip or a
    /// failure.
    pub fn chunk(&self, path: &str, src: &str) -> Result<Vec<Chunk>> {
        let lang = langs::by_extension(std::path::Path::new(path))
            .ok_or_else(|| Error::UnsupportedLanguage(std::path::PathBuf::from(path)))?;
        let mut parser = Parser::new();
        parser
            .set_language(&lang.language)
            .map_err(|e| Error::Parse {
                path: path.into(),
                message: e.to_string(),
            })?;
        let tree = parser.parse(src, None).ok_or_else(|| Error::Parse {
            path: path.into(),
            message: "parser returned no tree".to_string(),
        })?;
        let root = tree.root_node();

        let mut sections: Vec<Section> = Vec::new();
        self.emit_sections(root, lang, src, "", &mut sections);
        sections.sort_by_key(|s| s.start);

        // Gap pass: emit uncovered spans as `block` chunks so no code is lost.
        let mut blocks = self.gap_blocks(src, &sections);
        sections.append(&mut blocks);
        sections.sort_by_key(|s| s.start);

        // Whole-file fallback when nothing was recognized: no preamble (the
        // entire file *is* the chunk).
        let preamble = if sections.is_empty() {
            String::new()
        } else {
            normalize_preamble(
                &src[..sections
                    .first()
                    .map(|s| s.start)
                    .unwrap_or(src.len())
                    .min(src.len())],
                self.opts.preamble_chars,
            )
        };
        if sections.is_empty() {
            let text = src.trim();
            if text.len() >= self.opts.min_chunk_chars {
                sections.push(Section {
                    start: 0,
                    end: src.len(),
                    start_row: 0,
                    end_row: row_of(src, src.len().saturating_sub(1)),
                    kind: "file",
                    symbol: None,
                    headers: String::new(),
                });
            }
        }

        let mut chunks = Vec::with_capacity(sections.len());
        for section in sections {
            let content = src[section.start..section.end].trim_end();
            if content.trim().len() < self.opts.min_chunk_chars {
                continue;
            }
            let start_line = section.start_row as u32 + 1;
            let end_line = section.end_row as u32 + 1;
            let context = combine(&preamble, &section.headers);
            chunks.push(Chunk {
                id: chunk_id(path, start_line, end_line, content),
                path: path.to_string(),
                language: lang.name.to_string(),
                symbol: section.symbol,
                kind: section.kind.to_string(),
                start_line,
                end_line,
                content: content.to_string(),
                context,
            });
        }
        Ok(chunks)
    }

    /// Builds `block` sections for spans not covered by any definition.
    fn gap_blocks(&self, src: &str, sections: &[Section]) -> Vec<Section> {
        let mut blocks = Vec::new();
        let first = sections.first().map(|s| s.start).unwrap_or(src.len());
        // The leading gap becomes its own chunk only when it is too large to
        // be fully carried as preamble context on every other chunk.
        if first > 0 && src[..first].trim().len() > self.opts.preamble_chars {
            self.push_block(src, 0, first, &mut blocks);
        }
        let mut prev_end = first;
        let mut bounds: Vec<(usize, usize)> = sections.iter().map(|s| (s.start, s.end)).collect();
        bounds.push((src.len(), src.len())); // trailing-gap sentinel
        for (start, end) in bounds {
            if start > prev_end {
                self.push_block(src, prev_end, start, &mut blocks);
            }
            prev_end = prev_end.max(end);
        }
        blocks
    }

    fn push_block(&self, src: &str, raw_start: usize, raw_end: usize, out: &mut Vec<Section>) {
        let text = src[raw_start..raw_end].trim();
        if text.len() < self.opts.min_chunk_chars {
            return;
        }
        let (s, e) = trimmed_span(src, raw_start, raw_end);
        let (start_row, end_row) = rows_for(src, s, e);
        out.push(Section {
            start: s,
            end: e,
            start_row,
            end_row,
            kind: "block",
            symbol: None,
            headers: String::new(),
        });
    }

    /// Walks a node's children, emitting sections for definitions.
    fn emit_sections(
        &self,
        node: Node<'_>,
        lang: &LangDef,
        src: &str,
        headers: &str,
        out: &mut Vec<Section>,
    ) {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            let kind = child.kind();
            if lang.is_definition(kind) {
                // Attach contiguous leading comments/decorators by extending
                // the span backwards — the definition node stays authoritative.
                let mut start = child.start_byte();
                let mut start_row = child.start_position().row;
                let mut probe = child;
                while let Some(prev) = probe.prev_sibling() {
                    if !ATTACHABLE.contains(&prev.kind()) {
                        break;
                    }
                    // No blank line between the comment and the definition.
                    if prev.end_position().row + 1 < probe.start_position().row {
                        break;
                    }
                    if child.end_byte() - prev.start_byte() > self.opts.max_chunk_chars * 12 / 10 {
                        break;
                    }
                    start = prev.start_byte();
                    start_row = prev.start_position().row;
                    probe = prev;
                }
                let size = child.end_byte() - start;
                if size <= self.opts.max_chunk_chars {
                    let symbol = self.node_symbol(child, lang, src);
                    out.push(Section {
                        start,
                        end: child.end_byte(),
                        start_row,
                        end_row: child.end_position().row,
                        kind,
                        symbol,
                        headers: headers.to_string(),
                    });
                } else {
                    self.split_definition(child, lang, src, headers, out);
                }
            } else if lang.is_container(kind) {
                self.emit_sections(child, lang, src, headers, out);
            }
        }
    }

    /// Decomposes an oversized definition into nested definitions, block
    /// chunks for the leftovers, or statement-packed pieces.
    fn split_definition(
        &self,
        node: Node<'_>,
        lang: &LangDef,
        src: &str,
        headers: &str,
        out: &mut Vec<Section>,
    ) {
        let header = first_line_of(node, src);
        let combined = combine(headers, &header);
        let symbol = self.node_symbol(node, lang, src);

        let mut nested: Vec<Section> = Vec::new();
        self.emit_sections(node, lang, src, &combined, &mut nested);
        nested.sort_by_key(|s| s.start);

        if nested.is_empty() {
            self.split_statements(node, src, &combined, symbol.as_deref(), out);
            return;
        }
        // Leftovers between nested definitions become block chunks.
        let mut bounds: Vec<(usize, usize)> = nested.iter().map(|s| (s.start, s.end)).collect();
        out.extend(nested);
        bounds.push((node.end_byte(), node.end_byte()));
        let mut prev_end = node.start_byte();
        for (start, end) in bounds {
            if start > prev_end {
                let text = src[prev_end..start].trim();
                if text.len() >= self.opts.min_chunk_chars {
                    let (s, e) = trimmed_span(src, prev_end, start);
                    let (start_row, end_row) = rows_for(src, s, e);
                    out.push(Section {
                        start: s,
                        end: e,
                        start_row,
                        end_row,
                        kind: "block",
                        symbol: symbol.clone(),
                        headers: combined.clone(),
                    });
                }
            }
            prev_end = prev_end.max(end);
        }
    }

    /// Packs the children (statements) of a leaf definition into chunks that
    /// respect `max_chunk_chars`; a single oversized child is line-split.
    fn split_statements(
        &self,
        node: Node<'_>,
        src: &str,
        headers: &str,
        symbol: Option<&str>,
        out: &mut Vec<Section>,
    ) {
        let mut cursor = node.walk();
        let items: Vec<Node<'_>> = node
            .children(&mut cursor)
            .filter(|c| c.is_named())
            .collect();
        if items.len() <= 1 {
            self.split_by_lines(
                node.start_byte(),
                node.end_byte(),
                src,
                headers,
                node.kind(),
                symbol,
                out,
            );
            return;
        }
        let kind = node.kind();
        let mut packet_start = items[0].start_byte();
        let mut packet_end = packet_start;
        for item in items {
            let size = item.end_byte() - item.start_byte();
            if size > self.opts.max_chunk_chars {
                self.flush_packet(packet_start, packet_end, src, headers, kind, symbol, out);
                self.split_by_lines(
                    item.start_byte(),
                    item.end_byte(),
                    src,
                    headers,
                    kind,
                    symbol,
                    out,
                );
                packet_start = item.end_byte();
                packet_end = packet_start;
                continue;
            }
            if packet_end > packet_start
                && item.end_byte() - packet_start > self.opts.max_chunk_chars
            {
                self.flush_packet(packet_start, packet_end, src, headers, kind, symbol, out);
                packet_start = item.start_byte();
            }
            packet_end = item.end_byte();
        }
        self.flush_packet(packet_start, packet_end, src, headers, kind, symbol, out);
    }

    #[allow(clippy::too_many_arguments)]
    fn flush_packet(
        &self,
        start: usize,
        end: usize,
        src: &str,
        headers: &str,
        kind: &'static str,
        symbol: Option<&str>,
        out: &mut Vec<Section>,
    ) {
        if end <= start {
            return;
        }
        let text = src[start..end].trim();
        if text.len() < self.opts.min_chunk_chars {
            return;
        }
        let (s, e) = trimmed_span(src, start, end);
        let (start_row, end_row) = rows_for(src, s, e);
        out.push(Section {
            start: s,
            end: e,
            start_row,
            end_row,
            kind,
            symbol: symbol.map(String::from),
            headers: headers.to_string(),
        });
    }

    /// Last resort: byte-range line windows.
    #[allow(clippy::too_many_arguments)]
    fn split_by_lines(
        &self,
        start: usize,
        end: usize,
        src: &str,
        headers: &str,
        kind: &'static str,
        symbol: Option<&str>,
        out: &mut Vec<Section>,
    ) {
        let budget = (self.opts.max_chunk_chars * 9) / 10;
        let mut s = start;
        while s < end {
            let mut e = (s + budget).min(end);
            if e < end {
                if let Some(nl) = src[e..].find('\n') {
                    e += nl;
                } else {
                    e = end;
                }
            }
            let (ts, te) = trimmed_span(src, s, e);
            if te > ts && src[ts..te].trim().len() >= self.opts.min_chunk_chars {
                let (start_row, end_row) = rows_for(src, ts, te);
                out.push(Section {
                    start: ts,
                    end: te,
                    start_row,
                    end_row,
                    kind,
                    symbol: symbol.map(String::from),
                    headers: headers.to_string(),
                });
            }
            if e >= end {
                break;
            }
            s = e + 1;
        }
    }

    /// Extracts a best-effort symbol name for a definition node.
    fn node_symbol(&self, node: Node<'_>, lang: &LangDef, src: &str) -> Option<String> {
        for field in lang.name_fields {
            if let Some(name_node) = node.child_by_field_name(field) {
                if is_name_like(name_node.kind()) {
                    let text = &src[name_node.byte_range()];
                    if text.len() <= 128 {
                        return Some(text.to_string());
                    }
                }
            }
        }
        // Depth-limited BFS for identifier-ish nodes (handles Go type specs,
        // C declarators, TS interfaces, ...).
        let mut frontier = vec![node];
        for _ in 0..3 {
            let mut next = Vec::new();
            for current in &frontier {
                let mut cursor = current.walk();
                for child in current.children(&mut cursor) {
                    if is_name_like(child.kind()) {
                        let text = &src[child.byte_range()];
                        if text.len() <= 128 {
                            return Some(text.to_string());
                        }
                    }
                    next.push(child);
                }
            }
            if next.is_empty() {
                return None;
            }
            frontier = next;
        }
        None
    }
}

const ATTACHABLE: &[&str] = &[
    "comment",
    "line_comment",
    "block_comment",
    "doc_comment",
    "decorator",
    "decorator_item",
];

fn is_name_like(kind: &str) -> bool {
    matches!(
        kind,
        "identifier"
            | "type_identifier"
            | "field_identifier"
            | "property_identifier"
            | "nested_identifier"
            | "name"
    )
}

fn combine(a: &str, b: &str) -> String {
    match (a.is_empty(), b.is_empty()) {
        (true, _) => b.to_string(),
        (_, true) => a.to_string(),
        (false, false) => {
            let mut s = String::with_capacity(a.len() + b.len() + 1);
            s.push_str(a);
            if !a.ends_with('\n') {
                s.push('\n');
            }
            s.push_str(b);
            s
        }
    }
}

fn first_line_of(node: Node<'_>, src: &str) -> String {
    let text = &src[node.byte_range()];
    let line = text.lines().next().unwrap_or("").trim_end();
    line.chars().take(120).collect()
}

fn trimmed_span(src: &str, start: usize, end: usize) -> (usize, usize) {
    let mut s = start;
    let mut e = end;
    while s < e && src.as_bytes()[s].is_ascii_whitespace() {
        s += 1;
    }
    while e > s && src.as_bytes()[e - 1].is_ascii_whitespace() {
        e -= 1;
    }
    (s, e)
}

fn row_of(src: &str, byte: usize) -> usize {
    src.as_bytes()[..byte.min(src.len())]
        .iter()
        .filter(|&&b| b == b'\n')
        .count()
}

fn rows_for(src: &str, start: usize, end: usize) -> (usize, usize) {
    (
        row_of(src, start),
        row_of(src, end.saturating_sub(1).max(start)),
    )
}

fn normalize_preamble(text: &str, cap: usize) -> String {
    let mut out = String::new();
    let mut blank_run = 0;
    for line in text.lines() {
        if line.trim().is_empty() {
            blank_run += 1;
            if blank_run > 1 {
                continue;
            }
        } else {
            blank_run = 0;
        }
        out.push_str(line.trim_end());
        out.push('\n');
    }
    let trimmed = out.trim().to_string();
    if trimmed.is_empty() {
        return String::new();
    }
    if trimmed.len() > cap {
        let mut cut = cap;
        while !trimmed.is_char_boundary(cut) {
            cut -= 1;
        }
        format!("{}\n// ... (preamble truncated)", &trimmed[..cut])
    } else {
        format!("{}\n", trimmed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_rust_definitions() {
        let src = r#"use std::collections::HashMap;

/// Counts word frequencies.
pub fn word_count(text: &str) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for word in text.split_whitespace() {
        *counts.entry(word.to_string()).or_insert(0) += 1;
    }
    counts
}

pub struct Engine {
    pub name: String,
}

impl Engine {
    pub fn new(name: String) -> Self {
        Self { name }
    }

    /// Verbose description with padding so the impl exceeds the chunk budget
    /// and must be decomposed into its methods.
    pub fn describe(&self) -> String {
        let prefix = String::from("engine instance:");
        let suffix = String::from("(running)");
        let separator = String::from("--");
        format!("{prefix} {separator} {} {separator} {suffix}", self.name)
    }

    pub fn shutdown(&self) -> String {
        let message = String::from("engine stopped gracefully now");
        format!("{message}: {}", self.name)
    }
}
"#;
        let chunker = Chunker::new(ChunkOptions {
            max_chunk_chars: 260,
            min_chunk_chars: 24,
            preamble_chars: 512,
        });
        let chunks = chunker.chunk("src/lib.rs", src).unwrap();
        let symbols: Vec<Option<&str>> = chunks.iter().map(|c| c.symbol.as_deref()).collect();
        assert!(symbols.contains(&Some("word_count")), "{symbols:?}");
        assert!(symbols.contains(&Some("Engine")), "{symbols:?}");
        // impl is too big, so it must be split into its methods
        assert!(
            symbols.contains(&Some("new"))
                && symbols.contains(&Some("describe"))
                && symbols.contains(&Some("shutdown")),
            "{symbols:?}"
        );

        let wc = chunks
            .iter()
            .find(|c| c.symbol.as_deref() == Some("word_count"))
            .unwrap();
        assert_eq!(wc.kind, "function_item");
        assert!(
            wc.content.contains("Counts word frequencies"),
            "doc comment should be attached: {}",
            wc.content
        );
        assert!(wc.context.contains("use std::collections::HashMap;"));
        assert!(wc.start_line >= 1 && wc.end_line >= wc.start_line);
        // method chunks carry the impl header as context
        let describe = chunks
            .iter()
            .find(|c| c.symbol.as_deref() == Some("describe"))
            .unwrap();
        assert!(
            describe.context.contains("impl Engine {"),
            "missing impl header: {}",
            describe.context
        );
        // the preamble is context, not its own chunk
        assert!(!chunks
            .iter()
            .any(|c| c.content.trim() == "use std::collections::HashMap;"));
    }

    #[test]
    fn oversized_function_splits_by_statements() {
        let mut src = String::from("pub fn giant(x: u64) -> u64 {\n");
        for i in 0..60 {
            src.push_str(&format!("    let v{i} = x + {i} + {};\n", i * 7 + 13));
        }
        src.push_str("    x * 2\n}\n");
        let chunker = Chunker::new(ChunkOptions {
            max_chunk_chars: 400,
            min_chunk_chars: 24,
            preamble_chars: 128,
        });
        let chunks = chunker.chunk("giant.rs", &src).unwrap();
        assert!(
            chunks.len() > 1,
            "expected split, got {} chunks",
            chunks.len()
        );
        for c in &chunks {
            assert!(
                c.content.len() <= 500,
                "chunk too large: {}",
                c.content.len()
            );
            assert_eq!(c.symbol.as_deref(), Some("giant"));
        }
    }

    #[test]
    fn chunks_python_functions_and_classes() {
        let src = r#"import json


def load_config(path):
    with open(path) as f:
        return json.load(f)


class Renderer:
    def render(self, data):
        return json.dumps(data)
"#;
        let chunker = Chunker::new(ChunkOptions::default());
        let chunks = chunker.chunk("app.py", src).unwrap();
        let symbols: Vec<Option<&str>> = chunks.iter().map(|c| c.symbol.as_deref()).collect();
        assert!(symbols.contains(&Some("load_config")), "{symbols:?}");
        assert!(symbols.contains(&Some("Renderer")), "{symbols:?}");
        assert!(
            !symbols.contains(&Some("render")),
            "small class must stay whole: {symbols:?}"
        );
        let lc = chunks
            .iter()
            .find(|c| c.symbol.as_deref() == Some("load_config"))
            .unwrap();
        assert!(lc.context.contains("import json"));
        // A small class stays whole: its method is inside the class chunk.
        let renderer = chunks
            .iter()
            .find(|c| c.symbol.as_deref() == Some("Renderer"))
            .unwrap();
        assert!(
            renderer.content.contains("def render"),
            "{}",
            renderer.content
        );
    }

    #[test]
    fn file_without_definitions_becomes_file_chunk() {
        let src = "console.log(\"hello\");\nconst answer = compute();\n";
        let chunker = Chunker::new(ChunkOptions::default());
        let chunks = chunker.chunk("script.js", src).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].kind, "file");
        assert!(chunks[0].content.contains("answer"));
        assert!(chunks[0].context.is_empty());
    }

    #[test]
    fn chunk_ids_are_deterministic() {
        let chunker = Chunker::new(ChunkOptions::default());
        let a = chunker.chunk("x.rs", "fn a() {}\nfn b() {}\n").unwrap();
        let b = chunker.chunk("x.rs", "fn a() {}\nfn b() {}\n").unwrap();
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.id, y.id);
        }
    }
}
