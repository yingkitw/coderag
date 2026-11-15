//! Debug utility: print the chunks produced for one file.
//!
//! Usage: `cargo run -p coderag --example dump_chunks -- path/to/file.rs
//! [max_chunk_chars]`

use coderag::{ChunkOptions, Chunker};

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: dump_chunks <file> [max_chunk_chars]");
        std::process::exit(2);
    };
    let max_chunk_chars = args.next().and_then(|a| a.parse().ok()).unwrap_or(1600);
    let src = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        eprintln!("{path}: {e}");
        std::process::exit(1);
    });
    let rel = path.trim_start_matches("./").to_string();
    let chunker = Chunker::new(ChunkOptions {
        max_chunk_chars,
        min_chunk_chars: 24,
        preamble_chars: 512,
    });
    match chunker.chunk(&rel, &src) {
        Ok(chunks) => {
            for c in chunks {
                println!(
                    "=== kind={} symbol={:?} lines={}-{} len={}",
                    c.kind,
                    c.symbol,
                    c.start_line,
                    c.end_line,
                    c.content.len()
                );
                println!("{}", c.content);
                println!(
                    "--- context: {:?}",
                    c.context.lines().take(3).collect::<Vec<_>>()
                );
                println!();
            }
        }
        Err(e) => {
            eprintln!("chunk failed: {e}");
            std::process::exit(1);
        }
    }
}
