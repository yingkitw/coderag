//! Doc-drift enforcement: the AGENTS.md structure tree must match reality.
//!
//! Every Rust file in the workspace must be listed in AGENTS.md, and every
//! `.rs` path AGENTS.md mentions must exist. When this test fails, update
//! AGENTS.md — never the test.

use std::fs;
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../")
}

fn collect_rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn agents_md_lists_every_rust_file() {
    let root = workspace_root();
    let agents = fs::read_to_string(root.join("AGENTS.md")).expect("AGENTS.md must be readable");

    let mut files = Vec::new();
    collect_rust_files(&root.join("crates/coderag/src"), &mut files);
    collect_rust_files(&root.join("crates/coderag/examples"), &mut files);
    collect_rust_files(&root.join("crates/coderag/tests"), &mut files);
    collect_rust_files(&root.join("crates/coderag-cli/src"), &mut files);
    assert!(
        !files.is_empty(),
        "file discovery found nothing — path bug?"
    );

    let lib_root = root.join("crates/coderag");
    let cli_root = root.join("crates/coderag-cli");

    // Forward: every real file is listed in the AGENTS.md tree.
    let mut listed = Vec::new();
    for file in &files {
        let crate_root = if file.starts_with(&lib_root) {
            &lib_root
        } else {
            &cli_root
        };
        let rel = file
            .strip_prefix(crate_root)
            .expect("file under its crate root")
            .to_string_lossy()
            .replace('\\', "/");
        assert!(
            agents.contains(&rel),
            "AGENTS.md structure tree is missing `{rel}` — update AGENTS.md (not this test)"
        );
        listed.push(rel);
    }

    // Reverse: every `.rs` path listed in the AGENTS.md structure tree exists.
    // Only the *first* fenced block (the tree) is checked — prose and later
    // fences (bash snippets, the web2md tool signature) may legitimately
    // mention e.g. the docs.rs website.
    let mut fence_index = 0usize;
    let mut in_tree = false;
    for line in agents.lines() {
        if line.trim_start().starts_with("```") {
            fence_index += 1;
            in_tree = fence_index == 1;
            continue;
        }
        if !in_tree {
            continue;
        }
        for raw in line.split_whitespace() {
            let token = raw.trim_matches(|c: char| {
                !c.is_ascii_alphanumeric() && c != '.' && c != '/' && c != '-' && c != '_'
            });
            if !token.ends_with(".rs") {
                continue;
            }
            let exists = listed
                .iter()
                .any(|rel| rel == token || rel.ends_with(&format!("/{token}")));
            assert!(
                exists,
                "AGENTS.md structure tree lists `{token}`, but no such Rust file exists — fix the doc"
            );
        }
    }
}
