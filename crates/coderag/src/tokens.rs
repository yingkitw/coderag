//! Identifier-aware tokenization shared by the lexical scorer and the local
//! embedding provider.
//!
//! Code queries rarely match prose exactly: users type `tree insert` while the
//! code says `binary_tree_insert_node`. The tokenizer therefore splits on all
//! non-alphanumeric boundaries *and* inside camelCase / snake_case / kebab-case
//! identifiers, keeping both the full word and its parts.

use std::collections::HashSet;

const STOPWORDS: &[&str] = &[
    "the", "a", "an", "and", "or", "of", "in", "to", "for", "is", "are", "was", "be", "been",
    "with", "on", "at", "by", "from", "as", "that", "this", "these", "those", "it", "its", "into",
    "about", "how", "what", "where", "when", "which", "who", "whom", "will", "would", "can",
    "could", "should", "shall", "may", "might", "must", "do", "does", "did", "done", "have", "has",
    "had", "not", "no", "yes", "but", "if", "then", "else", "so", "than", "too", "very", "just",
    "also", "get", "set", "use", "used", "using", "make", "made", "new", "old", "one", "two",
    "some", "any", "all", "each", "every", "both", "few", "more", "most", "other", "such", "only",
    "own", "same", "s", "t", "don", "now",
];

/// Splits a single identifier into its snake/camel/kebab parts.
///
/// `binaryTreeInsertNode` becomes `["binary", "Tree", "Insert", "Node"]`,
/// `HTTP_retry_backoff` becomes `["HTTP", "retry", "backoff"]`.
pub fn split_identifier(word: &str) -> Vec<String> {
    let mut parts: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut prev_lower_or_digit = false;
    let mut run_of_upper = 0usize;

    for ch in word.chars() {
        if ch.is_uppercase() {
            if prev_lower_or_digit {
                parts.push(std::mem::take(&mut current));
                run_of_upper = 1;
            } else {
                run_of_upper += 1;
            }
            current.push(ch);
            prev_lower_or_digit = false;
        } else if ch.is_lowercase() || ch.is_ascii_digit() {
            // Close an acronym run: "HTTPRetry" -> HTTP | Retry
            if run_of_upper > 1 && !current.is_empty() {
                let (head, tail) = current.split_at(current.len() - 1);
                let head = head.to_string();
                let tail = tail.to_string();
                parts.push(head);
                current = tail;
            }
            current.push(ch);
            prev_lower_or_digit = true;
            run_of_upper = 0;
        } else {
            // Separator inside the word (_, -, ., $, @, etc.)
            if !current.is_empty() {
                parts.push(std::mem::take(&mut current));
            }
            prev_lower_or_digit = false;
            run_of_upper = 0;
        }
    }
    if !current.is_empty() {
        parts.push(current);
    }
    parts
}

/// Tokenizes free text into lowercase search terms.
///
/// Emits both full words and their identifier sub-parts, drops stopwords and
/// single characters. Numeric tokens are preserved.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut word = String::new();
    let flush = |word: &mut String, out: &mut Vec<String>| {
        if word.is_empty() {
            return;
        }
        let parts = split_identifier(word);
        if parts.len() > 1 {
            let joined = word.to_lowercase();
            if !STOPWORDS.contains(&joined.as_str()) && joined.chars().count() >= 2 {
                out.push(joined);
            }
        }
        for part in parts {
            let token = part.to_lowercase();
            if token.chars().count() < 2 || STOPWORDS.contains(&token.as_str()) {
                if token.chars().all(|c| c.is_ascii_digit()) && !token.is_empty() {
                    out.push(token); // keep numbers of any length
                }
                continue;
            }
            out.push(token);
        }
        word.clear();
    };

    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '_' || ch == '-' || ch == '$' {
            word.push(ch);
        } else {
            flush(&mut word, &mut out);
        }
    }
    flush(&mut word, &mut out);
    out
}

/// Unique token set for a piece of text (used by the lexical scorer).
pub fn unique_tokens(text: &str) -> HashSet<String> {
    tokenize(text).into_iter().collect()
}

/// FNV-1a 64-bit hash. Chosen over fancier hashes because it is tiny and its
/// value is fully deterministic across platforms and releases — a requirement
/// for the local embedding provider and its stored vectors.
pub const fn fnv1a64(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET;
    let mut i = 0;
    while i < bytes.len() {
        hash ^= bytes[i] as u64;
        hash = hash.wrapping_mul(PRIME);
        i += 1;
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_camel_and_snake() {
        assert_eq!(
            split_identifier("binaryTreeInsertNode"),
            vec!["binary", "Tree", "Insert", "Node"]
        );
        assert_eq!(
            split_identifier("HTTP_retry_backoff"),
            vec!["HTTP", "retry", "backoff"]
        );
        assert_eq!(
            split_identifier("parseJSONFile"),
            vec!["parse", "JSON", "File"]
        );
    }

    #[test]
    fn tokenize_merges_full_word_and_parts() {
        let tokens = tokenize("binary tree insert_node");
        assert!(tokens.contains(&"insert".to_string()));
        assert!(tokens.contains(&"insert_node".to_string()));
        assert!(tokens.contains(&"tree".to_string()));
    }

    #[test]
    fn tokenize_drops_stopwords() {
        let tokens = tokenize("the function of a parser");
        assert!(!tokens.iter().any(|t| t == "the" || t == "of"));
        assert!(tokens.contains(&"function".to_string()));
    }

    #[test]
    fn fnv_matches_reference_vectors() {
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
    }
}
// touched
