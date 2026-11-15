//! Tree-sitter language registry.
//!
//! Each entry declares, per language:
//! - which node kinds count as *definitions* (things that become chunks),
//! - which node kinds are *containers* to look through (e.g. `export`
//!   wrappers), and
//! - which fields hold a definition's name.
//!
//! Grammar support is feature-gated (`lang-rust`, `lang-python`, ...) so
//! downstream users can compile a slim binary.

use std::sync::OnceLock;

use tree_sitter::Language;

/// Static description of one supported language.
pub struct LangDef {
    /// Lowercase language id, e.g. `rust`.
    pub name: &'static str,
    /// Compiled tree-sitter grammar.
    pub language: Language,
    /// File extensions (without the dot, lowercase).
    pub extensions: &'static [&'static str],
    /// Node kinds that produce chunks.
    pub definitions: &'static [&'static str],
    /// Node kinds that are transparent for definition lookup.
    pub containers: &'static [&'static str],
    /// Fields probed, in order, to extract a symbol name.
    pub name_fields: &'static [&'static str],
}

impl LangDef {
    /// Whether a node kind is a definition.
    pub fn is_definition(&self, kind: &str) -> bool {
        self.definitions.contains(&kind)
    }

    /// Whether a node kind is a transparent container.
    pub fn is_container(&self, kind: &str) -> bool {
        self.containers.contains(&kind)
    }
}

/// Returns the registry of all languages compiled into this build.
pub fn registry() -> &'static [LangDef] {
    static REGISTRY: OnceLock<Vec<LangDef>> = OnceLock::new();
    REGISTRY.get_or_init(build_registry)
}

/// Names of every compiled-in language.
pub fn supported_names() -> Vec<&'static str> {
    registry().iter().map(|l| l.name).collect()
}

/// Finds the language definition for a path, by extension.
pub fn by_extension(path: &std::path::Path) -> Option<&'static LangDef> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    registry()
        .iter()
        .find(|l| l.extensions.contains(&ext.as_str()))
}

fn build_registry() -> Vec<LangDef> {
    let mut langs = Vec::new();
    #[cfg(feature = "lang-rust")]
    langs.push(rust());
    #[cfg(feature = "lang-python")]
    langs.push(python());
    #[cfg(feature = "lang-javascript")]
    langs.push(javascript());
    #[cfg(feature = "lang-typescript")]
    {
        langs.push(typescript());
        langs.push(tsx());
    }
    #[cfg(feature = "lang-go")]
    langs.push(go());
    #[cfg(feature = "lang-java")]
    langs.push(java());
    #[cfg(feature = "lang-c")]
    langs.push(c());
    #[cfg(feature = "lang-cpp")]
    langs.push(cpp());
    langs
}

#[cfg(feature = "lang-rust")]
fn rust() -> LangDef {
    LangDef {
        name: "rust",
        language: tree_sitter_rust::LANGUAGE.into(),
        extensions: &["rs"],
        definitions: &[
            "function_item",
            "struct_item",
            "enum_item",
            "union_item",
            "trait_item",
            "type_item",
            "impl_item",
            "mod_item",
            "macro_definition",
            "const_item",
            "static_item",
        ],
        containers: &["declaration_list"],
        name_fields: &["name", "type"],
    }
}

#[cfg(feature = "lang-python")]
fn python() -> LangDef {
    LangDef {
        name: "python",
        language: tree_sitter_python::LANGUAGE.into(),
        extensions: &["py", "pyi"],
        definitions: &["function_definition", "class_definition"],
        containers: &["decorated_definition", "block"],
        name_fields: &["name"],
    }
}

#[cfg(feature = "lang-javascript")]
fn javascript() -> LangDef {
    LangDef {
        name: "javascript",
        language: tree_sitter_javascript::LANGUAGE.into(),
        extensions: &["js", "mjs", "cjs", "jsx"],
        definitions: &[
            "function_declaration",
            "generator_function_declaration",
            "class_declaration",
            "method_definition",
        ],
        containers: &["export_statement", "class_body"],
        name_fields: &["name"],
    }
}

#[cfg(feature = "lang-typescript")]
fn typescript() -> LangDef {
    LangDef {
        name: "typescript",
        language: tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        extensions: &["ts", "mts", "cts"],
        definitions: &[
            "function_declaration",
            "generator_function_declaration",
            "class_declaration",
            "abstract_class_declaration",
            "interface_declaration",
            "type_alias_declaration",
            "enum_declaration",
            "method_definition",
        ],
        containers: &["export_statement", "ambient_declaration", "class_body"],
        name_fields: &["name"],
    }
}

#[cfg(feature = "lang-typescript")]
fn tsx() -> LangDef {
    LangDef {
        name: "tsx",
        language: tree_sitter_typescript::LANGUAGE_TSX.into(),
        extensions: &["tsx"],
        definitions: &[
            "function_declaration",
            "generator_function_declaration",
            "class_declaration",
            "abstract_class_declaration",
            "interface_declaration",
            "type_alias_declaration",
            "enum_declaration",
            "method_definition",
        ],
        containers: &["export_statement", "class_body"],
        name_fields: &["name"],
    }
}

#[cfg(feature = "lang-go")]
fn go() -> LangDef {
    LangDef {
        name: "go",
        language: tree_sitter_go::LANGUAGE.into(),
        extensions: &["go"],
        definitions: &[
            "function_declaration",
            "method_declaration",
            "type_declaration",
        ],
        containers: &[],
        name_fields: &["name"],
    }
}

#[cfg(feature = "lang-java")]
fn java() -> LangDef {
    LangDef {
        name: "java",
        language: tree_sitter_java::LANGUAGE.into(),
        extensions: &["java"],
        definitions: &[
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
            "record_declaration",
            "annotation_type_declaration",
            "method_declaration",
            "constructor_declaration",
        ],
        containers: &["class_body", "interface_body", "enum_body_declarations"],
        name_fields: &["name"],
    }
}

#[cfg(feature = "lang-c")]
fn c() -> LangDef {
    LangDef {
        name: "c",
        language: tree_sitter_c::LANGUAGE.into(),
        extensions: &["c", "h"],
        definitions: &[
            "function_definition",
            "struct_specifier",
            "union_specifier",
            "enum_specifier",
            "type_definition",
        ],
        containers: &["declaration_list", "field_declaration_list"],
        name_fields: &["name", "declarator"],
    }
}

#[cfg(feature = "lang-cpp")]
fn cpp() -> LangDef {
    LangDef {
        name: "cpp",
        language: tree_sitter_cpp::LANGUAGE.into(),
        extensions: &["cpp", "cc", "cxx", "c++", "hpp", "hh", "hxx", "ipp"],
        definitions: &[
            "function_definition",
            "class_specifier",
            "struct_specifier",
            "union_specifier",
            "enum_specifier",
            "type_definition",
            "alias_declaration",
            "template_declaration",
        ],
        containers: &[
            "namespace_definition",
            "extern_declaration",
            "declaration",
            "declaration_list",
            "field_declaration_list",
        ],
        name_fields: &["name", "declarator"],
    }
}
