//! `coderag` — semantic code retrieval from the command line.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use clap::{ArgAction, Parser, Subcommand, ValueEnum};
use coderag::engine::{Engine, IndexOptions, Progress, SearchOptions};
use coderag::{CodeRagConfig, ProviderKind};
use notify::Watcher;

#[derive(Parser, Debug)]
#[command(
    name = "coderag",
    version,
    about = "Semantic code retrieval for your codebase",
    long_about = "coderag indexes a codebase with syntax-aware chunking and \
                  retrieval-augmented search. It runs fully offline by default and can be \
                  switched to any OpenAI-compatible embedding API."
)]
struct Cli {
    /// Project directory to operate on.
    #[arg(long, global = true, default_value = ".")]
    project: PathBuf,

    /// Increase log verbosity (-v: info, -vv: debug).
    #[arg(short = 'v', long = "verbose", action = ArgAction::Count, global = true)]
    verbose: u8,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Initialize `.coderag/` with a config and an empty index.
    Init {
        /// Embedding provider to configure.
        #[arg(long)]
        provider: Option<ProviderArg>,
        /// Model name (defaults per provider).
        #[arg(long)]
        model: Option<String>,
        /// Embedding dimensions (defaults per provider: local 256, openai 1536).
        #[arg(long)]
        dimensions: Option<usize>,
        /// Overwrite an existing configuration.
        #[arg(long)]
        force: bool,
    },
    /// Incrementally (re)index the project.
    Index {
        /// Wipe the index and start over.
        #[arg(long)]
        rebuild: bool,
        /// Emit the report as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Search the index.
    Search {
        /// The query (natural language or identifiers).
        query: String,
        /// Maximum number of results.
        #[arg(short = 'k', long)]
        top_k: Option<usize>,
        /// Cap hits per file.
        #[arg(long)]
        max_per_file: Option<usize>,
        /// Discard hits below this score.
        #[arg(long)]
        min_score: Option<f32>,
        /// Restrict to a language (e.g. `rust`, `go`).
        #[arg(long)]
        lang: Option<String>,
        /// Restrict to paths matching a glob (e.g. `"src/**/*.rs"`).
        #[arg(long)]
        path: Option<String>,
        /// Only definition chunks (skip file/block chunks).
        #[arg(long)]
        symbols_only: bool,
        /// Force the brute-force scan even in sqlite-vec builds.
        #[arg(long)]
        no_ann: bool,
        /// Output full chunk bodies.
        #[arg(long)]
        full: bool,
        /// Output JSON.
        #[arg(long)]
        json: bool,
    },
    /// Show index status.
    Status {
        /// Output JSON.
        #[arg(long)]
        json: bool,
    },
    /// Keep the index fresh by re-indexing on file changes.
    Watch {
        /// Quiet period before a re-index is triggered.
        #[arg(long, default_value_t = 400)]
        debounce_ms: u64,
    },
    /// Print the effective configuration.
    Config {
        /// Output JSON.
        #[arg(long)]
        json: bool,
    },
    /// List languages compiled into this binary.
    Languages,
}

#[derive(ValueEnum, Clone, Copy, Debug)]
enum ProviderArg {
    Local,
    Openai,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(cli.verbose);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(run(cli))
}

async fn run(cli: Cli) -> Result<()> {
    let project = std::fs::canonicalize(&cli.project)
        .with_context(|| format!("project dir {} not found", cli.project.display()))?;

    match cli.command {
        Command::Init {
            provider,
            model,
            dimensions,
            force,
        } => init(&project, provider, model, dimensions, force),
        Command::Index { rebuild, json } => index(&project, rebuild, json).await,
        Command::Search {
            query,
            top_k,
            max_per_file,
            min_score,
            lang,
            path,
            symbols_only,
            no_ann,
            full,
            json,
        } => {
            let opts = SearchOptions {
                top_k,
                max_per_file,
                min_score,
                language: lang,
                path_glob: path,
                symbols_only,
                use_ann: Some(!no_ann),
            };
            search(&project, query, opts, full, json).await
        }
        Command::Status { json } => status(&project, json),
        Command::Watch { debounce_ms } => watch(&project, debounce_ms).await,
        Command::Config { json } => print_config(&project, json),
        Command::Languages => {
            let langs = coderag::supported_languages();
            println!("supported languages: {}", langs.join(", "));
            Ok(())
        }
    }
}

fn init_tracing(verbose: u8) {
    use tracing_subscriber::EnvFilter;
    let default = match verbose {
        0 => "warn",
        1 => "info",
        _ => "debug",
    };
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();
}

fn config_path(project: &Path) -> PathBuf {
    project.join(".coderag").join("config.toml")
}

fn init(
    project: &Path,
    provider: Option<ProviderArg>,
    model: Option<String>,
    dimensions: Option<usize>,
    force: bool,
) -> Result<()> {
    let path = config_path(project);
    if path.exists() && !force {
        bail!(
            "{} already exists (use --force to overwrite)",
            path.display()
        );
    }

    let kind = match provider {
        Some(ProviderArg::Openai) => ProviderKind::OpenAi,
        _ => ProviderKind::Local,
    };
    let dimensions = dimensions.unwrap_or(match kind {
        ProviderKind::Local => 256,
        ProviderKind::OpenAi => 1536,
    });
    let mut cfg = CodeRagConfig::default();
    cfg.embedding.provider = kind;
    cfg.embedding.dimensions = dimensions;
    if let Some(model) = model {
        cfg.embedding.model = model;
    }
    cfg.validate()?;

    std::fs::create_dir_all(project.join(".coderag"))?;
    std::fs::write(&path, cfg.to_toml()?)?;
    Engine::open(project, cfg.clone())?; // creates the database

    println!("initialized {}", project.display());
    println!(
        "  provider   : {} ({})",
        cfg.embedding.provider.as_str(),
        cfg.embedding.effective_model()
    );
    println!("  dimensions : {}", cfg.embedding.dimensions);
    println!(
        "  index      : {}",
        project.join(".coderag/index.db").display()
    );
    println!();
    println!("next steps:");
    println!("  coderag index         # build the index");
    println!("  coderag search \"...\"  # query it");
    Ok(())
}

fn load_engine(project: &Path) -> Result<Engine> {
    Engine::load(project).map_err(|e| {
        anyhow::Error::from(e).context(format!(
            "no coderag project in {} — run `coderag init` first",
            project.display()
        ))
    })
}

async fn index(project: &Path, rebuild: bool, json: bool) -> Result<()> {
    let mut engine = load_engine(project)?;
    let progress: Option<Arc<dyn Fn(Progress) + Send + Sync>> = if json {
        None
    } else {
        Some(Arc::new(|event| match event {
            Progress::Discovered { files } => {
                eprintln!("discovered {files} candidate files");
            }
            Progress::Plan { to_embed, removed } => {
                eprintln!("plan: {to_embed} chunks to embed, {removed} files removed");
            }
            Progress::Embedding { done, total } => {
                eprint!("\rembedding {done}/{total}");
                if done == total {
                    eprintln!();
                }
            }
        }))
    };
    let report = engine
        .index(IndexOptions { rebuild, progress })
        .await
        .with_context(|| "indexing failed")?;

    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }
    println!(
        "indexed {} files: {} new, {} changed, {} unchanged, {} removed",
        report.files_seen,
        report.files_new,
        report.files_changed,
        report.files_unchanged,
        report.files_removed
    );
    println!(
        "  chunks stored this run : {} ({} embedded)",
        report.chunks_indexed, report.chunks_embedded
    );
    if report.files_skipped_unknown > 0 {
        println!(
            "  skipped (no grammar)   : {} — see `coderag languages`",
            report.files_skipped_unknown
        );
    }
    if report.files_skipped_large > 0 {
        println!("  skipped (too large)    : {}", report.files_skipped_large);
    }
    if report.files_skipped_binary > 0 {
        println!("  skipped (non-utf8)     : {}", report.files_skipped_binary);
    }
    for err in &report.errors {
        eprintln!("  error: {err}");
    }
    println!("  elapsed                : {} ms", report.elapsed_ms);
    Ok(())
}

async fn search(
    project: &Path,
    query: String,
    opts: SearchOptions,
    full: bool,
    json: bool,
) -> Result<()> {
    let engine = load_engine(project)?;
    let hits = engine.search(&query, opts).await?;
    if json {
        println!("{}", serde_json::to_string_pretty(&hits)?);
        return Ok(());
    }
    if hits.is_empty() {
        println!("no hits (is the index built? try `coderag index`)");
        return Ok(());
    }
    for hit in &hits {
        let chunk = &hit.chunk;
        let symbol = chunk.symbol.as_deref().unwrap_or("-");
        println!(
            "{:.3}  {}:{}-{}  {}  [{}]",
            hit.score, chunk.path, chunk.start_line, chunk.end_line, symbol, chunk.language
        );
        let preview: Vec<&str> = chunk
            .content
            .lines()
            .take(if full { usize::MAX } else { 8 })
            .collect();
        for (i, line) in preview.iter().enumerate() {
            if full && i >= 8 {
                break;
            }
            println!("  │ {line}");
        }
        let total_lines = chunk.content.lines().count();
        if !full && total_lines > 8 {
            println!("  │ … ({} more lines, --full to expand)", total_lines - 8);
        }
        println!();
    }
    Ok(())
}

fn status(project: &Path, json: bool) -> Result<()> {
    let engine = load_engine(project)?;
    let status = engine.status()?;
    if json {
        println!("{}", serde_json::to_string_pretty(&status)?);
        return Ok(());
    }
    println!("project   : {}", status.project.display());
    println!(
        "index     : {} ({})",
        status.db_path.display(),
        human_bytes(status.db_size_bytes)
    );
    println!("files     : {}", status.files);
    println!("chunks    : {}", status.chunks);
    println!(
        "embedding : {}",
        status
            .embedding_signature
            .as_deref()
            .unwrap_or("(empty index)")
    );
    println!(
        "ann       : {}",
        if status.ann_available {
            "sqlite-vec (feature on)"
        } else {
            "brute-force scan (feature off)"
        }
    );
    if !status.languages.is_empty() {
        let langs = status
            .languages
            .iter()
            .map(|(name, count)| format!("{name}: {count}"))
            .collect::<Vec<_>>()
            .join(", ");
        println!("languages : {langs}");
    }
    Ok(())
}

fn print_config(project: &Path, json: bool) -> Result<()> {
    let engine = load_engine(project)?;
    if json {
        println!("{}", serde_json::to_string_pretty(engine.config())?);
    } else {
        println!("{}", engine.config().to_toml()?);
    }
    Ok(())
}

fn human_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB"];
    let mut value = bytes as f64;
    let mut unit = 0usize;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

fn watch_skip_set(engine: &Engine) -> HashSet<String> {
    let mut skip: HashSet<String> = engine.config().index.skip_dirs.iter().cloned().collect();
    skip.insert(".git".to_string());
    skip.insert(".coderag".to_string());
    skip
}

async fn watch(project: &Path, debounce_ms: u64) -> Result<()> {
    let mut engine = load_engine(project)?;
    let skip = watch_skip_set(&engine);

    println!("initial index pass…");
    let report = engine.index(IndexOptions::default()).await?;
    println!(
        "watching {} ({} files indexed) — press Ctrl-C to stop",
        project.display(),
        report.files_new + report.files_changed + report.files_unchanged
    );

    let (notify_tx, notify_rx) = std::sync::mpsc::channel::<notify::Result<notify::Event>>();
    let watcher = notify::recommended_watcher(notify_tx)?;
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Vec<PathBuf>>(64);

    let watched_root: PathBuf = project.to_path_buf();
    std::thread::spawn(move || {
        let mut watcher = watcher;
        if let Err(err) = watcher.watch(&watched_root, notify::RecursiveMode::Recursive) {
            eprintln!("watcher failed to start: {err}");
            return;
        }
        while let Ok(first) = notify_rx.recv() {
            let mut batch: Vec<PathBuf> = first.map(|e| e.paths).unwrap_or_default();
            while let Ok(more) = notify_rx.recv_timeout(Duration::from_millis(debounce_ms)) {
                match more {
                    Ok(event) => batch.extend(event.paths),
                    Err(err) => eprintln!("watch error: {err}"),
                }
            }
            batch.retain(|path| {
                !path.components().any(|c| {
                    let name = c.as_os_str().to_string_lossy();
                    skip.contains(&*name)
                })
            });
            if !batch.is_empty() && tx.blocking_send(batch).is_err() {
                break;
            }
        }
    });

    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                println!();
                return Ok(());
            }
            batch = rx.recv() => {
                if batch.is_none() {
                    return Ok(());
                }
                match engine.index(IndexOptions::default()).await {
                    Ok(report)
                        if report.files_new + report.files_changed + report.files_removed > 0 =>
                    {
                        println!(
                            "re-indexed: {} new, {} changed, {} removed ({} chunks embedded)",
                            report.files_new, report.files_changed, report.files_removed,
                            report.chunks_embedded
                        );
                    }
                    Ok(_) => {}
                    Err(err) => eprintln!("re-index failed: {err}"),
                }
            }
        }
    }
}
