//! Golden corpus for the retrieval benchmark (see `retrieval_fixtures.rs`).
//!
//! Deliberately distinct topic areas so lexical scoring separates them; the
//! Rust `retry.rs` impl and the TS `ApiClient` class exceed the default
//! chunk budget, which exercises oversized-definition decomposition and
//! gives their methods their own retrievable chunks.

use std::fs;
use std::path::Path;

/// Writes the corpus under `root` and returns the list of relative paths.
pub fn write(root: &Path) {
    let files: &[(&str, &str)] = &[
        ("src/retry.rs", RETRY_RS),
        ("src/btree.rs", BTREE_RS),
        ("src/config_loader.rs", CONFIG_LOADER_RS),
        ("src/event_bus.rs", EVENT_BUS_RS),
        ("src/geometry.rs", GEOMETRY_RS),
        ("scripts/pipeline.py", PIPELINE_PY),
        ("scripts/stats.py", STATS_PY),
        ("web/api-client.ts", API_CLIENT_TS),
        ("queue/worker.go", WORKER_GO),
    ];
    for (rel, contents) in files {
        let path = root.join(rel);
        fs::create_dir_all(path.parent().expect("parent dir")).expect("create dir");
        fs::write(path, contents).expect("write corpus file");
    }
}

const RETRY_RS: &str = r#"//! HTTP retry policy with exponential backoff.

use std::time::Duration;

/// Policy describing how failed HTTP requests are retried.
pub struct HttpRetryPolicy {
    max_attempts: u32,
    base_delay_ms: u64,
    max_delay_ms: u64,
}

impl HttpRetryPolicy {
    /// Creates a policy with the given attempt budget and delay bounds.
    pub fn new(max_attempts: u32, base_delay_ms: u64, max_delay_ms: u64) -> Self {
        HttpRetryPolicy {
            max_attempts,
            base_delay_ms,
            max_delay_ms,
        }
    }

    /// Computes the exponential backoff delay before the next retry
    /// attempt. Each successive retry doubles the previous delay, starting
    /// from `base_delay_ms`, until `max_delay_ms` is reached. A small
    /// deterministic jitter is added on top so concurrent clients do not
    /// retry in lockstep.
    ///
    /// The delay schedule looks like:
    ///
    /// | attempt | computed delay (base 100ms, cap 10s) |
    /// |---------|--------------------------------------|
    /// | 0       | 110 ms                               |
    /// | 1       | 220 ms                               |
    /// | 2       | 440 ms                               |
    /// | 3       | 880 ms                               |
    /// | 4       | 1760 ms                              |
    /// | 5+      | capped near `max_delay_ms`           |
    pub fn compute_backoff_delay(&self, attempt: u32) -> Duration {
        let shift = attempt.min(16);
        let exponential = self.base_delay_ms.saturating_mul(1 << shift);
        let capped = exponential.min(self.max_delay_ms);
        let jitter = (capped / 10).max(1);
        let total = capped.saturating_add(jitter);
        Duration::from_millis(total)
    }

    /// Returns true when the retry budget is exhausted and the caller
    /// should give up on the request instead of scheduling another
    /// backoff delay. Callers typically pair this with
    /// [`HttpRetryPolicy::compute_backoff_delay`]: as long as attempts
    /// remain, sleep for the returned delay and retry the failed
    /// request; once the budget is gone, surface the last error to the
    /// caller and stop retrying entirely.
    pub fn attempts_exhausted(&self, attempt: u32) -> bool {
        attempt >= self.max_attempts
    }
}
"#;

const BTREE_RS: &str = r#"//! Binary search tree helpers.

/// A node in the binary search tree.
pub struct TreeNode {
    pub value: i64,
    pub left: Option<Box<TreeNode>>,
    pub right: Option<Box<TreeNode>>,
}

/// Inserts a value into the binary search tree rooted at `node`,
/// returning the new subtree root. Duplicate values are ignored.
pub fn binary_search_tree_insert(
    node: Option<Box<TreeNode>>,
    value: i64,
) -> Option<Box<TreeNode>> {
    match node {
        None => Some(Box::new(TreeNode {
            value,
            left: None,
            right: None,
        })),
        Some(mut current) => {
            if value < current.value {
                current.left = binary_search_tree_insert(current.left, value);
            } else if value > current.value {
                current.right = binary_search_tree_insert(current.right, value);
            }
            Some(current)
        }
    }
}

/// Walks the tree in order, appending the sorted values to `out`.
pub fn inorder_traversal(node: &TreeNode, out: &mut Vec<i64>) {
    if let Some(left) = &node.left {
        inorder_traversal(left, out);
    }
    out.push(node.value);
    if let Some(right) = &node.right {
        inorder_traversal(right, out);
    }
}
"#;

const CONFIG_LOADER_RS: &str = r#"//! Configuration loading and merging.

use std::collections::BTreeMap;
use std::path::Path;

/// Parsed configuration: dotted keys to string values.
pub type ConfigMap = BTreeMap<String, String>;

/// Parses TOML configuration text into a flat map of dotted keys to
/// string values. Malformed lines are reported as errors instead of
/// panicking.
pub fn parse_toml_config(text: &str) -> Result<ConfigMap, String> {
    let mut config = ConfigMap::new();
    let mut section = String::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            section = line[1..line.len() - 1].to_string();
            continue;
        }
        if let Some((key, value)) = line.split_once('=') {
            let full_key = if section.is_empty() {
                key.trim().to_string()
            } else {
                format!("{section}.{}", key.trim())
            };
            config.insert(full_key, value.trim().trim_matches('"').to_string());
        } else {
            return Err(format!("invalid config line: {line}"));
        }
    }
    Ok(config)
}

/// Loads configuration from a file and merges it over defaults.
pub struct ConfigLoader {
    defaults: ConfigMap,
}

impl ConfigLoader {
    pub fn new(defaults: ConfigMap) -> Self {
        ConfigLoader { defaults }
    }

    /// Reads the configuration file, parses it, and merges the parsed
    /// values over the defaults so missing fields fall back to their
    /// default values.
    pub fn merge_defaults(&self, path: &Path) -> Result<ConfigMap, String> {
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let parsed = parse_toml_config(&text)?;
        let mut merged = self.defaults.clone();
        for (key, value) in parsed {
            merged.insert(key, value);
        }
        Ok(merged)
    }
}
"#;

const EVENT_BUS_RS: &str = r#"//! Minimal synchronous event bus.

/// An event bus that fans events out to registered listeners.
pub struct EventBus {
    listeners: Vec<String>,
}

impl EventBus {
    pub fn new() -> Self {
        EventBus {
            listeners: Vec::new(),
        }
    }

    /// Registers a listener by name so it receives published events.
    pub fn subscribe(&mut self, listener: String) {
        self.listeners.push(listener);
    }

    /// Publishes an event to every subscribed listener, returning their
    /// acknowledgements in subscription order.
    pub fn publish(&self, event: &str) -> Vec<String> {
        self.listeners
            .iter()
            .map(|listener| format!("{listener} received {event}"))
            .collect()
    }
}
"#;

const GEOMETRY_RS: &str = r#"//! Planar geometry helpers.

/// Computes the area of a polygon from its vertices using the shoelace
/// formula. Vertices must be ordered around the boundary.
pub fn polygon_area(vertices: &[(f64, f64)]) -> f64 {
    let count = vertices.len();
    if count < 3 {
        return 0.0;
    }
    let mut doubled = 0.0;
    for index in 0..count {
        let (x1, y1) = vertices[index];
        let (x2, y2) = vertices[(index + 1) % count];
        doubled += x1 * y2 - x2 * y1;
    }
    (doubled / 2.0).abs()
}

/// Computes the centroid of a triangle: the intersection of its medians,
/// i.e. the coordinate-wise average of its three vertices.
pub fn triangle_centroid(
    a: (f64, f64),
    b: (f64, f64),
    c: (f64, f64),
) -> (f64, f64) {
    ((a.0 + b.0 + c.0) / 3.0, (a.1 + b.1 + c.1) / 3.0)
}
"#;

const PIPELINE_PY: &str = r#""""Data pipeline stages."""

import json


def load_config(path):
    """Reads the pipeline configuration from a JSON file."""
    with open(path) as handle:
        return json.load(handle)


class DataPipeline:
    """Runs records through a series of transform stages."""

    def __init__(self, stages):
        self.stages = stages

    def transform_records(self, records):
        """Applies every transform stage to each record in order."""
        transformed = []
        for record in records:
            for stage in self.stages:
                record = stage(record)
            transformed.append(record)
        return transformed

    def run(self, config):
        """Loads records from the configured source and transforms them."""
        records = load_config(config["source"])
        return self.transform_records(records)
"#;

const STATS_PY: &str = r#""""Robust statistics helpers."""

import statistics


def median_absolute_deviation(samples):
    """Computes the median absolute deviation (MAD): a robust measure of
    spread that stays stable in the presence of outliers."""
    median = statistics.median(samples)
    deviations = [abs(sample - median) for sample in samples]
    return statistics.median(deviations)


def winsorize_outliers(samples, limit):
    """Clips extreme samples to the percentile bounds so outliers do not
    dominate downstream analysis."""
    ordered = sorted(samples)
    lower = ordered[int(limit * len(ordered))]
    upper = ordered[int((1.0 - limit) * len(ordered)) - 1]
    return [min(max(sample, lower), upper) for sample in samples]
"#;

const API_CLIENT_TS: &str = r#"// HTTP API client with session handling.

export interface ApiClientOptions {
  baseUrl: string;
  timeoutMs: number;
  token?: string;
}

export class ApiClient {
  private options: ApiClientOptions;
  private sessionToken: string | undefined;

  constructor(options: ApiClientOptions) {
    this.options = options;
    this.sessionToken = options.token;
  }

  /** Performs a JSON request against the API, failing after the
   * configured timeout so a hanging server cannot stall callers. */
  async fetchJson(path: string, init?: RequestInit): Promise<unknown> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.options.timeoutMs);
    try {
      const response = await fetch(this.options.baseUrl + path, {
        ...init,
        signal: controller.signal,
        headers: {
          ...init?.headers,
          ...(this.sessionToken
            ? { authorization: `Bearer ${this.sessionToken}` }
            : {}),
        },
      });
      if (!response.ok) {
        throw new Error(`request failed with status ${response.status}`);
      }
      return await response.json();
    } finally {
      clearTimeout(timer);
    }
  }

  /** Exchanges an expired session token for a fresh one by posting a
   * renewal request to the session endpoint. */
  async refreshSessionToken(): Promise<void> {
    const payload = await this.fetchJson("/session/renew", { method: "POST" });
    const renewal = payload as { token: string };
    this.sessionToken = renewal.token;
  }

  /** Convenience wrapper that swallows errors and returns null. */
  async getOptional(path: string): Promise<unknown | null> {
    try {
      return await this.fetchJson(path);
    } catch {
      return null;
    }
  }
}
"#;

const WORKER_GO: &str = r#"// Package queue implements a background job worker.
package queue

import "time"

// JobWorker drains a queue of pending jobs in batches.
type JobWorker struct {
	queue   chan Job
	batch   int
	process func(Job) error
}

// Job is one unit of background work.
type Job struct {
	ID      string
	Payload []byte
}

// processJobQueue loops forever, handing every incoming job to the
// worker's processing function.
func processJobQueue(worker *JobWorker) {
	for job := range worker.queue {
		_ = worker.process(job)
		time.Sleep(1 * time.Millisecond)
	}
}

// drainQueue takes up to batch pending jobs and runs each one.
func (worker *JobWorker) drainQueue() int {
	handled := 0
	for handled < worker.batch {
		select {
		case job := <-worker.queue:
			_ = worker.process(job)
			handled++
		default:
			return handled
		}
	}
	return handled
}
"#;
