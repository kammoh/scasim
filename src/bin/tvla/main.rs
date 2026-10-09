use clap::{ArgGroup, ArgMatches, CommandFactory, FromArgMatches, Parser, ValueEnum};
use itertools::Itertools;
use log::*;
use miette::{IntoDiagnostic, WrapErr, miette};
use ndarray::{Array1, Array2, Axis};
use ndarray_npz::NpzWriter;
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};
use scasim::batch::{
    BatchDiagnostics, EdgeReport, EdgeSampling, LengthPolicy, Sampling, compute_batch,
    read_batch_meta, read_trace_cache, write_trace_cache,
};
use scasim::fold::{Fold, Loaded, create_output_dir, read_path_list, write_t_values_npz};
use scasim::hierarchy::{HierarchyIndex, Selection};
use scasim::plot::*;
use scasim::power::edges::EdgeKind;
use scasim::power::{PowerPlan, hierarchy_index};
use scasim::scopes::{group_by_scope, scope_plan};
use scasim::shuffle::shuffle_labels;
use scasim::stats::threshold::{CONVENTIONAL, bonferroni, family_size, t_bonferroni};
use scasim::stats::{HistAccumulator, TestResult};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::path::{Path, PathBuf};

mod cache;
mod channels;
mod summary;

/// The family-wise error level of the Bonferroni thresholds in the summary and the plots.
const ALPHA: f64 = 1e-5;

/// The conventional TVLA threshold on |t|.
const T_THRESHOLD: f64 = 4.5;

/// Which clock edges open a bin (`--edges`).
#[derive(Clone, Copy, Debug, ValueEnum)]
enum EdgesArg {
    Rising,
    Falling,
    Both,
}

impl From<EdgesArg> for EdgeKind {
    fn from(value: EdgesArg) -> Self {
        match value {
            EdgesArg::Rising => EdgeKind::Rising,
            EdgesArg::Falling => EdgeKind::Falling,
            EdgesArg::Both => EdgeKind::Both,
        }
    }
}

/// What to do when traces have different lengths (`--length-policy`).
#[derive(Clone, Copy, Debug, ValueEnum)]
enum LengthPolicyArg {
    Pad,
    Truncate,
    Error,
}

impl From<LengthPolicyArg> for LengthPolicy {
    fn from(value: LengthPolicyArg) -> Self {
        match value {
            LengthPolicyArg::Pad => LengthPolicy::Pad,
            LengthPolicyArg::Truncate => LengthPolicy::Truncate,
            LengthPolicyArg::Error => LengthPolicy::Error,
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "scasim-tvla")]
#[command(author = "Kamyar Mohajerani <kamyar@kamyar.xyz>")]
#[command(version)]
#[command(about = "Test-Vector Leakage Analysis", long_about = None)]
#[command(group(
    ArgGroup::new("meta")
        .required(true)
        .multiple(true)
        .args(["maybe_metadata", "maybe_meta_list_path", "merge_stats"])
))]
struct Args {
    /// Evaluate this pair of labels. Caches keep every label.
    #[arg(long, num_args=2, default_values=["0","1"])]
    pair: Vec<u16>,
    /// Evaluate this group.
    #[arg(long, conflicts_with = "pool_groups")]
    group: Option<u64>,
    /// Pool all groups explicitly.
    #[arg(long)]
    pool_groups: bool,
    /// Write a validated per-batch statistics cache. Requires one --meta-json batch.
    #[arg(long, requires="maybe_metadata", conflicts_with_all=["maybe_meta_list_path","merge_stats"])]
    stats_out: Option<PathBuf>,
    /// Merge statistics caches in the given order.
    #[arg(long, num_args=1.., conflicts_with_all=["maybe_metadata","maybe_meta_list_path","list_signals","clock","edges","offset","include","exclude","per_scope","depth","shuffle_labels"])]
    merge_stats: Vec<PathBuf>,
    /// Curve checkpoints: every, every:K (K > 0), or final.
    #[arg(long, default_value = "every")]
    curve: String,
    /// Metadata file (`meta.json` or `meta.json.gz`) of one batch.
    #[arg(long = "meta-json", value_name = "META_JSON")]
    maybe_metadata: Option<String>,
    /// File with one metadata file path per line. Relative paths are relative to the directory of
    /// this file. If both options are given, `--meta-list` is used.
    #[arg(long = "meta-list", value_name = "META_LIST_PATH")]
    maybe_meta_list_path: Option<String>,
    #[arg(
        long,
        help = "number of threads to use for parallel processing, defaults to the number of available CPU cores",
        value_name = "NUM_THREADS"
    )]
    num_threads: Option<usize>,
    /// The highest order of t-test to perform
    #[arg(short = 'd', default_value_t = 2, value_parser = clap::value_parser!(u64).range(1..))]
    order: u64,
    #[arg(
        long = "show",
        help = "Show the plots in a web browser",
        required = false,
        action = clap::ArgAction::SetTrue,
    )]
    show_plots: bool,
    #[arg(
        long,
        help = "Plot the t-test results",
        action = clap::ArgAction::Set,
        num_args = 0..=1,
        default_missing_value = "true",
        default_value_t = true
    )]
    plot: bool,
    #[arg(
        long,
        help = "Run the chi-squared test and write its results and plots (`chi2.npz`, `chi2.*`, `max_chi2.*`)",
        action = clap::ArgAction::Set,
        num_args = 0..=1,
        default_missing_value = "true",
        default_value_t = true
    )]
    chi2: bool,
    #[arg(
        long = "use-existing",
        help = "Skip generation of power trace data if `traces.npz` exists next to the metadata file and is newer than the waveform and the metadata file. Use its stored data instead.",
        action = clap::ArgAction::Set,
        num_args = 0..=1,
        default_missing_value = "true",
        default_value_t = true
    )]
    use_existing: bool,
    /// Select signals by rule (repeatable): scope:PATH, signal:PATH, regex:PATTERN, or
    /// module:NAME. The rules of --include and --exclude apply in the order on the command line.
    /// The last rule that matches a signal decides. A signal with several names (aliases)
    /// matches a rule if one of its names matches. Without any rule, all selectable signals are
    /// selected.
    /// If the first rule is an --include, no signal is selected before it. If the first rule is
    /// an --exclude, all signals are. With rules, `traces.npz` files are neither read nor
    /// written.
    #[arg(long = "include", value_name = "KIND:VALUE")]
    include: Vec<String>,
    /// Remove signals from the selection by rule (repeatable). See --include.
    #[arg(long = "exclude", value_name = "KIND:VALUE")]
    exclude: Vec<String>,
    /// Print every selectable signal of the first waveform, with its names and whether the rules
    /// select it, then exit.
    #[arg(long = "list-signals")]
    list_signals: bool,
    /// Sample on the edges of this clock signal (the exact path of a 1-bit signal, see
    /// --list-signals). One sample is the number of toggles of the selected signals between two
    /// edges. The `clock_period` of the metadata is not used. The clock signal stays in the
    /// selection unless you exclude it with `--exclude signal:PATH`. With --clock, `traces.npz`
    /// files are neither read nor written. Without --clock, `tvla` keeps the time points at
    /// multiples of the `clock_period` of the metadata.
    #[arg(long, value_name = "PATH")]
    clock: Option<String>,
    /// Which edges of the --clock open a sample. A change to or from `x` or `z` is not an edge.
    #[arg(long, value_enum, default_value_t = EdgesArg::Rising, requires = "clock")]
    edges: EdgesArg,
    /// Ticks (time units of the waveform) added to every edge of the --clock. Use it to move the
    /// samples, for example to a phase of the clock period. The shifted time must not be below 0.
    #[arg(
        long,
        default_value_t = 0,
        allow_hyphen_values = true,
        requires = "clock"
    )]
    offset: i64,
    /// Rank the leakage by scope. Make one channel for each scope that is exactly --depth
    /// levels below SCOPE and holds selected signals, and one channel named SCOPE for the
    /// signals directly in SCOPE. A signal in a deeper scope belongs to its ancestor at that
    /// depth. A signal with several names (aliases) belongs to the channel of its name with
    /// the deepest scope in SCOPE (the smallest name if several are equally deep). A selected
    /// signal with no name in SCOPE goes to the channel `(outside SCOPE)`. The
    /// selection rules apply first. The usual outputs stay for the whole selection. Also
    /// writes `channels.tsv` (the ranking by max |t|), `channels.txt`, `t_values_channels.npz`,
    /// and with --chi2 `chi2_channels.npz`. Plots only the best channel, into `top_channel/`.
    /// Turns `traces.npz` off.
    #[arg(long = "per-scope", value_name = "SCOPE")]
    per_scope: Option<String>,
    /// The number of scope levels below --per-scope SCOPE.
    #[arg(
        long,
        default_value_t = 1,
        value_parser = clap::value_parser!(u64).range(1..),
        requires = "per_scope"
    )]
    depth: u64,
    /// Null run: shuffle the labels of each batch before the statistics, with a generator seeded
    /// by SEED and the stable batch id. The run is reproducible and the class counts do not change.
    /// All outputs are written as usual. Use it to see how large |t| gets without a leak.
    #[arg(long = "shuffle-labels", value_name = "SEED")]
    shuffle_labels: Option<u64>,
    /// What to do when the traces have different lengths. `pad`: pad shorter traces with zeros.
    /// Both axes grow to the longest batch. `truncate`: cut traces to
    /// the shortest trace of the batch, and to the shortest length of all batches so far. The
    /// final result does not depend on the order of the batches. `error`: any difference is an error. Default: `pad` without
    /// --clock, `error` with --clock. A policy other than `pad` turns `traces.npz` off.
    #[arg(long = "length-policy", value_enum)]
    length_policy: Option<LengthPolicyArg>,
    #[arg(
        long,
        value_name = "PLOTS_OUTPUT_DIR",
        help = "Directory to save the ttest results and plot files",
        default_value = ""
    )]
    ttest_output_dir: String,
}

/// The values of `--include` and `--exclude` as rules (`+kind:value` and `-kind:value`), in the
/// order on the command line. `clap` keeps the values of each flag in separate lists, so the
/// order across the two flags comes from the indices of the values.
fn ordered_rules(matches: &ArgMatches) -> Vec<String> {
    let mut rules: Vec<(usize, String)> = Vec::new();
    for (flag, sign) in [("include", '+'), ("exclude", '-')] {
        if let (Some(indices), Some(values)) =
            (matches.indices_of(flag), matches.get_many::<String>(flag))
        {
            rules.extend(indices.zip(values).map(|(i, v)| (i, format!("{sign}{v}"))));
        }
    }
    rules.sort_by_key(|(index, _)| *index);
    rules.into_iter().map(|(_, rule)| rule).collect()
}

/// Prints every selectable signal of a waveform with its names and the selection result.
fn list_signals(index: &HierarchyIndex, selection: &Selection) -> miette::Result<()> {
    let resolution = selection.resolve(index).into_diagnostic()?;
    let mut selectable = 0;
    for (handle, paths) in index.paths.iter().enumerate() {
        if paths.is_empty() {
            continue;
        }
        selectable += 1;
        let names = paths
            .iter()
            .map(|p| {
                if p.is_alias {
                    format!("{} (alias)", p.path)
                } else {
                    p.path.clone()
                }
            })
            .join(", ");
        let selected = if resolution.selected[handle] {
            "yes"
        } else {
            "no"
        };
        println!("{handle}\t{selected}\t{names}");
    }
    let count = resolution.selected.iter().filter(|&&s| s).count();
    // The summary and the warnings go to stderr. Stdout has only the lines of the signals.
    eprintln!("{count} of {selectable} selectable signals are selected");
    for rule in &resolution.unmatched_rules {
        eprintln!("warning: the rule {rule} matches no signal");
    }
    Ok(())
}

/// Tells the user what the selection covers, and warns about selections that probably do not
/// measure what the user wants.
fn report_selection(batch: &str, d: &BatchDiagnostics) {
    for rule in &d.info.unmatched_rules {
        warn!("{batch}: the rule {rule} matches no signal");
    }
    if d.spans_several_top_scopes() {
        info!(
            "{batch}: the selection spans {} top-level scopes: {}. To measure only the design \
             under test, select it, for example with --include scope:TOP.dut",
            d.info.top_scopes.len(),
            d.info.top_scopes.join(", ")
        );
    }
    if d.sampling_drops_most() {
        warn!(
            "{batch}: the sampling at multiples of the clock period keeps only {} of {} toggles",
            d.kept_toggles, d.total_toggles
        );
    }
    if let Some(e) = &d.edges {
        info!(
            "{batch}: {} clock edges; periods: min {}, max {}, mean {:.2}; first edge at {}; \
             toggles: {} inside the bins, {} before the first edge, {} after the last edge \
             ({:.2}% outside)",
            e.summary.edges,
            e.summary.min_period,
            e.summary.max_period,
            e.summary.mean_period,
            e.summary.first_edge,
            e.inside,
            e.before,
            e.after,
            100.0 * e.outside_fraction()
        );
        if e.offset_varies() {
            warn!(
                "{batch}: the segments start at different places in the clock period. The \
                 distance from a segment start to its first bin is between {} and {} ticks. \
                 Samples of different traces are then at different places in the period",
                e.offset_min, e.offset_max
            );
        }
    }
}

/// How `tvla` turns a batch into traces.
struct BatchSettings<'a> {
    selection: &'a Selection,
    pair: [u16; 2],
    sampling: Sampling,
    policy: LengthPolicy,
    use_existing: bool,
    /// False if `traces.npz` must be neither read nor written.
    cache_allowed: bool,
    /// The scope and the depth of the per-scope channels.
    per_scope: Option<(String, usize)>,
    /// The seed of the shuffle of the labels (`--shuffle-labels`).
    shuffle_seed: Option<u64>,
    stats_common: Option<cache::CommonKey>,
    stats_out: Option<PathBuf>,
}

/// The traces of one per-scope channel in one batch.
struct ScopeTraces {
    name: String,
    handles: usize,
    traces: Array2<f32>,
}

/// One batch: the traces of the whole selection, and the per-scope channels (if asked).
struct BatchResult {
    meta: scasim::batch::BatchMeta,
    identities: Vec<cache::ChannelIdentity>,
    key: Option<cache::CacheKey>,
    total: Loaded,
    scopes: Vec<ScopeTraces>,
    edges: Option<EdgeReport>,
    /// The number of aliased handles, and of selected handles outside the scope.
    aliased: usize,
    outside: usize,
}

/// True if `cache` can replace the computation for the batch. The cache must be newer than the
/// waveform and newer than the metadata file. If the waveform no longer exists, the cache is the
/// only source, so it is used whatever the age of the metadata file.
fn cache_is_fresh(cache: &Path, waveform: &Path, metadata: &Path) -> bool {
    let modified = |path: &Path| std::fs::metadata(path).and_then(|m| m.modified());
    let Ok(cache_time) = modified(cache) else {
        return false;
    };
    if !waveform.exists() {
        return true;
    }
    match (modified(waveform), modified(metadata)) {
        (Ok(waveform_time), Ok(metadata_time)) => {
            cache_time > waveform_time && cache_time > metadata_time
        }
        _ => false,
    }
}

/// Reads the traces and labels of one batch from the cache or computes them from the waveform.
/// Also returns the edge report of the batch, if it was computed in edges mode.
fn batch_data(
    _batch_index: usize,
    metadata_path: &Path,
    settings: &BatchSettings<'_>,
) -> miette::Result<BatchResult> {
    let mut result = load_batch(metadata_path, settings)?;
    if let Some(seed) = settings.shuffle_seed {
        // Stable batch identity makes separate --stats-out jobs match a normal run.
        let id = batch_id(&result.meta, metadata_path);
        let hash = cache::digest(id.as_bytes());
        let identity = u64::from_le_bytes(hash[..8].try_into().expect("eight bytes")) as usize;
        let groups = batch_groups(&result.meta, result.total.labels.len());
        for group in groups.iter().copied().collect::<BTreeSet<_>>() {
            let rows: Vec<usize> = groups
                .iter()
                .enumerate()
                .filter(|(_, g)| **g == group)
                .map(|(i, _)| i)
                .collect();
            let mut labels = result.total.labels.select(Axis(0), &rows);
            shuffle_labels(&mut labels, seed, identity);
            for (i, l) in rows.into_iter().zip(labels) {
                result.total.labels[i] = l;
            }
        }
    }
    if let Some(path) = &settings.stats_out {
        let mut cache = batch_cache(&result)?;
        for channel in &mut cache.channels {
            for h in channel.groups.values_mut() {
                h.compact();
            }
        }
        cache.write(path)?;
    }
    Ok(result)
}

/// Like [`batch_data`], without the shuffle of the labels.
fn load_batch(metadata_path: &Path, settings: &BatchSettings<'_>) -> miette::Result<BatchResult> {
    if !metadata_path.exists() {
        return Err(miette!(
            "the metadata file {} does not exist",
            metadata_path.display()
        ));
    }
    let meta = read_batch_meta(metadata_path)?;
    if meta.v1.is_some() && meta.markers.len() < 2 {
        return Err(miette!(
            "the batch {} has {} traces; a t-test needs at least two traces",
            metadata_path.display(),
            meta.markers.len()
        ));
    }
    let key = settings
        .stats_common
        .as_ref()
        .map(|common| {
            cache::BatchKey::new(&meta).map(|batch| cache::CacheKey {
                common: common.clone(),
                batch,
            })
        })
        .transpose()?;
    let trace_file_path = meta.trace_path.clone();
    let parent_folder_path = metadata_path.parent().unwrap_or(Path::new("."));
    let npz_path = parent_folder_path.join("traces.npz");
    let loaded = |source: PathBuf, (traces, labels): (Array2<f32>, Array1<u16>)| Loaded {
        metadata: metadata_path.to_path_buf(),
        source,
        traces,
        labels,
    };

    if settings.use_existing
        && settings.cache_allowed
        && meta.v1.is_none()
        && npz_path.exists()
        && cache_is_fresh(&npz_path, &trace_file_path, metadata_path)
    {
        println!(
            "Using existing traces and labels from {}",
            npz_path.display()
        );
        let data = read_trace_cache(&npz_path)?;
        require_pair_labels(
            data.1.as_slice().expect("labels are contiguous"),
            settings.pair,
            &metadata_path.display().to_string(),
        )?;
        return Ok(BatchResult {
            meta,
            identities: Vec::new(),
            key,
            total: loaded(npz_path, data),
            scopes: Vec::new(),
            edges: None,
            aliased: 0,
            outside: 0,
        });
    }

    if meta.v1.is_none() && meta.markers.len() <= 1 {
        return Err(miette!(
            "the batch {} has {} traces; a t-test needs at least two traces",
            metadata_path.display(),
            meta.markers.len()
        ));
    }

    println!(
        "Computing power traces from {}...",
        trace_file_path.display()
    );
    let start_time = std::time::Instant::now();
    let mut aliased = 0;
    let mut outside = 0;
    let mut handle_counts = Vec::new();
    let plan = match &settings.per_scope {
        None => PowerPlan::toggles(settings.selection.clone()),
        Some((scope, depth)) => {
            let index = hierarchy_index(&trace_file_path)
                .into_diagnostic()
                .wrap_err_with(|| format!("cannot read {}", trace_file_path.display()))?;
            let selected = settings
                .selection
                .resolve(&index)
                .into_diagnostic()
                .wrap_err("cannot apply the selection rules")?
                .selected;
            let groups = group_by_scope(&index, &selected, scope, *depth);
            let handles: usize = groups.channels.iter().map(|c| c.paths.len()).sum();
            // Only signals outside the scope: the scope name is probably wrong.
            if handles == groups.outside {
                return Err(miette!(
                    "{}: no selected signal is in the scope {scope}, so --per-scope makes no \
                     channel",
                    trace_file_path.display()
                ));
            }
            aliased = groups.aliased;
            outside = groups.outside;
            handle_counts = groups.channels.iter().map(|c| c.paths.len()).collect();
            info!(
                "{}: {} channels below {scope} (depth {depth}); {} signals with several names \
                 (aliases) are in the channel of their deepest name; {} selected signals outside the \
                 scope are in the channel (outside {scope})",
                trace_file_path.display(),
                groups.channels.len(),
                groups.aliased,
                groups.outside
            );
            scope_plan(settings.selection.clone(), &groups)
        }
    };
    let identities = if settings.stats_out.is_some() {
        let index = hierarchy_index(&trace_file_path).into_diagnostic()?;
        plan.channels
            .iter()
            .enumerate()
            .map(|(channel_index, channel)| {
                let selected = channel
                    .selection
                    .resolve(&index)
                    .into_diagnostic()?
                    .selected;
                let handles: Vec<_> = selected
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| **s)
                    .map(|(h, _)| {
                        let mut paths: Vec<_> =
                            index.paths[h].iter().map(|p| p.path.clone()).collect();
                        paths.sort();
                        (h, paths)
                    })
                    .collect();
                Ok(cache::ChannelIdentity {
                    name: if channel_index == 0 {
                        "total".into()
                    } else {
                        channel.name.clone()
                    },
                    is_total: channel_index == 0,
                    handles: handles.len(),
                    handles_hash: cache::digest(&serde_json::to_vec(&handles).into_diagnostic()?),
                })
            })
            .collect::<miette::Result<Vec<_>>>()?
    } else {
        Vec::new()
    };
    let output = compute_batch(&meta, &plan, &settings.sampling, settings.policy)?;
    require_pair_labels(
        output.labels.as_slice().expect("labels are contiguous"),
        settings.pair,
        &metadata_path.display().to_string(),
    )?;
    if let Some(key) = &key {
        let stat = std::fs::metadata(&trace_file_path).into_diagnostic()?;
        let mtime = stat
            .modified()
            .into_diagnostic()?
            .duration_since(std::time::UNIX_EPOCH)
            .into_diagnostic()?;
        if stat.len() != key.batch.waveform_size
            || mtime.as_secs() != key.batch.waveform_mtime_seconds
            || mtime.subsec_nanos() != key.batch.waveform_mtime_nanos
        {
            return Err(miette!(
                "{} changed during analysis",
                trace_file_path.display()
            ));
        }
    }
    let diagnostics = output.diagnostics;
    let labels_array = output.labels;
    let mut channel_traces = output.channels.into_iter();
    let traces_array = channel_traces.next().expect("the plan has a channel");
    let scopes: Vec<ScopeTraces> = plan
        .channels
        .iter()
        .skip(1)
        .zip(handle_counts)
        .zip(channel_traces)
        .map(|((spec, handles), traces)| ScopeTraces {
            name: spec.name.clone(),
            handles,
            traces,
        })
        .collect();
    let (num_traces, cur_samples_per_trace) = traces_array.dim();
    println!(
        "Computed {num_traces} traces with up to {cur_samples_per_trace} samples in {:.2}s",
        start_time.elapsed().as_secs_f32()
    );
    info!(
        "{}: {} signals selected; toggles: {} in total, {} at the sampled time points, \
         {} inside the segments",
        trace_file_path.display(),
        diagnostics.info.selected_handles,
        diagnostics.total_toggles,
        diagnostics.kept_toggles,
        diagnostics.segment_toggles
    );
    report_selection(&trace_file_path.display().to_string(), &diagnostics);

    if settings.cache_allowed && meta.v1.is_none() {
        println!("Saving traces and labels to NPZ file...");
        let start_time = std::time::Instant::now();
        write_trace_cache(&npz_path, &traces_array, &labels_array)?;
        println!(
            "Saved traces and labels to {} in {:.2}s\n",
            npz_path.display(),
            start_time.elapsed().as_secs_f32()
        );
    }
    Ok(BatchResult {
        meta,
        identities,
        key,
        total: loaded(trace_file_path, (traces_array, labels_array)),
        scopes,
        edges: diagnostics.edges,
        aliased,
        outside,
    })
}

fn batch_id(meta: &scasim::batch::BatchMeta, path: &Path) -> String {
    meta.v1
        .as_ref()
        .map(|v| v.batch.id.clone())
        .unwrap_or_else(|| {
            format!(
                "legacy:{}",
                path.canonicalize()
                    .unwrap_or_else(|_| path.to_path_buf())
                    .display()
            )
        })
}
fn batch_groups(meta: &scasim::batch::BatchMeta, rows: usize) -> Vec<u64> {
    meta.v1
        .as_ref()
        .map(|v| v.segments.iter().map(|s| s.group).collect())
        .unwrap_or_else(|| vec![0; rows])
}
fn batch_names(meta: &scasim::batch::BatchMeta) -> (BTreeMap<u16, String>, BTreeMap<u64, String>) {
    if let Some(v) = &meta.v1 {
        (v.labels.clone(), v.groups.clone())
    } else {
        (
            meta.markers
                .iter()
                .map(|m| {
                    (
                        m.2,
                        match m.2 {
                            0 => "fixed".into(),
                            1 => "random".into(),
                            n => n.to_string(),
                        },
                    )
                })
                .collect(),
            BTreeMap::from([(0, "default".into())]),
        )
    }
}
fn merge_names<K: Ord + std::fmt::Display>(
    dest: &mut BTreeMap<K, String>,
    source: BTreeMap<K, String>,
    kind: &str,
) -> miette::Result<()> {
    for (id, name) in source {
        if dest.get(&id).is_some_and(|n| n != &name) {
            return Err(miette!("{kind} {id} has different names"));
        }
        dest.insert(id, name);
    }
    Ok(())
}
fn batch_cache(result: &BatchResult) -> miette::Result<cache::Cache> {
    let (labels, groups) = batch_names(&result.meta);
    let group_ids = batch_groups(&result.meta, result.total.labels.len());
    let mut counts: BTreeMap<u64, BTreeMap<u16, u64>> = BTreeMap::new();
    for (&group, &label) in group_ids.iter().zip(&result.total.labels) {
        *counts.entry(group).or_default().entry(label).or_default() += 1;
    }
    let matrices =
        std::iter::once(&result.total.traces).chain(result.scopes.iter().map(|s| &s.traces));
    let channels = matrices
        .zip(&result.identities)
        .map(|(traces, identity)| {
            let mut groups = BTreeMap::new();
            for group in counts.keys() {
                let rows: Vec<usize> = group_ids
                    .iter()
                    .enumerate()
                    .filter(|(_, g)| *g == group)
                    .map(|(i, _)| i)
                    .collect();
                let mut hist = HistAccumulator::new(traces.ncols(), scasim::stats::Binning::Exact);
                hist.update(
                    traces.select(Axis(0), &rows).view(),
                    result.total.labels.select(Axis(0), &rows).view(),
                )
                .into_diagnostic()?;
                if hist.rejected() > 0 {
                    return Err(miette!(
                        "the traces from {} (batch {}) have {} values that are not integers or are 2^53 or larger. The statistics need integer-valued traces",
                        result.total.source.display(),
                        result.total.metadata.display(),
                        hist.rejected()
                    ));
                }
                groups.insert(*group, hist);
            }
            Ok(cache::Channel {
                identity: identity.clone(),
                groups,
            })
        })
        .collect::<miette::Result<Vec<_>>>()?;
    Ok(cache::Cache {
        key: result.key.clone().expect("statistics output has a key"),
        batch_id: batch_id(&result.meta, &result.total.metadata),
        axis: cache::SampleAxis {
            relative_to_segment: true,
            length: result.total.traces.ncols(),
        },
        counts,
        labels,
        groups,
        channels,
        extensions: result
            .meta
            .v1
            .as_ref()
            .map(|v| v.extensions.to_string())
            .unwrap_or_else(|| "{}".into()),
        edges: result.edges.clone(),
        aliased: result.aliased,
        outside: result.outside,
    })
}

fn require_pair_counts(hist: &HistAccumulator, pair: [u16; 2], batch: &str) -> miette::Result<()> {
    for label in pair {
        if hist.class_count(label) == 0 {
            return Err(miette!(
                "{batch}: --pair label {label} is absent from the batch"
            ));
        }
    }
    Ok(())
}

fn require_pair_labels(labels: &[u16], pair: [u16; 2], batch: &str) -> miette::Result<()> {
    for label in pair {
        if !labels.contains(&label) {
            return Err(miette!(
                "{batch}: --pair label {label} is absent from the batch"
            ));
        }
    }
    Ok(())
}

fn unknown_policy_cache_name(policy: scasim::power::UnknownPolicy) -> String {
    policy.cache_key_name().into()
}

/// The fold of one per-scope channel.
struct ScopeFold {
    name: String,
    handles: usize,
    fold: Fold,
}

/// Adds the traces of the channels of one batch to their folds. The first batch makes the
/// folds. The channels are the same in all batches. The folds run in parallel.
fn add_to_scope_folds(
    folds: &mut Vec<ScopeFold>,
    total: &Loaded,
    scopes: Vec<ScopeTraces>,
    order: usize,
    chi2: bool,
    policy: LengthPolicy,
    pair: [u16; 2],
) -> miette::Result<()> {
    if folds.is_empty() {
        folds.extend(scopes.iter().map(|s| {
            let mut fold = Fold::without_curves(order, chi2);
            fold.policy = policy;
            fold.pair = pair;
            ScopeFold {
                name: s.name.clone(),
                handles: s.handles,
                fold,
            }
        }));
    } else if folds
        .iter()
        .map(|f| &f.name)
        .ne(scopes.iter().map(|s| &s.name))
    {
        return Err(miette!(
            "the batch {} has other per-scope channels than the first batch. All batches must \
             have the same scopes",
            total.metadata.display()
        ));
    }
    folds
        .par_iter_mut()
        .zip(scopes.into_par_iter())
        .try_for_each(|(f, scope)| {
            f.fold.add(Loaded {
                metadata: total.metadata.clone(),
                source: total.source.clone(),
                traces: scope.traces,
                labels: total.labels.clone(),
            })
        })
}

fn main() -> miette::Result<()> {
    // set default log level to info
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp(None)
        .init();

    let matches = Args::command().get_matches();
    let mut args = Args::from_arg_matches(&matches).into_diagnostic()?;
    let rules = ordered_rules(&matches);
    let selection = Selection::parse(&rules)
        .into_diagnostic()
        .wrap_err("invalid value for --include or --exclude")?;

    let pair = [args.pair[0], args.pair[1]];
    if pair[0] == pair[1] {
        return Err(miette!("--pair needs two different labels"));
    }
    let merging = !args.merge_stats.is_empty();
    let filenames: Vec<PathBuf> = if merging {
        Vec::new()
    } else if let Some(meta_list_path) = &args.maybe_meta_list_path {
        read_path_list(Path::new(meta_list_path))?
    } else if let Some(filename) = &args.maybe_metadata {
        vec![PathBuf::from(filename)]
    } else {
        // `clap` requires one of the two options.
        unreachable!("clap requires --meta-json or --meta-list");
    };
    if filenames.is_empty() && !merging {
        return Err(miette!(
            "the meta list file {} has no metadata files",
            args.maybe_meta_list_path.as_deref().unwrap_or_default()
        ));
    }
    if args.list_signals {
        let meta = read_batch_meta(&filenames[0])?;
        let index = hierarchy_index(&meta.trace_path)
            .into_diagnostic()
            .wrap_err_with(|| format!("cannot read {}", meta.trace_path.display()))?;
        return list_signals(&index, &selection);
    }
    let order = usize::try_from(args.order)
        .into_diagnostic()
        .wrap_err("the order is too large")?;

    let sampling = match &args.clock {
        Some(clock) => Sampling::Edges(EdgeSampling {
            clock: clock.clone(),
            kind: args.edges.into(),
            offset: args.offset,
        }),
        None => Sampling::Legacy,
    };
    let mut policy = args
        .length_policy
        .map(LengthPolicy::from)
        .unwrap_or(match sampling {
            Sampling::Legacy => LengthPolicy::Pad,
            Sampling::Edges(_) => LengthPolicy::Error,
        });
    let mut fold = if args.curve == "final" {
        Fold::without_curves(order, args.chi2)
    } else {
        Fold::new(order, args.chi2)
    };
    fold.set_curve(&args.curve)?;
    fold.pair = pair;
    fold.policy = policy;

    // `traces.npz` holds the traces of the default selection (all signals), sampled at the
    // multiples of the clock period of the metadata, and padded to the longest trace. A run with
    // rules, with --clock, or with a policy other than `pad` computes other traces. It must not
    // reuse the file, and it must not overwrite it, because a later run without these options
    // would then read these traces.
    let cache_allowed = rules.is_empty()
        && args.clock.is_none()
        && args.per_scope.is_none()
        && policy == LengthPolicy::Pad;
    let stats_common = cache::CommonKey {
        settings: cache::Settings {
            sampling: if args.clock.is_some() {
                "edges"
            } else {
                "legacy"
            }
            .into(),
            clock: args.clock.clone(),
            edges: format!("{:?}", args.edges).to_lowercase(),
            offset: args.offset,
            rules: rules.clone(),
            per_scope: args.per_scope.clone(),
            depth: args.depth as usize,
            policy,
            shuffle_seed: args.shuffle_labels,
            unknown: unknown_policy_cache_name(scasim::power::UnknownPolicy::default()),
        },
        ..cache::CommonKey::default()
    };
    let settings = BatchSettings {
        selection: &selection,
        pair,
        sampling,
        policy,
        use_existing: args.use_existing,
        cache_allowed: cache_allowed && args.stats_out.is_none(),
        stats_common: args.stats_out.as_ref().map(|_| stats_common.clone()),
        stats_out: args.stats_out.clone(),
        per_scope: args
            .per_scope
            .clone()
            .map(|scope| (scope, args.depth as usize)),
        shuffle_seed: args.shuffle_labels,
    };
    if let Some(clock) = &args.clock {
        info!(
            "Sampling on the {:?} edges of {clock}. The clock signal is part of the selection \
             unless you exclude it with --exclude signal:{clock}",
            EdgeKind::from(args.edges)
        );
    }
    let mut edge_totals = summary::EdgeTotals::default();

    if let Some(n) = args.num_threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .into_diagnostic()
            .wrap_err("cannot set the number of threads")?;
    }

    let threads = rayon::current_num_threads();
    println!("Using {threads} threads for parallel processing");

    // Load the batches in windows of one batch per thread, in parallel. Fold each window in the
    // order of the meta list, then drop it. At most one window of traces is in memory. The
    // histograms hold exact counts, so the result does not depend on the window size. With
    // per-scope channels, a batch holds the traces of all channels, so a window is one batch.
    let window_size = if settings.per_scope.is_some() {
        1
    } else {
        threads
    };
    let mut scope_folds: Vec<ScopeFold> = Vec::new();
    let (mut aliased, mut outside) = (0, 0);
    let mut group_names = BTreeMap::new();
    let mut label_names = BTreeMap::new();
    if merging {
        let mut state = cache::MergeState::default();
        for (batch_index, path) in args.merge_stats.iter().enumerate() {
            let cache = cache::read(path)?;
            if batch_index == 0 {
                let cached = &cache.key.common.settings;
                if args
                    .length_policy
                    .is_some_and(|p| LengthPolicy::from(p) != cached.policy)
                {
                    return Err(miette!("--length-policy differs from the cache key"));
                }
                policy = cached.policy;
                fold.policy = policy;
                args.per_scope = cached.per_scope.clone();
                args.depth = cached.depth as u64;
                args.shuffle_labels = cached.shuffle_seed;
            }
            state.add(cache.clone())?;
            let all = state.cache.as_ref().expect("a cache was added");
            if args.group.is_none() && !args.pool_groups && all.groups.len() > 1 {
                return Err(miette!(
                    "more than one group exists; use --group G or --pool-groups"
                ));
            }
            group_names = all.groups.clone();
            if let Some(edges) = &cache.edges {
                edge_totals.add(edges);
            }
            (aliased, outside) = (cache.aliased, cache.outside);
            let total = cache
                .channels
                .iter()
                .position(|c| c.identity.is_total)
                .expect("cache validation requires total");
            let hist = cache.selected(total, args.group, args.pool_groups)?;
            require_pair_counts(&hist, pair, &path.display().to_string())?;
            fold.add_histogram(hist)?;
            if scope_folds.is_empty() {
                for channel in cache.channels.iter().filter(|c| !c.identity.is_total) {
                    let mut fold = Fold::without_curves(order, args.chi2);
                    fold.policy = policy;
                    fold.pair = pair;
                    scope_folds.push(ScopeFold {
                        name: channel.identity.name.clone(),
                        handles: channel.identity.handles,
                        fold,
                    });
                }
            }
            for f in &mut scope_folds {
                let index = cache
                    .channels
                    .iter()
                    .position(|c| !c.identity.is_total && c.identity.name == f.name)
                    .expect("channel identities were checked");
                f.fold
                    .add_histogram(cache.selected(index, args.group, args.pool_groups)?)?;
            }
        }
    } else {
        for (window_number, window) in filenames.chunks(window_size).enumerate() {
            let first_index = window_number * window_size;
            let loaded: Vec<BatchResult> = window
                .par_iter()
                .enumerate()
                .map(|(i, metadata_path)| batch_data(first_index + i, metadata_path, &settings))
                .collect::<miette::Result<_>>()?;
            for mut result in loaded {
                let (labels, groups) = batch_names(&result.meta);
                merge_names(&mut label_names, labels, "label")?;
                merge_names(&mut group_names, groups, "group")?;
                if args.group.is_none() && !args.pool_groups && group_names.len() > 1 {
                    return Err(miette!(
                        "more than one group exists; use --group G or --pool-groups"
                    ));
                }
                let group = args.group.or_else(|| {
                    (!args.pool_groups)
                        .then(|| group_names.keys().next().copied())
                        .flatten()
                });
                if let Some(group) = group {
                    let groups = batch_groups(&result.meta, result.total.labels.len());
                    let rows: Vec<usize> = groups
                        .iter()
                        .enumerate()
                        .filter(|(_, g)| **g == group)
                        .map(|(i, _)| i)
                        .collect();
                    if rows.len() != result.total.labels.len() {
                        result.total.traces = result.total.traces.select(Axis(0), &rows);
                        result.total.labels = result.total.labels.select(Axis(0), &rows);
                        for scope in &mut result.scopes {
                            scope.traces = scope.traces.select(Axis(0), &rows);
                        }
                    }
                }
                require_pair_labels(
                    result
                        .total
                        .labels
                        .as_slice()
                        .expect("labels are contiguous"),
                    pair,
                    &result.total.metadata.display().to_string(),
                )?;
                if let Some(edges) = &result.edges {
                    edge_totals.add(edges);
                }
                (aliased, outside) = (result.aliased, result.outside);
                if !result.scopes.is_empty() {
                    add_to_scope_folds(
                        &mut scope_folds,
                        &result.total,
                        result.scopes,
                        order,
                        args.chi2,
                        policy,
                        pair,
                    )?;
                }
                fold.add(result.total)?;
            }
        }
    }
    if let Some(group) = args.group
        && !group_names.contains_key(&group)
    {
        return Err(miette!("group {group} does not exist"));
    }
    fold.finish()?;
    if edge_totals.batches > 1
        && edge_totals.offsets_differ()
        && let Some((min, max)) = edge_totals.offset_range()
    {
        warn!(
            "the segments of the batches start at different places in the clock period. The \
             distance from a segment start to its first bin is between {min} and {max} ticks \
             across the batches. Samples of different traces are then at different places in the \
             period"
        );
    }
    let total_collected_traces = fold.num_traces.last().copied().unwrap_or(0);
    let t_values = fold
        .t_values
        .take()
        .ok_or_else(|| miette!("there is no batch to analyze"))?;
    if t_values.iter().any(|t| !t.is_finite()) {
        warn!(
            "some t-values are not finite (for example, a sample is constant or a class has too \
             few traces). Infinite maxima are kept; undefined values are counted separately"
        );
    }

    log::info!("Total number of traces: {}", total_collected_traces);

    // The thresholds of the summary and of the chi-squared plots.
    let samples = fold.samples;
    let t_family = family_size(1, samples as u64, order as u64)
        .ok_or_else(|| miette!("the number of t-tests does not fit in 64 bits"))?;
    let chi2_thresholds = [CONVENTIONAL, bonferroni(ALPHA, samples as u64)];
    let chi2_report = fold
        .chi2_results
        .as_deref()
        .map(|results| summary::chi2_report(results, chi2_thresholds));
    let total_memory = fold.hist.as_ref().map_or(0, HistAccumulator::memory_bytes);
    let channels_memory: usize = scope_folds
        .iter()
        .filter_map(|f| f.fold.hist.as_ref())
        .map(HistAccumulator::memory_bytes)
        .sum();
    let memory_bytes = total_memory + channels_memory;
    for f in &mut scope_folds {
        f.fold.finish()?;
    }
    let channel_t: Vec<Array2<f64>> = scope_folds
        .iter_mut()
        .map(|f| f.fold.t_values.take().expect("a batch was added"))
        .collect();
    let channel_results: Vec<channels::ChannelResult<'_>> = scope_folds
        .iter()
        .zip(&channel_t)
        .map(|(f, t)| channels::ChannelResult {
            name: &f.name,
            handles: f.handles,
            t_values: t,
            chi2: f.fold.chi2_results.as_deref(),
        })
        .collect();
    let ranking = channels::rank_channels(&channel_results);
    let channels_summary = args
        .per_scope
        .as_ref()
        .map(|scope| summary::ChannelsSummary {
            scope: scope.clone(),
            depth: args.depth as usize,
            count: ranking.len(),
            aliased,
            outside,
            memory_bytes: channels_memory,
            top: ranking.iter().take(5).map(summary::describe_rank).collect(),
        });
    info!(
        "{}",
        summary::render(&summary::SummaryInput {
            t_values: t_values.view(),
            conventional: T_THRESHOLD,
            alpha: ALPHA,
            bonferroni: t_bonferroni(ALPHA, t_family),
            family: t_family,
            chi2: chi2_report,
            memory_bytes,
            edges: (edge_totals.batches > 0).then_some(edge_totals),
            channels: channels_summary,
            shuffle_seed: args.shuffle_labels,
        })
    );
    if let Some(c) = &chi2_report
        && c.summary.failed > 0
    {
        warn!(
            "{} chi-squared p-values failed to converge. They are not in the maxima and counts",
            c.summary.failed
        );
    }

    let output_dir = PathBuf::from(&args.ttest_output_dir);
    create_output_dir(&output_dir)?;
    write_t_values_npz(&output_dir, &t_values)?;
    write_curves(&output_dir, &fold)?;

    if let Some(results) = &fold.chi2_results {
        write_chi2_npz(&output_dir.join("chi2.npz"), results)?;
    }
    if !channel_results.is_empty() {
        channels::write_channel_files(&output_dir, &channel_results)?;
    }

    if args.plot {
        if let Some(top) = ranking.first() {
            let dir = output_dir.join("top_channel");
            create_output_dir(&dir)?;
            info!(
                "Plotting the t-values of the best channel {} into {}",
                top.name,
                dir.display()
            );
            plot_t_traces(
                channel_t[top.index].view(),
                Some(T_THRESHOLD),
                false,
                &dir,
                args.show_plots,
            )?;
        }
        plot_t_traces(
            t_values.view(),
            Some(T_THRESHOLD),
            false, // abs_values
            &output_dir,
            args.show_plots,
        )?;

        if fold.max_t.first().is_some_and(|row| row.len() > 1) {
            plot_max_t_values(
                &fold.max_t,
                &fold.curve_traces,
                Some(T_THRESHOLD),
                &output_dir,
                args.show_plots,
            )?;
        }

        if let Some(results) = &fold.chi2_results {
            plot_chi2(
                results,
                &fold.max_chi2,
                &fold.curve_traces,
                chi2_thresholds[1],
                &output_dir,
                args.show_plots,
            )?;
        }
    }

    Ok(())
}

/// Saves descriptive checkpoints, including infinite maxima and undefined counts.
fn write_curves(dir: &Path, fold: &Fold) -> miette::Result<()> {
    if fold.curve_traces.len() <= 1 {
        return Ok(());
    }
    let mut text =
        String::from("# Descriptive repeated looks. This curve is not an inference.\ntraces");
    for order in 1..=fold.order {
        text.push_str(&format!("\tmax_abs_t_d{order}\tundefined_d{order}"));
    }
    if fold.chi2 {
        text.push_str("\tmax_chi2");
    }
    text.push('\n');
    for (i, count) in fold.curve_traces.iter().enumerate() {
        text.push_str(&count.to_string());
        for order in 0..fold.order {
            text.push_str(&format!(
                "\t{}\t{}",
                fold.max_t[order][i], fold.undefined_t[order][i]
            ));
        }
        if fold.chi2 {
            text.push_str(&format!("\t{}", fold.max_chi2[i]));
        }
        text.push('\n');
    }
    std::fs::write(dir.join("curves.tsv"), text).into_diagnostic()
}

/// Writes the chi-squared results of all samples to a compressed `.npz` file.
fn write_chi2_npz(path: &Path, results: &[TestResult]) -> miette::Result<()> {
    info!("Saving chi-squared results to {}", path.display());
    let column = |f: fn(&TestResult) -> f64| Array1::from_iter(results.iter().map(f));
    let mut npz = NpzWriter::new_compressed(
        File::create(path)
            .into_diagnostic()
            .wrap_err_with(|| format!("cannot create {}", path.display()))?,
    );
    let add = |name: &str, result: Result<(), _>| {
        result
            .into_diagnostic()
            .wrap_err_with(|| format!("cannot write {name} to {}", path.display()))
    };
    add(
        "neg_log10_p",
        npz.add_array("neg_log10_p", &column(|r| r.neg_log10_p)),
    )?;
    add(
        "statistic",
        npz.add_array("statistic", &column(|r| r.statistic)),
    )?;
    add(
        "dof",
        npz.add_array("dof", &Array1::from_iter(results.iter().map(|r| r.dof))),
    )?;
    add(
        "columns",
        npz.add_array(
            "columns",
            &Array1::from_iter(results.iter().map(|r| r.columns)),
        ),
    )?;
    add(
        "merged",
        npz.add_array(
            "merged",
            &Array1::from_iter(results.iter().map(|r| r.merged)),
        ),
    )?;
    add(
        "min_expected",
        npz.add_array("min_expected", &column(|r| r.min_expected)),
    )?;
    add(
        "n",
        npz.add_array("n", &Array1::from_iter(results.iter().map(|r| r.n))),
    )?;
    npz.finish()
        .into_diagnostic()
        .wrap_err_with(|| format!("cannot write {}", path.display()))?;
    Ok(())
}

/// Plots -log10(p) per sample (`chi2.*`) and its maximum versus the number of traces
/// (`max_chi2.*`).
fn plot_chi2(
    results: &[TestResult],
    max_chi2: &[f64],
    num_traces: &[usize],
    bonferroni_threshold: f64,
    output_dir: &Path,
    show: bool,
) -> miette::Result<()> {
    let thresholds = vec![
        Threshold::new(CONVENTIONAL, "5"),
        Threshold::new(bonferroni_threshold, "Bonferroni"),
    ];
    let per_sample: Vec<f64> = results.iter().map(|r| r.neg_log10_p).collect();
    let opts = LineOptions {
        y_label: "-log10(p)".into(),
        thresholds: thresholds.clone(),
        symmetric: false,
        ..LineOptions::t_values()
    };
    plot_series(
        "chi2",
        &[Series::indexed("chi2", &per_sample)],
        &opts,
        output_dir,
        show,
    )?;
    if num_traces.len() <= 1 {
        return Ok(());
    }
    let x: Vec<f64> = num_traces.iter().map(|&n| n as f64).collect();
    let opts = LineOptions {
        y_label: "max(-log10(p))".into(),
        thresholds,
        symmetric: false,
        ..LineOptions::max_t()
    };
    plot_series(
        "max_chi2",
        &[Series::with_x("max chi2", &x, max_chi2)],
        &opts,
        output_dir,
        show,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn rules(args: &[&str]) -> Vec<String> {
        let mut argv = vec!["tvla", "--meta-json", "meta.json"];
        argv.extend_from_slice(args);
        ordered_rules(&Args::command().try_get_matches_from(argv).unwrap())
    }

    #[test]
    fn rules_keep_the_command_line_order_across_both_flags() {
        assert_eq!(
            rules(&[
                "--exclude",
                "scope:tb.dut.u_rng",
                "--include",
                "signal:tb.dut.u_rng.state",
                "--exclude=regex:.*clk"
            ]),
            [
                "-scope:tb.dut.u_rng",
                "+signal:tb.dut.u_rng.state",
                "-regex:.*clk"
            ]
        );
        assert_eq!(
            rules(&[
                "--include",
                "scope:a",
                "--exclude",
                "signal:a.b",
                "--include",
                "scope:c"
            ]),
            ["+scope:a", "-signal:a.b", "+scope:c"]
        );
    }

    #[test]
    fn no_flags_give_no_rules() {
        assert!(rules(&[]).is_empty());
        assert!(rules(&["--list-signals"]).is_empty());
    }

    #[test]
    fn batch_cache_reports_rejected_non_integer_trace_values() {
        let metadata = PathBuf::from("meta.json");
        let result = BatchResult {
            meta: scasim::batch::BatchMeta {
                metadata_path: metadata.clone(),
                v1: None,
                trace_path: PathBuf::from("wave.vcd"),
                clock_period: None,
                markers: vec![(0, 1, 0), (1, 2, 1)],
            },
            identities: vec![cache::ChannelIdentity {
                name: "total".into(),
                is_total: true,
                handles_hash: [0; 32],
                handles: 1,
            }],
            key: Some(cache::CacheKey {
                common: cache::CommonKey::default(),
                batch: cache::BatchKey::default(),
            }),
            total: Loaded {
                metadata: metadata.clone(),
                source: PathBuf::from("traces"),
                traces: array![[0.5_f32], [1.0]],
                labels: array![0_u16, 1],
            },
            scopes: Vec::new(),
            edges: None,
            aliased: 0,
            outside: 0,
        };
        let err = batch_cache(&result).unwrap_err().to_string();
        assert!(
            err.contains("not integers") && err.contains("meta.json"),
            "{err}"
        );
    }

    #[test]
    fn cache_key_uses_the_selected_unknown_policy() {
        assert_eq!(
            unknown_policy_cache_name(scasim::power::UnknownPolicy::AsOne),
            "as-one"
        );
    }
}
