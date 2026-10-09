use clap::{ArgGroup, ArgMatches, CommandFactory, FromArgMatches, Parser, ValueEnum};
use itertools::Itertools;
use log::*;
use miette::{IntoDiagnostic, WrapErr, miette};
use ndarray::{Array1, Array2};
use ndarray_npz::NpzWriter;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use scasim::batch::{
    BatchDiagnostics, EdgeReport, EdgeSampling, LengthPolicy, Sampling, compute_batch,
    read_batch_meta, read_trace_cache, write_trace_cache,
};
use scasim::fold::{Fold, Loaded, create_output_dir, read_path_list, write_t_values_npz};
use scasim::hierarchy::{HierarchyIndex, Selection};
use scasim::plot::*;
use scasim::power::edges::EdgeKind;
use scasim::power::{PowerPlan, hierarchy_index};
use scasim::stats::threshold::{CONVENTIONAL, bonferroni, family_size, t_bonferroni};
use scasim::stats::{HistAccumulator, TestResult};
use std::fs::File;
use std::path::{Path, PathBuf};

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
        .args(["maybe_metadata", "maybe_meta_list_path"])
))]
struct Args {
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
    /// What to do when the traces have different lengths. `pad`: pad shorter traces with zeros.
    /// A later batch is padded or cut to the length of the first batch. `truncate`: cut traces to
    /// the shortest trace of the batch, and a longer batch to the length of the first batch. A
    /// shorter batch is an error. `error`: any difference is an error. Default: `pad` without
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
    sampling: Sampling,
    policy: LengthPolicy,
    use_existing: bool,
    /// False if `traces.npz` must be neither read nor written.
    cache_allowed: bool,
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
    metadata_path: &Path,
    settings: &BatchSettings<'_>,
) -> miette::Result<(Loaded, Option<EdgeReport>)> {
    if !metadata_path.exists() {
        return Err(miette!(
            "the metadata file {} does not exist",
            metadata_path.display()
        ));
    }
    let meta = read_batch_meta(metadata_path)?;
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
        && npz_path.exists()
        && cache_is_fresh(&npz_path, &trace_file_path, metadata_path)
    {
        println!(
            "Using existing traces and labels from {}",
            npz_path.display()
        );
        let data = read_trace_cache(&npz_path)?;
        return Ok((loaded(npz_path, data), None));
    }

    println!(
        "Computing power traces from {}...",
        trace_file_path.display()
    );
    let start_time = std::time::Instant::now();
    let plan = PowerPlan::toggles(settings.selection.clone());
    let output = compute_batch(&meta, &plan, &settings.sampling, settings.policy)?;
    let diagnostics = output.diagnostics;
    let labels_array = output.labels;
    let traces_array = output
        .channels
        .into_iter()
        .next()
        .expect("the plan has one channel");
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

    if settings.cache_allowed {
        println!("Saving traces and labels to NPZ file...");
        let start_time = std::time::Instant::now();
        write_trace_cache(&npz_path, &traces_array, &labels_array)?;
        println!(
            "Saved traces and labels to {} in {:.2}s\n",
            npz_path.display(),
            start_time.elapsed().as_secs_f32()
        );
    }
    Ok((
        loaded(trace_file_path, (traces_array, labels_array)),
        diagnostics.edges,
    ))
}

fn main() -> miette::Result<()> {
    // set default log level to info
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp(None)
        .init();

    let matches = Args::command().get_matches();
    let args = Args::from_arg_matches(&matches).into_diagnostic()?;
    let rules = ordered_rules(&matches);
    let selection = Selection::parse(&rules)
        .into_diagnostic()
        .wrap_err("invalid value for --include or --exclude")?;

    let filenames: Vec<PathBuf> = if let Some(meta_list_path) = &args.maybe_meta_list_path {
        read_path_list(Path::new(meta_list_path))?
    } else if let Some(filename) = &args.maybe_metadata {
        vec![PathBuf::from(filename)]
    } else {
        // `clap` requires one of the two options.
        unreachable!("clap requires --meta-json or --meta-list");
    };
    if filenames.is_empty() {
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
    let policy = args
        .length_policy
        .map(LengthPolicy::from)
        .unwrap_or(match sampling {
            Sampling::Legacy => LengthPolicy::Pad,
            Sampling::Edges(_) => LengthPolicy::Error,
        });
    let mut fold = Fold::new(order, args.chi2);
    fold.policy = policy;

    // `traces.npz` holds the traces of the default selection (all signals), sampled at the
    // multiples of the clock period of the metadata, and padded to the longest trace. A run with
    // rules, with --clock, or with a policy other than `pad` computes other traces. It must not
    // reuse the file, and it must not overwrite it, because a later run without these options
    // would then read these traces.
    let cache_allowed = rules.is_empty() && args.clock.is_none() && policy == LengthPolicy::Pad;
    let settings = BatchSettings {
        selection: &selection,
        sampling,
        policy,
        use_existing: args.use_existing,
        cache_allowed,
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
    // histograms hold exact counts, so the result does not depend on the window size.
    for window in filenames.chunks(threads) {
        let loaded: Vec<(Loaded, Option<EdgeReport>)> = window
            .par_iter()
            .map(|metadata_path| batch_data(metadata_path, &settings))
            .collect::<miette::Result<_>>()?;
        for (batch, edges) in loaded {
            if let Some(edges) = &edges {
                edge_totals.add(edges);
            }
            fold.add(batch)?;
        }
    }
    let total_collected_traces = fold.num_traces.last().copied().unwrap_or(0);
    let t_values = fold
        .t_values
        .take()
        .ok_or_else(|| miette!("there is no batch to analyze"))?;
    if t_values.iter().any(|t| !t.is_finite()) {
        warn!(
            "some t-values are not finite (for example, a sample is constant or a class has too \
             few traces). The maximum |t| ignores them and is NaN if no t-value is finite"
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
    let memory_bytes = fold.hist.as_ref().map_or(0, HistAccumulator::memory_bytes);
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

    if let Some(results) = &fold.chi2_results {
        write_chi2_npz(&output_dir.join("chi2.npz"), results)?;
    }

    if args.plot {
        plot_t_traces(
            t_values.view(),
            Some(T_THRESHOLD),
            false, // abs_values
            &output_dir,
            args.show_plots,
        )?;

        plot_max_t_values(
            &fold.max_t,
            &fold.num_traces,
            Some(T_THRESHOLD),
            &output_dir,
            args.show_plots,
        )?;

        if let Some(results) = &fold.chi2_results {
            plot_chi2(
                results,
                &fold.max_chi2,
                &fold.num_traces,
                chi2_thresholds[1],
                &output_dir,
                args.show_plots,
            )?;
        }
    }

    Ok(())
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
}
