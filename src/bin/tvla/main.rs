use clap::{ArgGroup, ArgMatches, CommandFactory, FromArgMatches, Parser};
use itertools::Itertools;
use log::*;
use miette::{IntoDiagnostic, WrapErr, miette};
use ndarray::{Array1, Array2, s};
use ndarray_npz::NpzWriter;
use plotly::plotly_static;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use scalib::ttest;
use scasim::batch::{
    BatchDiagnostics, batch_traces, read_batch_meta, read_trace_cache, write_trace_cache,
};
use scasim::hierarchy::{HierarchyIndex, Selection};
use scasim::plot::*;
use scasim::power::hierarchy_index;
use std::fs::File;
use std::path::{Path, PathBuf};

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
    #[arg(short = 'd', default_value_t = 2)]
    order: usize,
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
    /// matches a rule if one of its names matches. Without any rule, all signals are selected.
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

/// The largest |t| in a row of t-values, or NaN if no value is finite. Values that are not
/// finite (NaN or infinite) are skipped.
fn max_abs_finite(row: impl IntoIterator<Item = f64>) -> f64 {
    row.into_iter()
        .filter(|x| x.is_finite())
        .map(f64::abs)
        .fold(f64::NAN, f64::max)
}

/// Reads the paths of the metadata files from a meta list file. Relative paths are relative to
/// the directory of the list file.
fn read_meta_list(list_path: &Path) -> miette::Result<Vec<PathBuf>> {
    let root = list_path.parent().unwrap_or(Path::new(""));
    let text = std::fs::read_to_string(list_path)
        .into_diagnostic()
        .wrap_err_with(|| format!("cannot read the meta list file {}", list_path.display()))?;
    Ok(text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(|line| {
            let path = PathBuf::from(line);
            if path.is_absolute() {
                path
            } else {
                root.join(path)
            }
        })
        .collect())
}

/// Reads the traces and labels of one batch from the cache or computes them from the waveform.
fn batch_data(
    metadata_path: &Path,
    use_existing: bool,
    cache_allowed: bool,
    selection: &Selection,
) -> miette::Result<(Array2<f32>, Array1<u16>)> {
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

    if use_existing
        && cache_allowed
        && npz_path.exists()
        && cache_is_fresh(&npz_path, &trace_file_path, metadata_path)
    {
        println!(
            "Using existing traces and labels from {}",
            npz_path.display()
        );
        return read_trace_cache(&npz_path);
    }

    println!(
        "Computing power traces from {}...",
        trace_file_path.display()
    );
    let start_time = std::time::Instant::now();
    let (traces_array, labels_array, diagnostics) = batch_traces(&meta, selection)?;
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

    if cache_allowed {
        println!("Saving traces and labels to NPZ file...");
        let start_time = std::time::Instant::now();
        write_trace_cache(&npz_path, &traces_array, &labels_array)?;
        println!(
            "Saved traces and labels to {} in {:.2}s\n",
            npz_path.display(),
            start_time.elapsed().as_secs_f32()
        );
    }
    Ok((traces_array, labels_array))
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
        read_meta_list(Path::new(meta_list_path))?
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
    let order = args.order;

    let mut samples_per_trace = 0;
    let mut max_t_values = vec![Vec::<f64>::new(); order];
    let mut num_traces_so_far = vec![];
    // Initial max |t| is 0.0 for each order corresponding to 0 traces
    max_t_values.iter_mut().for_each(|v| {
        v.push(0.0);
    });
    num_traces_so_far.push(0);

    let mut maybe_ttacc: Option<ttest::Ttest> = None;

    // `traces.npz` holds the traces of the default selection (all signals). A run with rules
    // computes other traces. It must not reuse the file, and it must not overwrite it, because
    // a later run without rules would then read the traces of this selection.
    let cache_allowed = rules.is_empty();

    if let Some(n) = args.num_threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .into_diagnostic()
            .wrap_err("cannot set the number of threads")?;
    }

    println!(
        "Using {} threads for parallel processing",
        rayon::current_num_threads()
    );

    let batches: Vec<(PathBuf, Array2<f32>, Array1<u16>)> = filenames
        .into_par_iter()
        .map(|metadata_path| {
            let (traces, labels) =
                batch_data(&metadata_path, args.use_existing, cache_allowed, &selection)?;
            Ok((metadata_path, traces, labels))
        })
        .collect::<miette::Result<_>>()?;

    let mut total_collected_traces: usize = 0;
    let mut last_t_values: Option<Array2<f64>> = None;
    // must be done sequentially
    for (metadata_path, traces_array, labels_array) in batches {
        let (num_traces, cur_samples_per_trace) = traces_array.dim();
        total_collected_traces += num_traces;
        if num_traces <= 1 {
            return Err(miette!(
                "the batch {} has {num_traces} traces; a t-test needs at least two traces",
                metadata_path.display()
            ));
        }
        if labels_array.len() != num_traces {
            return Err(miette!(
                "the batch {} has {num_traces} traces but {} labels",
                metadata_path.display(),
                labels_array.len()
            ));
        }
        let traces_array = if samples_per_trace == 0 {
            // Initialize samples_per_trace with the length of the first trace
            samples_per_trace = cur_samples_per_trace;
            traces_array
        } else if samples_per_trace == cur_samples_per_trace {
            traces_array
        } else {
            error!(
                "Inconsistent number of samples per trace: expected {}, found {}",
                samples_per_trace, cur_samples_per_trace
            );
            if cur_samples_per_trace > samples_per_trace {
                warn!(
                    "Using the first {} samples of the longer trace",
                    samples_per_trace
                );
                traces_array.slice(s![.., ..samples_per_trace]).to_owned()
            } else {
                error!(
                    "skipping trace with {} samples as expected {}",
                    cur_samples_per_trace, samples_per_trace
                );
                // create a larger array with zeros
                let mut t = Array2::<f32>::zeros((num_traces, samples_per_trace));
                // fill in each row with the available samples
                for (i, row) in traces_array.outer_iter().enumerate() {
                    t.slice_mut(s![i, ..row.len()]).assign(&row);
                }
                t
            }
        };
        num_traces_so_far.push(
            num_traces_so_far
                .last()
                .map_or(num_traces, |&last| last + num_traces),
        );

        let ttacc = maybe_ttacc.get_or_insert_with(|| ttest::Ttest::new(samples_per_trace, order));
        // Update the ttest accumulator with the current traces and labels
        ttacc.update(traces_array.view(), labels_array.view());

        let t_values = ttacc.get_ttest();
        for (max_t, t_row) in max_t_values.iter_mut().zip(t_values.rows()) {
            max_t.push(max_abs_finite(t_row.iter().copied()));
        }
        last_t_values = Some(t_values);
    }
    let t_values = last_t_values.ok_or_else(|| miette!("there is no batch to analyze"))?;
    if t_values.iter().any(|t| !t.is_finite()) {
        warn!(
            "some t-values are not finite (for example, a sample is constant or a class has too \
             few traces). The maximum |t| ignores them and is NaN if no t-value is finite"
        );
    }

    log::info!("Total number of traces: {}", total_collected_traces);

    let output_dir = PathBuf::from(&args.ttest_output_dir);
    if !output_dir.exists() {
        std::fs::create_dir_all(&output_dir)
            .into_diagnostic()
            .wrap_err_with(|| format!("cannot create the directory {}", output_dir.display()))?;
    }

    // Save t_values to a npz file
    let npz_path = output_dir.join("t_values.npz");
    info!("Saving t-test results to {}", npz_path.display());
    let mut npz = NpzWriter::new_compressed(
        File::create(&npz_path)
            .into_diagnostic()
            .wrap_err_with(|| format!("cannot create {}", npz_path.display()))?,
    );
    npz.add_array("t_values", &t_values)
        .into_diagnostic()
        .wrap_err("cannot write the t-values")?;
    npz.finish()
        .into_diagnostic()
        .wrap_err("cannot write the t-values")?;
    info!("Saved t_values to {}", npz_path.display());

    if args.plot {
        let mut image_exporter = plotly_static::StaticExporterBuilder::default()
            .pdf_export_timeout(1000)
            // .offline_mode(true)
            .build()
            .map_err(|e| miette!("cannot create the static plot exporter: {e}"))?;

        let plots_config = plotly::Configuration::new()
            .display_mode_bar(plotly::configuration::DisplayModeBar::Hover)
            .show_link(false)
            .display_logo(false)
            .editable(false)
            .responsive(true)
            .typeset_math(true);

        let t_threshold = Some(4.5);

        plot_t_traces(
            t_values,
            t_threshold,
            false, // abs_values
            &output_dir,
            args.show_plots,
            &plots_config,
            &mut image_exporter,
        )?;

        plot_max_t_values(
            max_t_values,
            num_traces_so_far,
            t_threshold,
            &output_dir,
            args.show_plots,
            &plots_config,
            &mut image_exporter,
        )?;
    }

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
    fn max_abs_finite_skips_values_that_are_not_finite() {
        assert_eq!(max_abs_finite([1.0, -3.0, 2.0]), 3.0);
        assert_eq!(max_abs_finite([f64::NAN, -2.0, f64::INFINITY]), 2.0);
        assert!(max_abs_finite([f64::NAN, f64::NEG_INFINITY]).is_nan());
        assert!(max_abs_finite([]).is_nan());
    }

    #[test]
    fn no_flags_give_no_rules() {
        assert!(rules(&[]).is_empty());
        assert!(rules(&["--list-signals"]).is_empty());
    }
}
