use clap::{ArgMatches, CommandFactory, FromArgMatches, Parser};
use itertools::Itertools;
use log::*;
use miette::{IntoDiagnostic, WrapErr};
use ndarray::{Array1, Array2, s};
use ndarray_npz::{NpzReader, NpzWriter};
use plotly::plotly_static;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use scalib::ttest;
use scasim::batch::{BatchDiagnostics, batch_traces, read_batch_meta};
use scasim::hierarchy::{HierarchyIndex, Selection};
use scasim::plot::*;
use scasim::power::hierarchy_index;
use std::fs::File;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "scasim-tvla")]
#[command(author = "Kamyar Mohajerani <kamyar@kamyar.xyz>")]
#[command(version)]
#[command(about = "Test-Vector Leakage Analysis", long_about = None)]
struct Args {
    #[arg(long = "meta-json", value_name = "META_JSON")]
    maybe_metadata: Option<String>,
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
        help = "Skip generation of power trace data if the NPZ file already exists and is not older than the corresponding trace file. Use their stored data instead.",
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

    let filenames: Vec<PathBuf> = if let Some(meta_list_path) = args.maybe_meta_list_path {
        let meta_root_path = PathBuf::from(&meta_list_path)
            .parent()
            .unwrap_or_else(|| {
                panic!(
                    "Meta list path '{}' does not have a parent directory",
                    meta_list_path
                )
            })
            .to_owned();
        // Read the meta list file and collect filenames
        std::fs::read_to_string(meta_list_path)
            .expect("Failed to read meta list file")
            .lines()
            .filter_map(|line| {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    return None; // Skip empty lines
                }
                let mut p = PathBuf::from(trimmed);
                if !p.is_absolute() {
                    p = meta_root_path.join(p);
                }
                Some(p)
            })
            .collect_vec()
    } else if let Some(filename) = args.maybe_metadata {
        vec![PathBuf::from(filename)]
    } else {
        panic!("No meta files provided. Please specify at least one NPZ file.");
    };
    if filenames.is_empty() {
        panic!("No meta files provided. Please specify at least one NPZ file.");
    }
    if args.list_signals {
        let meta = read_batch_meta(&filenames[0])?;
        let index = hierarchy_index(&meta.trace_path).into_diagnostic()?;
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

    let npz_filename = "traces.npz";
    // `traces.npz` holds the traces of the default selection (all signals). A run with rules
    // computes other traces. It must not reuse the file, and it must not overwrite it, because
    // a later run without rules would then read the traces of this selection.
    let cache_allowed = rules.is_empty();

    args.num_threads.iter().for_each(|&n| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .unwrap()
    });

    let default_num_threads = rayon::current_num_threads();

    println!(
        "Using {} threads for parallel processing",
        default_num_threads
    );

    let collected_traces = filenames.into_par_iter().filter_map(|metadata_path| {
        if !metadata_path.exists() {
            log::error!(
                "Metadata file '{}' does not exist!",
                metadata_path.display()
            );
            return None;
        }

        let meta = read_batch_meta(&metadata_path).expect("Failed to read batch metadata");
        let trace_file_path = meta.trace_path.clone();

        let parent_folder_path = metadata_path
            .parent()
            .expect("Failed to get parent folder of metadata file")
            .to_path_buf();

        let npz_path = parent_folder_path.join(npz_filename);

        let use_existing = if args.use_existing && cache_allowed && npz_path.exists() {
            if !trace_file_path.exists() {
                true
            } else {
                // Check if the npz file is older than the trace file
                let npz_modified = std::fs::metadata(&npz_path).and_then(|m| m.modified());
                let trace_modified = std::fs::metadata(&trace_file_path).and_then(|m| m.modified());
                if let (Ok(npz_modified), Ok(trace_modified)) = (npz_modified, trace_modified) {
                    // Use existing if npz file is newer than trace file
                    npz_modified > trace_modified
                } else {
                    false
                }
            }
        } else {
            false
        };

        if use_existing {
            println!(
                "Using existing traces and labels from {}",
                npz_path.display()
            );
            let mut npz_reader =
                NpzReader::new(File::open(&npz_path).expect("Failed to open npz file"))
                    .expect("Failed to read npz file");
            let labels_array: Array1<u16> = npz_reader
                .by_name("labels")
                .expect("Failed to find 'labels' in NPZ file");

            let traces: Vec<Array1<f32>> = npz_reader
                .names()
                .expect("Failed to get names from NPZ file")
                .iter()
                .filter(|&name| name.starts_with("trace_")).map(|name| npz_reader
                            .by_name(name.as_str())
                            .unwrap_or_else(|_| panic!("Failed to find '{}' in NPZ file", name)))
                .collect_vec();
            let num_traces = traces.len();
            let traces_array: Array2<f32> = Array2::from_shape_vec(
                (num_traces, traces[0].len()),
                traces.into_iter().flatten().collect(),
            )
            .expect("Failed to create traces array");
            Some((traces_array, labels_array))
        } else {
            println!("Computing power traces from {}...", trace_file_path.display());
            let start_time = std::time::Instant::now();
            let (traces_array, labels_array, diagnostics) =
                batch_traces(&meta, &selection).expect("Failed to compute traces");
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
                let start_time: std::time::Instant = std::time::Instant::now();
                let mut npz = NpzWriter::new_compressed(
                    File::create(&npz_path).expect("Failed to create npz file"),
                );
                for (tidx, trace) in traces_array.outer_iter().enumerate() {
                    npz.add_array(format!("trace_{tidx}"), &trace)
                        .expect("Failed to add array 'a' to npz");
                }
                npz.add_array("labels", &labels_array)
                    .expect("Failed to add array 'labels' to npz");
                npz.finish().expect("Failed to finish writing npz file");
                println!(
                    "Saved traces and labels to {} in {:.2}s\n",
                    npz_path.display(),
                    start_time.elapsed().as_secs_f32()
                );
            }

            Some((traces_array, labels_array))
        }
    }).collect_vec_list();

    let mut total_collected_traces: usize = 0;
    // must be done sequentially
    let t_values = collected_traces
        .into_iter()
        .flatten()
        .fold(None, |_prev_tvalues, (traces_array, labels_array)| {
            let (num_traces, cur_samples_per_trace) = traces_array.dim();
            total_collected_traces += num_traces;
            let traces_array = if samples_per_trace == 0 {
                // Initialize samples_per_trace with the length of the first trace
                samples_per_trace = cur_samples_per_trace;
                traces_array
            } else {
                if samples_per_trace == cur_samples_per_trace {
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
                        // Array2::<f32>::from(traces_array.slice(s![.., ..samples_per_trace]))
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
                }
            };
            num_traces_so_far.push(
                num_traces_so_far
                    .last()
                    .map_or(num_traces, |&last| last + num_traces),
            );

            assert!(num_traces > 1, "Number of traces must be greater than 1");
            assert!(
                labels_array.len() == num_traces,
                "Number of trace labels does not match number of traces"
            );

            if maybe_ttacc.is_none() {
                maybe_ttacc = Some(ttest::Ttest::new(samples_per_trace, order));
            }

            if let Some(ref mut ttacc) = maybe_ttacc {
                // Update the ttest accumulator with the current traces and labels
                ttacc.update(traces_array.view(), labels_array.view());

                let t_values = ttacc.get_ttest();
                max_t_values
                    .iter_mut()
                    .zip(t_values.rows())
                    .for_each(|(max_t, t_row)| {
                        max_t.push(
                            t_row
                                .iter()
                                .filter_map(|&x| x.is_finite().then_some(x.abs()))
                                .max_by(|a, b| a.partial_cmp(b).unwrap())
                                .expect("Failed to find max t-value in current row"),
                        );
                    });
                Some(t_values)
            } else {
                panic!("Ttest accumulator is not initialized");
            }
        })
        .expect("Failed to compute t-test values");

    log::info!("Total number of traces: {}", total_collected_traces);

    let output_dir = PathBuf::from(&args.ttest_output_dir);
    if !output_dir.exists() {
        std::fs::create_dir_all(&output_dir).expect("Failed to create output directory for plots");
    }

    // sage t_values to a npz file
    let npz_path = output_dir.join("t_values.npz");
    info!("Saving t-test results to {}", npz_path.display());
    let mut npz =
        NpzWriter::new_compressed(File::create(&npz_path).expect("Failed to create npz file"));
    npz.add_array("t_values", &t_values)
        .expect("Failed to add t_values array to npz");
    npz.finish().expect("Failed to finish writing npz file");
    info!("Saved t_values to {}", npz_path.display());

    if args.plot {
        let mut image_exporter = plotly_static::StaticExporterBuilder::default()
            .pdf_export_timeout(1000)
            // .offline_mode(true)
            .build()
            .expect("Failed to create static exporter");

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
    fn no_flags_give_no_rules() {
        assert!(rules(&[]).is_empty());
        assert!(rules(&["--list-signals"]).is_empty());
    }
}
