use clap::Parser;
use itertools::Itertools;
use log::*;
use ndarray::{Array1, Array2, s};
use ndarray_npz::{NpzReader, NpzWriter};
use plotly::plotly_static;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use scalib::ttest;
use scasim::plot::*;
use scasim::*;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

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
        help = "disable multi-threaded loading of the waveform and signals",
        default_value_t = false
    )]
    single_thread: bool,
    #[arg(
        long,
        help = "number of threads to use for parallel processing, defaults to the number of available CPU cores",
        value_name = "NUM_THREADS"
    )]
    num_threads: Option<usize>,
    #[arg(
        long,
        help = "show progress bar while loading the file",
        default_value_t = true
    )]
    show_progress: bool,
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
        default_value_t = true
    )]
    plot: bool,
    #[arg(
        long = "use-existing",
        help = "Skip generation of power trace data if the NPZ file already exists and is not older than the corresponding trace file. Use their stored data instead.",
        default_value_t = true
    )]
    use_existing: bool,
    #[arg(
        long,
        value_name = "PLOTS_OUTPUT_DIR",
        help = "Directory to save the ttest results and plot files",
        default_value = ""
    )]
    ttest_output_dir: String,
}

fn get_metadata<P: AsRef<Path>>(
    filename: P,
    is_compressed: bool,
) -> Result<serde_json::Value, std::io::Error> {
    let file = File::open(&filename)?;
    if is_compressed {
        let mut decoder = flate2::read::GzDecoder::new(file);
        let mut buffer = Vec::new();
        decoder.read_to_end(&mut buffer)?;
        Ok(serde_json::from_slice(&buffer)?)
    } else {
        Ok(serde_json::from_reader(file)?)
    }
}

fn cut_trace(
    power_table: &[f32],
    time_table: &[u64],
    meta_markers: &[(u64, u64, u16)],
) -> (Array2<f32>, Array1<u16>) {
    println!("Converting markers to time indices...");
    let start_time = std::time::Instant::now();
    let time_indices_and_labels = markers_to_time_indices(meta_markers, time_table);
    println!(
        "Converted markers to time indices in {:.2}s",
        start_time.elapsed().as_secs_f32()
    );

    let max_len = time_indices_and_labels
        .iter()
        .map(|(lo, hi, _)| hi - lo)
        .max()
        .unwrap_or(0);

    println!("Cutting the monolithic trace...");

    let mut all_traces = Array2::<f32>::zeros((time_indices_and_labels.len(), max_len));
    let mut trace_labels = Array1::<u16>::zeros(time_indices_and_labels.len());

    for (i, (start_idx, end_idx, label)) in time_indices_and_labels.into_iter().enumerate() {
        trace_labels[i] = label;
        all_traces
            .slice_mut(s![i, ..end_idx - start_idx])
            .assign(&Array1::from_vec(power_table[start_idx..end_idx].to_vec()));
    }

    (all_traces, trace_labels)
}

fn main() -> miette::Result<()> {
    let args = Args::parse();

    // set default log level to info
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp(None)
        .init();

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

    if filenames.is_empty() {
        panic!("No meta files provided. Please specify at least one NPZ file.");
    }

    let npz_filename = "traces.npz";

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

        let metadata_json = get_metadata(
            &metadata_path,
            metadata_path.extension().map_or(false, |ext| ext == "gz"),
        )
        .expect("Failed to load metadata!");

        let parent_folder_path = metadata_path
            .parent()
            .expect("Failed to get parent folder of metadata file")
            .to_path_buf();

        let trace_filename = metadata_json
            .get("trace_filename")
            .and_then(|v| v.as_str())
            .expect("trace_filename not found in metadata");

        let trace_file_path = parent_folder_path.join(trace_filename);

        let npz_path = parent_folder_path.join(npz_filename);

        let use_existing = if args.use_existing && npz_path.exists() {
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
                .filter_map(|name| {
                    name.starts_with("trace_").then(|| {
                        npz_reader
                            .by_name(name.as_str())
                            .expect(&format!("Failed to find '{}' in NPZ file", name))
                    })
                })
                .collect_vec();
            let num_traces = traces.len();
            let traces_array: Array2<f32> = Array2::from_shape_vec(
                (num_traces, traces[0].len()),
                traces.into_iter().flatten().collect(),
            )
            .expect("Failed to create traces array");
            Some((traces_array, labels_array))
        } else {
            let clock_period = metadata_json.get("clock_period").and_then(|v| v.as_u64());
            let cp = clock_period.unwrap_or_default();
            // .expect("clock_period not found in the metadata"); // FIXME optional
            let meta_markers = metadata_json
                .get("markers")
                .map(|v| {
                    v.as_array()
                        .unwrap()
                        .into_iter()
                        .map(|e| {
                            let (start_time, end_time, label) =  e.as_array()
                                .unwrap()
                                .into_iter()
                                .map(|i| i.as_u64().unwrap())
                                .collect_tuple().unwrap();
                            (start_time, end_time, label as u16)
                        })
                        .collect_vec()
                })
                .expect("markers not found in metadata");

            if false {
                let (traces_array, labels_array, _) = traces_from_fst(
                    &trace_file_path,
                    &meta_markers,
                    |t| clock_period.map(|cp| t % cp == 0).unwrap_or(true),
                ).expect("Failed to load traces from FST file");
                Some((traces_array, labels_array))
            } else {
            println!("Loading signals from the waveform...");
            let start_time = std::time::Instant::now();
            let (signals, time_table) =
                load_waveform(&trace_file_path, !args.single_thread, args.show_progress)
                    .expect("Failed to load waveform!");
            println!(
                "It took {:.2}s to load {} signals with {} time points",
                start_time.elapsed().as_secs_f32(),
                signals.len(),
                time_table.len()
            );

            println!("Generating power trace...");
            let start_time = std::time::Instant::now();
            let (time_table, power_table) = generate_power_trace(
                &signals,
                &time_table,
                |(t, _)| *t % cp == 0,
                clock_period.is_some(),
            )
            .expect("Failed to convert waveform to power trace!");
            println!(
                "It took {:.2}s to generate the power trace",
                start_time.elapsed().as_secs_f32()
            );

            

            println!("Cutting traces based on markers...");
            let start_time = std::time::Instant::now();
            let (traces_array, labels_array) = cut_trace(&power_table, &time_table, &meta_markers);

            let (num_traces, cur_samples_per_trace) = traces_array.dim();
            println!(
                "Cut traces in {:.2}s, resulting in {} traces with a maximum of {} samples each",
                start_time.elapsed().as_secs_f32(),
                num_traces,
                cur_samples_per_trace
            );
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

            Some((traces_array, labels_array))
        }
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
            }};
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

    log::info!(
        "Total number of traces: {}",
        total_collected_traces
    );

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
mod tests {}
