use clap::Parser;
use itertools::Itertools;
use log::info;
use miette::{Diagnostic, SourceSpan};
use ndarray::{Array1, Array2, ArrayBase, Dimension, OwnedRepr, s};
use ndarray_npz::NpzWriter;
use num_ordinal::{Ordinal, Osize};
use plotly::common::{Mode, Title};
use plotly::{Plot, Scatter, Trace};
use scalib::ttest;
use scasim::plot::*;
use scasim::*;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Parser, Debug)]
#[command(name = "scasim-tvla")]
#[command(author = "Kamyar Mohajerani <kamyar@kamyar.xyz>")]
#[command(version)]
#[command(about = "Test-Vector Leakage Analysis", long_about = None)]
struct Args {
    // #[arg(value_name = "WAVE_FILE", index = 1)]
    // trace_filename: String,
    #[arg(value_name = "META_JSON", index = 1)]
    maybe_metadata: Option<String>,
    #[arg(long = "meta-list", value_name = "META_LIST_PATH")]
    maybe_meta_list_path: Option<String>,
    // #[arg(value_name = "OUTPUT_DIR", help = "Directory to save the output files")]
    // output_dir: Option<String>,
    #[arg(
        long,
        help = "disable multi-threaded loading of the waveform and signals",
        default_value_t = false
    )]
    single_thread: bool,
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
        help = "Plot the t-test results",
        required = false,
        default_value_t = true
    )]
    plot: bool,
    #[arg(
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

fn markers_to_time_indices(
    meta_markers: &[Vec<u64>],
    time_table: &[u64],
) -> Vec<(usize, usize, u16)> {
    // convert the markers to time index intervals
    // each marker has 3 u64 elements: start_time, end_time, and label
    // returns a vector of tuples of (low_index, high_index, label) where low_index..high_index is the range of time indices for the marker
    // use the fact that both meta_markers and time_table are sorted by time

    meta_markers
        .iter()
        .map(|marker| {
            let start_time = marker[0];
            let end_time = marker[1];
            let label = marker[2] as u16;

            // find the start index
            let low_index = time_table.binary_search(&start_time).unwrap_or_else(|x| x);
            // find the end index
            let high_index = time_table.binary_search(&end_time).unwrap_or_else(|x| x);

            (low_index, high_index, label)
        })
        .collect()
}

fn cut_trace(
    power_table: &[f32],
    time_table: &[u64],
    meta_markers: &[Vec<u64>],
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
            .map(|line| {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    panic!("Empty line in meta list file");
                }
                let mut p = PathBuf::from(trimmed);
                if !p.is_absolute() {
                    p = meta_root_path.join(p);
                }
                p
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

    let mut maybe_ttacc: Option<ttest::Ttest> = None;

    if filenames.is_empty() {
        panic!("No meta files provided. Please specify at least one NPZ file.");
    }

    let t_values = filenames
        .iter()
        .fold(None, |_, metadata_path| {
            if !metadata_path.exists() {
                panic!(
                    "Metadata file '{}' does not exist!",
                    metadata_path.display()
                );
            }

            let metadata_json = get_metadata(
                metadata_path,
                metadata_path.extension().map_or(false, |ext| ext == "gz"),
            )
            .expect("Failed to load metadata!");

            let parent_folder_path = metadata_path
                .parent()
                .expect("Failed to get parent folder of metadata file")
                .to_path_buf();

            let output_dir = &parent_folder_path;

            let trace_filename = metadata_json
                .get("trace_filename")
                .and_then(|v| v.as_str())
                .expect("trace_filename not found in metadata");

            let trace_file_path = parent_folder_path.join(trace_filename);

            let clock_period = metadata_json.get("clock_period").and_then(|v| v.as_u64());
            let cp = clock_period.unwrap_or_default();
            // .expect("clock_period not found in the metadata"); // FIXME optional

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

            // if false {
            //     println!("Plotting {} time points", time_table.len());

            //     let trace1 =
            //         Scatter::new(time_table.clone(), power_table.clone()).mode(Mode::Lines);
            //     let mut plot = Plot::new();
            //     plot.add_trace(trace1);
            //     plot.set_layout(plotly::Layout::new().title("Power Trace"));
            //     plot.set_configuration(plots_config.clone());

            //     plot.show();
            // }

            let meta_markers = metadata_json
                .get("markers")
                .map(|v| {
                    v.as_array()
                        .unwrap()
                        .into_iter()
                        .map(|e| {
                            e.as_array()
                                .unwrap()
                                .into_iter()
                                .map(|i| i.as_u64().unwrap())
                                .collect_vec()
                        })
                        .collect_vec()
                })
                .expect("markers not found in metadata");

            println!("Cutting traces based on markers...");
            let start_time = std::time::Instant::now();
            let (traces_array, labels_array) = cut_trace(&power_table, &time_table, &meta_markers);
            let (num_traces, cur_samples_per_trace) = traces_array.dim();
            if samples_per_trace == 0 {
                // Initialize samples_per_trace with the length of the first trace
                samples_per_trace = cur_samples_per_trace;
            } else if samples_per_trace != cur_samples_per_trace {
                panic!(
                    "Inconsistent number of samples per trace: expected {}, found {}",
                    samples_per_trace, cur_samples_per_trace
                );
            }
            num_traces_so_far.push(
                num_traces_so_far
                    .last()
                    .map_or(num_traces, |&last| last + num_traces),
            );
            println!(
                "Cut traces in {:.2}s, resulting in {} traces with a maximum of {} samples each",
                start_time.elapsed().as_secs_f32(),
                num_traces,
                samples_per_trace
            );

            assert!(num_traces > 1, "Number of traces must be greater than 1");
            assert!(
                labels_array.len() == num_traces,
                "Number of trace labels does not match number of traces"
            );

            println!("Saving traces and labels to NPZ file...");
            let start_time: std::time::Instant = std::time::Instant::now();
            let npz_path = output_dir.join("traces.npz");

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

            if maybe_ttacc.is_none() {
                maybe_ttacc = Some(ttest::Ttest::new(samples_per_trace, order));
            }

            if let Some(ref mut ttacc) = maybe_ttacc {
                // Update the ttest accumulator with the current traces and labels
                ttacc.update(traces_array.view(), labels_array.view());

                // let is_last_file = file_index == num_files - 1;

                // if !is_last_file {
                let t_values = ttacc.get_ttest();
                max_t_values
                    .iter_mut()
                    .zip(t_values.rows())
                    .for_each(|(max_t, t_row)| {
                        // for (i, &t_value) in t_row.iter().enumerate() {
                        //     if t_value.is_finite() && t_value.abs() > max_t[i] {
                        //         max_t[i] = t_value.abs();
                        //     }
                        // }
                        max_t.push(
                            t_row
                                .iter()
                                .filter_map(|&x| x.is_finite().then_some(x.abs()))
                                .max_by(|a, b| a.partial_cmp(b).unwrap())
                                .expect("Failed to find max t-value in current row"),
                        );
                    });
                Some(t_values)
                // }
            } else {
                panic!("Ttest accumulator is not initialized");
            }
        })
        .expect("Failed to compute t-test values");

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
            &output_dir,
            args.show_plots,
            &plots_config,
        )?;

        // Plot the max t-values
        plot_max_t_values(
            max_t_values,
            num_traces_so_far,
            t_threshold,
            &output_dir,
            args.show_plots,
            &plots_config,
        )?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {}
