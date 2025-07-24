use clap::{Parser, Subcommand};
use itertools::Itertools;
use log::info;
use ndarray::{Array1, Array2};
use plotly::common::Mode;
use plotly::{Plot, Scatter};
use scalib::ttest;
use scasim::plot::{plot_max_t_values, plot_t_traces};
use std::fs::File;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[clap(version)]
struct Args {
    #[command(subcommand)]
    cmd: Commands,

    #[arg(
        value_name = "OUTPUT_DIR",
        help = "Directory to save the output files",
        default_value = ""
    )]
    output_dir: String,
    #[arg(
        long = "show",
        help = "Show the plots in a web browser",
        action = clap::ArgAction::SetTrue,
    )]
    show_plots: bool,
}

#[derive(Subcommand, Debug)]
enum Commands {
    #[clap(name = "plot-traces", about = "Plot traces from a NPZ file")]
    PlotTraces {
        /// Indices of the traces to plot
        #[arg(value_name = "INDICES", index = 1)]
        trace_indices: Vec<usize>,

        #[arg(value_name = "NPZ_FILE", index = 2)]
        filename: String,
    },
    #[clap(
        name = "ttest",
        about = "Perform t-test on traces from accumulated NPZ files"
    )]
    TTest {
        /// The highest order of t-test to perform
        #[arg(short = 'd', default_value_t = 2)]
        order: usize,

        #[arg(long ="filenames", value_name = "NPZ_FILE", num_args = 1..)]
        maybe_filenames: Option<Vec<String>>,
        #[arg(long = "npz-list", value_name = "NPZ_LIST_PATH")]
        maybe_npz_list_path: Option<String>,
    },
}

fn main() -> miette::Result<()> {
    let args = Args::parse();

    let output_dir = Path::new(&args.output_dir);

    let plots_config = plotly::Configuration::new()
        .display_mode_bar(plotly::configuration::DisplayModeBar::Hover)
        .show_link(false)
        .display_logo(false)
        .editable(false)
        .responsive(true)
        .typeset_math(true);

    // set default log level to info
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp(None)
        .init();

    match args.cmd {
        Commands::PlotTraces {
            trace_indices,
            filename,
        } => {
            if trace_indices.is_empty() {
                panic!("No trace indices provided. Please specify at least one trace index.");
            }

            // Load the NPZ file using a bufferred reader
            let file = File::open(&filename).expect("Failed to open NPZ file");
            let reader = std::io::BufReader::new(file);

            // Parse the NPZ file
            let mut npz = ndarray_npz::NpzReader::new(reader).expect("Failed to parse NPZ file");
            let labels: Array1<u16> = npz
                .by_name("labels")
                .expect("Failed to find 'labels' in NPZ file");
            // Create a plot and add the trace
            let mut plot = Plot::new();
            for index in &trace_indices {
                let trace_name = format!("trace_{}", index);
                let trace_data: Array1<f32> = npz
                    .by_name(&trace_name)
                    .expect(&format!("Failed to find '{}' in NPZ file", trace_name));

                let label = labels
                    .get(*index)
                    .expect(&format!("Failed to get label for index {}", index));

                let scatter_trace =
                    Scatter::from_array(Array1::range(0., trace_data.len() as f32, 1.), trace_data)
                        .mode(Mode::Lines)
                        .name(format!("Trace {} (Label: {})", index, label))
                        .line(
                            plotly::common::Line::new()
                                .width(1.0)
                                .auto_color_scale(true),
                        );

                plot.add_trace(scatter_trace);
            }
            plot.set_layout(plotly::Layout::new().title("Power Traces"));

            plot.set_configuration(plots_config.clone());
            if args.show_plots {
                plot.show();
            }
        }
        Commands::TTest {
            order,
            maybe_filenames,
            maybe_npz_list_path: maybe_meta_list_path,
        } => {
            assert!(order > 0, "Order must be greater than 0");

            let filenames: Vec<PathBuf> = if let Some(meta_list_path) = maybe_meta_list_path {
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
            } else if let Some(filenames) = maybe_filenames {
                filenames.into_iter().map(PathBuf::from).collect_vec()
            } else {
                panic!("No NPZ files provided. Please specify at least one NPZ file.");
            };

            let mut samples_per_trace = 0;
            let mut max_t_values = vec![Vec::<f64>::new(); order];
            let mut num_traces_so_far = vec![];

            let mut maybe_ttacc: Option<ttest::Ttest> = None;

            if filenames.is_empty() {
                panic!("No NPZ files provided. Please specify at least one NPZ file.");
            }

            let t_values = filenames
                .iter()
                .fold(None, |_, filename| {
                    // Load the NPZ file using a bufferred reader
                    let file = File::open(filename).expect("Failed to open NPZ file");
                    let reader = std::io::BufReader::new(file);

                    // Parse the NPZ file
                    info!("Processing file: {}", filename.display());
                    let mut npz_reader =
                        ndarray_npz::NpzReader::new(reader).expect("Failed to parse NPZ file");
                    let labels: Array1<u16> = npz_reader
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
                        .collect();
                    let num_traces = traces.len();

                    let traces_array: Array2<f32> = Array2::from_shape_vec(
                        (num_traces, traces[0].len()),
                        traces.into_iter().flatten().collect(),
                    )
                    .expect("Failed to create traces array");

                    num_traces_so_far.push(
                        num_traces_so_far
                            .last()
                            .map_or(num_traces, |&last| last + num_traces),
                    );

                    if samples_per_trace == 0 {
                        // Initialize samples_per_trace with the length of the first trace
                        samples_per_trace = traces_array.shape()[1];
                    } else if samples_per_trace != traces_array.shape()[1] {
                        panic!(
                            "Inconsistent number of samples per trace: expected {}, found {}",
                            samples_per_trace,
                            traces_array.shape()[1]
                        );
                    }

                    if maybe_ttacc.is_none() {
                        maybe_ttacc = Some(ttest::Ttest::new(samples_per_trace, order));
                    }

                    if let Some(ref mut ttacc) = maybe_ttacc {
                        ttacc.update(traces_array.view(), labels.view());

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

            let t_threshold = Some(4.5);

            plot_t_traces(
                t_values,
                t_threshold,
                false, // abs_values
                output_dir,
                args.show_plots,
                &plots_config,
            )?;

            assert!(max_t_values.len() == order);
            assert!(num_traces_so_far.len() == max_t_values[0].len());

            plot_max_t_values(
                max_t_values,
                num_traces_so_far,
                t_threshold,
                output_dir,
                args.show_plots,
                &plots_config,
            )?;
        }
    }
    Ok(())
}
