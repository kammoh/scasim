use clap::{Parser, Subcommand};
use log::info;
use miette::{IntoDiagnostic, WrapErr, miette};
use plotly::common::Mode;
use plotly::{Plot, Scatter};
use scasim::batch::read_trace_cache;
use scasim::fold::{Fold, Loaded, create_output_dir, read_path_list, write_t_values_npz};
use scasim::plot::{plot_max_t_values, plot_t_traces};
use std::path::{Path, PathBuf};

/// The conventional TVLA threshold on |t|.
const T_THRESHOLD: f64 = 4.5;

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
        #[arg(value_name = "INDICES", index = 1, required = true)]
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
        #[arg(short = 'd', default_value_t = 2, value_parser = clap::value_parser!(u64).range(1..))]
        order: u64,

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
        .typeset_math(false);

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
                return Err(miette!(
                    "no trace indices provided. Please specify at least one trace index"
                ));
            }
            let (traces, labels) = read_trace_cache(Path::new(&filename))?;
            let mut plot = Plot::new();
            for &index in &trace_indices {
                if index >= traces.nrows() {
                    return Err(miette!(
                        "{filename} has {} traces, so there is no trace {index}",
                        traces.nrows()
                    ));
                }
                let scatter_trace = Scatter::new(
                    (0..traces.ncols()).collect::<Vec<_>>(),
                    traces.row(index).to_vec(),
                )
                .mode(Mode::Lines)
                .name(format!("Trace {} (Label: {})", index, labels[index]))
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
            maybe_npz_list_path,
        } => {
            let order = usize::try_from(order)
                .into_diagnostic()
                .wrap_err("the order is too large")?;
            let filenames: Vec<PathBuf> = if let Some(list_path) = &maybe_npz_list_path {
                read_path_list(Path::new(list_path))?
            } else if let Some(filenames) = maybe_filenames {
                filenames.into_iter().map(PathBuf::from).collect()
            } else {
                return Err(miette!(
                    "no NPZ files provided. Please specify --filenames or --npz-list"
                ));
            };
            if filenames.is_empty() {
                return Err(miette!("no NPZ files provided. The list has no files"));
            }

            // The chi-squared test is not part of this subcommand.
            let mut fold = Fold::new(order, false);
            for filename in &filenames {
                info!("Processing file: {}", filename.display());
                let (traces, labels) = read_trace_cache(filename)?;
                if traces.nrows() <= 1 {
                    return Err(miette!(
                        "the batch {} has {} traces; a t-test needs at least two traces",
                        filename.display(),
                        traces.nrows()
                    ));
                }
                fold.add(Loaded {
                    metadata: filename.clone(),
                    source: filename.clone(),
                    traces,
                    labels,
                })?;
            }
            let t_values = fold
                .t_values
                .take()
                .ok_or_else(|| miette!("there is no batch to analyze"))?;
            let total_num_traces = fold.num_traces.last().copied().unwrap_or(0);
            log::info!("Total number of traces: {}", total_num_traces);

            create_output_dir(output_dir)?;
            write_t_values_npz(output_dir, &t_values)?;

            plot_t_traces(
                t_values.view(),
                Some(T_THRESHOLD),
                false, // abs_values
                output_dir,
                args.show_plots,
            )?;
            plot_max_t_values(
                &fold.max_t,
                &fold.num_traces,
                Some(T_THRESHOLD),
                output_dir,
                args.show_plots,
            )?;
        }
    }
    Ok(())
}
