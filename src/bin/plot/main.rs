use clap::{Parser, Subcommand};
use itertools::Itertools;
use ndarray::{Array1, Array2, s};
use ndarray_npz::NpzWriter;
use plotly::common::Mode;
use plotly::{Plot, Scatter, Trace};
use scalib::ttest;
use scasim::*;
use std::fs::File;
use std::io::Read;
use std::path::Path;

#[derive(Parser, Debug)]
#[clap(version)]
struct Args {
    #[command(subcommand)]
    cmd: Commands,

    #[arg(value_name = "NPZ_FILE", index = 1)]
    filename: String,

    #[arg(value_name = "OUTPUT_DIR", help = "Directory to save the output files")]
    output_dir: String,
}

#[derive(Subcommand, Debug)]
enum Commands {
    #[clap(name = "plot-traces", about = "Plot traces from a NPZ file")]
    PlotTraces {
        /// Optionally set a title for what you are going to write about
        #[arg(index = 1)]
        trace_indices: Vec<usize>,
    },
    TTest {
        /// Optionally set a title for what you are going to write about
        #[arg(default_value_t = 2)]
        order: usize,
    },
}

fn main() {
    let args = Args::parse();

    // Load the NPZ file using a bufferred reader
    let file = File::open(&args.filename).expect("Failed to open NPZ file");
    let reader = std::io::BufReader::new(file);

    // Create a plot and add the trace
    let mut plot = Plot::new();

    // Parse the NPZ file
    let mut npz = ndarray_npz::NpzReader::new(reader).expect("Failed to parse NPZ file");
    let labels: Array1<u16> = npz
        .by_name("labels")
        .expect("Failed to find 'labels' in NPZ file");

    let output_dir = Path::new(&args.output_dir);

    let plots_config = plotly::Configuration::new()
        .display_mode_bar(plotly::configuration::DisplayModeBar::Hover)
        .show_link(false)
        .display_logo(false)
        .editable(false)
        .responsive(true);

    match args.cmd {
        Commands::PlotTraces { trace_indices } => {
            if trace_indices.is_empty() {
                panic!("No trace indices provided. Please specify at least one trace index.");
            }

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
            // Show the plot
            plot.show();
        }
        Commands::TTest { order } => {
            let num_traces = npz.len() - 1; // Exclude labels
            let traces: Vec<Array1<f32>> = (0..num_traces)
                .map(|i| {
                    let trace_name = format!("trace_{}", i);
                    npz.by_name(trace_name.as_str())
                        .expect(&format!("Failed to find '{}' in NPZ file", trace_name))
                })
                .collect();

            let traces_array: Array2<f32> = Array2::from_shape_vec(
                (num_traces, traces[0].len()),
                traces.into_iter().flatten().collect(),
            )
            .expect("Failed to create traces array");

            let samples_per_trace = traces_array.shape()[1];

            let mut ttacc = ttest::Ttest::new(samples_per_trace, order);

            ttacc.update(traces_array.view(), labels.view());

            let t_values = ttacc.get_ttest();

            let t1_values = t_values
                .row(0)
                .iter()
                .filter_map(|&x| x.is_finite().then_some(x.abs()))
                .collect_vec();

            // print!(
            //     "First 100 t-values: {:?}\n",
            //     t_values.row(0).slice(s![..100]).iter().join(", ")
            // );

            let t1_values_len = t1_values.len();
            println!("Plotting t-values...");
            //plot the t-values
            let t_trace =
                Scatter::new((0..t1_values.len()).collect_vec(), t1_values).mode(Mode::Lines);
            let mut t_plot = Plot::new();
            t_plot.add_trace(t_trace);
            t_plot.set_layout(plotly::Layout::new().title("T-test Values").shapes(
                vec![plotly::layout::Shape::new()
            .shape_type(plotly::layout::ShapeType::Line)
            .x0(0)
            .x1(t1_values_len as f64 - 1.0)
            .y0(4.5)
            .y1(4.5)
            .line(plotly::layout::ShapeLine::new()
                    .color(plotly::color::NamedColor::Orange)
                    .width(2.5)
                    .dash(plotly::common::DashType::Dot))
            ],
            ));
            t_plot.set_configuration(plots_config);
            t_plot.write_html(output_dir.join("t_test_plot.html"));
            // t_plot.write_image(
            //     output_dir.join("t_test_plot.png"),
            //     plotly::ImageFormat::PNG,
            //     1920,
            //     400,
            //     1.0,
            // );
            let t_plot_json_path = output_dir.join("t_plot.json");
            std::fs::write(t_plot_json_path, t_plot.to_json())
                .expect("Failed to write t_plot to JSON file");
            t_plot.show();
        }
    }
}
