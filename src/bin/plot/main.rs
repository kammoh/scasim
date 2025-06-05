use clap::Parser;
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
#[command(name = "scasim-plot")]
#[command(author = "Kamyar Mohajerani <kammoh@gmail.com>")]
#[command(version = "0.1.0")]
#[command(about = "Plot power traces from NPZ file", long_about = None)]
struct Args {
    #[arg(value_name = "NPZ_FILE", index = 1)]
    filename: String,
    #[arg(index = 2)]
    trace_indices: Vec<usize>,
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

    for (i, index) in args.trace_indices.iter().enumerate() {
        let trace_name = format!("trace_{}", index);
        let trace_data: Array1<f32> = npz
            .by_name(&trace_name)
            .expect(&format!("Failed to find '{}' in NPZ file", trace_name));

        let label = labels
            .get(*index)
            .expect(&format!("Failed to get label for index {}", index));

        let scatter_trace = Scatter::from_array(
            Array1::<f32>::linspace(0.0, trace_data.len() as f32, trace_data.len()),
            trace_data,
        )
        .mode(Mode::Lines)
        .name(format!("Trace {} (Label: {})", index, label))
        .line(
            plotly::common::Line::new()
                .width(1.0)
                .auto_color_scale(true),
        );

        plot.add_trace(scatter_trace);
    }
    plot.set_layout(plotly::Layout::new().title(format!("Power Traces")));

    plot.set_configuration(
        plotly::Configuration::new()
            .show_link(false)
            .display_logo(false),
    );
    // Show the plot
    plot.show();
}
