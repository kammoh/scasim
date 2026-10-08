use clap::Parser;

use plotly::common::Mode;
use plotly::{Plot, Scatter};
use scasim::hierarchy::Selection;
use scasim::power::power_trace;
use std::path::Path;

#[derive(Parser, Debug)]
#[command(name = "scasim-power")]
#[command(author = "Kamyar Mohajerani <kamyar@kamyar.xyz>")]
#[command(version)]
#[command(about = "Generate power trace from waveform", long_about = None)]
struct Args {
    #[arg(value_name = "WAVE_FILE", index = 1)]
    filename: String,
}

fn main() {
    let args = Args::parse();

    let (trace, _info) = power_trace(Path::new(&args.filename), &Selection::all())
        .expect("Failed to compute the power trace from the waveform");

    println!("Plotting {} time points", trace.times.len());

    let power: Vec<f32> = trace.power.iter().map(|&p| p as f32).collect();
    let trace1 = Scatter::new(trace.times, power).mode(Mode::Lines);
    let mut plot = Plot::new();
    plot.add_trace(trace1);
    plot.set_layout(plotly::Layout::new().title("Power Trace"));

    plot.show();
}
