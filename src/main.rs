use clap::Parser;

use plotly::common::Mode;
use plotly::{Plot, Scatter};
use scasim::wave_to_powertrace;

#[derive(Parser, Debug)]
#[command(name = "scasim-power")]
#[command(author = "Kamyar Mohajerani <kamyar@kamyar.xyz>")]
#[command(version)]
#[command(about = "Generate power trace from waveform", long_about = None)]
struct Args {
    #[arg(value_name = "WAVE_FILE", index = 1)]
    filename: String,
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
}

fn main() {
    let args = Args::parse();

    let (time_table, power_table) = wave_to_powertrace(
        &args.filename,
        !args.single_thread,
        args.show_progress,
        |_| true,
        false,
    )
    .expect("Failed to load and process the waveform");

    println!("Plotting {} time points", time_table.len());

    let trace1 = Scatter::new(time_table, power_table).mode(Mode::Lines);
    let mut plot = Plot::new();
    plot.add_trace(trace1);
    plot.set_layout(plotly::Layout::new().title("Power Trace"));

    plot.show();
}
