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
use miette::{Diagnostic, SourceSpan};
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
    metadata: String,
    #[arg(value_name = "OUTPUT_DIR", help = "Directory to save the output files")]
    output_dir: Option<String>,
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

    let metadata_path = Path::new(&args.metadata);

    let metadata_json = get_metadata(metadata_path, args.metadata.ends_with(".gz"))
        .expect("Failed to load metadata!");

    let parent_folder_path = metadata_path
        .parent()
        .expect("Failed to get parent folder of metadata file")
        .to_path_buf();

    let output_dir = args
        .output_dir
        .map(|s| Path::new(&s).to_path_buf())
        .unwrap_or(parent_folder_path.clone());

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

    let plots_config = plotly::Configuration::new()
        .display_mode_bar(plotly::configuration::DisplayModeBar::Hover)
        .show_link(false)
        .display_logo(false)
        .editable(false)
        .responsive(true);

    if false {
        println!("Plotting {} time points", time_table.len());

        let trace1 = Scatter::new(time_table.clone(), power_table.clone()).mode(Mode::Lines);
        let mut plot = Plot::new();
        plot.add_trace(trace1);
        plot.set_layout(plotly::Layout::new().title("Power Trace"));
        plot.set_configuration(plots_config.clone());

        plot.show();
    }

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
    let (num_traces, samples_per_trace) = traces_array.dim();
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

    let mut npz = NpzWriter::new(File::create(&npz_path).expect("Failed to create npz file"));
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

    println!(
        "Plotting {} time points for a single trace",
        time_table.len()
    );

    let sample_trace_index = 0;
    let trace1 = Scatter::from_array(
        Array1::linspace(0., samples_per_trace as f64, samples_per_trace),
        traces_array.row(sample_trace_index).to_owned(),
    )
    .mode(Mode::Lines)
    .line(plotly::common::Line::new().width(1.0).color("navy"));

    // save trace1 as json
    let trace_json_path = output_dir.join(format!("trace{sample_trace_index}.json"));
    std::fs::write(trace_json_path, trace1.to_json()).expect("Failed to write trace1 to JSON file");

    let mut plot = Plot::new();
    plot.add_trace(trace1);
    plot.set_layout(plotly::Layout::new().title(format!(
        "Power trace #{sample_trace_index} class={}",
        labels_array[sample_trace_index]
    )));
    plot.set_configuration(plots_config.clone());
    plot.write_html(output_dir.join(format!("trace{sample_trace_index}_plot.html")));
    plot.write_image(
        output_dir.join(format!("trace{sample_trace_index}_plot.png")),
        plotly::ImageFormat::PNG,
        1920,
        1080 / 4,
        1.0,
    );

    plot.show();

    println!("Running t-test on traces...");
    let start_time = std::time::Instant::now();
    let ttest_order = 1; // order of the t-test

    let mut ttacc = ttest::Ttest::new(samples_per_trace, ttest_order);

    ttacc.update(traces_array.view(), labels_array.view());

    let t_values = ttacc.get_ttest();

    let t1_values = t_values
        .row(0)
        .iter()
        .filter_map(|&x| x.is_finite().then_some(x.abs()))
        .collect_vec();

    println!(
        "T-test completed in {:.2}s.\nt-values shape: {:?}\n",
        start_time.elapsed().as_secs_f32(),
        t_values.shape(),
    );

    // print!(
    //     "First 100 t-values: {:?}\n",
    //     t_values.row(0).slice(s![..100]).iter().join(", ")
    // );

    let t1_values_len = t1_values.len();
    println!("Plotting t-values...");
    //plot the t-values
    let t_trace = Scatter::new((0..t1_values.len()).collect_vec(), t1_values).mode(Mode::Lines);
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
    t_plot.write_image(
        output_dir.join("t_test_plot.png"),
        plotly::ImageFormat::PNG,
        1920,
        400,
        1.0,
    );
    t_plot.write_image(
        output_dir.join("t_test_plot.svg"),
        plotly::ImageFormat::SVG,
        1920,
        400,
        1.0,
    );
    let t_plot_json_path = output_dir.join("t_plot.json");
    std::fs::write(t_plot_json_path, t_plot.to_json())
        .expect("Failed to write t_plot to JSON file");
    t_plot.show();

    Ok(())
}

#[cfg(test)]
mod tests {}
