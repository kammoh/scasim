use indicatif::ProgressStyle;
use itertools::Itertools;
use log::{debug, info};
use num_format::{Locale, ToFormattedString};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;

pub mod fst;
pub mod optional_filter;
pub mod plot;
pub mod power_model;

pub use fst::*;
pub use optional_filter::*;
pub use power_model::*;

pub fn markers_to_time_indices(
    meta_markers: &[(u64, u64, u16)],
    time_table: &[u64],
) -> Vec<(usize, usize, u16)> {
    // convert the markers to time index intervals
    // each marker has 3 u64 elements: start_time, end_time, and label
    // returns a vector of tuples of (low_index, high_index, label) where low_index..high_index is the range of time indices for the marker
    // use the fact that both meta_markers and time_table are sorted by time

    meta_markers
        .into_iter()
        .map(|marker| {
            // let start_time = marker[0];
            // let end_time = marker[1];
            // let label = marker[2] as u16;
            let (start_time, end_time, label) = marker.to_owned();

            // find the start index
            let low_index = time_table.binary_search(&start_time).unwrap_or_else(|x| x);
            // find the end index
            let high_index = time_table.binary_search(&end_time).unwrap_or_else(|x| x);

            (low_index, high_index, label)
        })
        .collect()
}

pub fn load_waveform<P: AsRef<Path>>(
    filename: P,
    multi_thread: bool,
    show_progress: bool,
) -> Result<(Vec<(wellen::SignalRef, wellen::Signal)>, Vec<u64>), wellen::WellenError> {
    let load_opts = wellen::LoadOptions {
        multi_thread,
        remove_scopes_with_empty_name: false,
    };
    // load header
    let header = wellen::viewers::read_header_from_file(&filename, &load_opts)
        .expect("Failed to load file!");

    let body_len = header.body_len;
    let (body_progress, progress) = if !show_progress || body_len == 0 {
        debug!("show_progress: {}, body_len: {}", show_progress, body_len);
        (None, None)
    } else {
        let p = Arc::new(AtomicU64::new(0));
        let p_out = p.clone();
        let done = Arc::new(AtomicBool::new(false));
        let done_out = done.clone();
        let t = thread::spawn(move || {
            let bar = indicatif::ProgressBar::new(body_len);
            bar.set_style(
                ProgressStyle::with_template(
                    "[{elapsed_precise}] {bar:40.cyan/blue} {decimal_bytes} ({percent_precise}%)",
                )
                .unwrap(),
            );
            loop {
                // always update
                let new_value = p.load(Ordering::SeqCst);
                bar.set_position(new_value);
                thread::sleep(std::time::Duration::from_millis(500));
                // see if we are done
                let now_done = done.load(Ordering::SeqCst);
                if now_done {
                    if bar.position() != body_len {
                        debug!(
                            "Final progress value was: {}, expected {}",
                            bar.position(),
                            body_len
                        );
                    }
                    bar.finish_and_clear();
                    break;
                }
            }
        });

        (Some(p_out), Some((done_out, t)))
    };
    // load body
    let hierarchy = header.hierarchy;

    let start_time = std::time::Instant::now();
    let body = wellen::viewers::read_body(header.body, &hierarchy, body_progress)
        .expect("Failed to load the waveform body!");
    if let Some((done, t)) = progress {
        done.store(true, Ordering::SeqCst);
        t.join().unwrap();
    }
    info!("Read body in {:.2}s", start_time.elapsed().as_secs_f32());

    info!(
        "Total number of samples: {}",
        body.time_table.len().to_formatted_string(&Locale::en)
    );

    let signal_refs = hierarchy.iter_vars().map(|v| v.signal_ref()).collect_vec();

    let mut wave_source = body.source;

    wave_source.print_statistics();

    info!(
        "Loading {} signals..",
        signal_refs.len().to_formatted_string(&Locale::en)
    );
    let start_time = std::time::Instant::now();
    // wave_source.print_statistics();
    let signals = wave_source.load_signals(&signal_refs, &hierarchy, load_opts.multi_thread);
    info!(
        "Loaded signals in {:.2}s",
        start_time.elapsed().as_secs_f32()
    );

    Ok((signals, body.time_table))
}

pub fn generate_power_trace<F: Fn(&(&u64, f32)) -> bool>(
    signals: &[(wellen::SignalRef, wellen::Signal)],
    time_table: &[u64],
    filter_predicate: F,
    //    filter_predicate: Option<fn((u64, f32)) -> bool>,
    // filter_predicate: fn(&(&u64, f32)) -> bool,
    do_filter: bool,
) -> Result<(Vec<u64>, Vec<f32>), wellen::WellenError> {
    let mut power_table = vec![0f32; time_table.len()];

    for (_, signal) in signals.iter() {
        let mut prev_value: Option<wellen::SignalValue> = None;
        for (time_index, new_value) in signal.iter_changes() {
            if let Some(prev_value) = prev_value {
                // we have a previous value, compute the power
                power_table[time_index as usize] += power_model(&prev_value, &new_value);
            }
            prev_value = Some(new_value);
        }
        // debug!("{}: {}", s.full_name(&hierarchy), signal.size_in_memory());
    }
    let mut leftover = 0.0;

    let (left, right): (Vec<u64>, Vec<f32>) = time_table
        .iter()
        .zip(power_table)
        .opt_filter(filter_predicate, do_filter) // filter out zero power values
        .tuple_windows()
        .filter_map(|((t0, p0), (t1, _))| {
            if t0 == t1 {
                // If two consecutive time points are the same, average their power values
                leftover = p0;
                None
            } else {
                let p: f32 = p0 + leftover;
                leftover = 0.0; // reset leftover
                Some((t0, p))
            }
        })
        .unzip(); // .collect::<Vec<_>>();

    Ok((left, right))
}

/// Convert a waveform file to a power trace.
///  * `filename` is the path to the waveform file.
///  * `multi_thread` enables multi-threaded loading of the waveform and signals.
///  * `show_progress` enables a progress bar while loading the file.
///  * `filter_predicate` is an optional function that filters the time points and power values.
/// returns a tuple of two vectors: time points and power values.
pub fn wave_to_powertrace<F: Fn(&(&u64, f32)) -> bool, P: AsRef<Path>>(
    filename: P,
    multi_thread: bool,
    show_progress: bool,
    filter_predicate: F,
    do_filter: bool,
) -> Result<(Vec<u64>, Vec<f32>), wellen::WellenError> {
    let (signals, time_table) = load_waveform(filename, multi_thread, show_progress)?;
    generate_power_trace(&signals, &time_table, filter_predicate, do_filter)
}
