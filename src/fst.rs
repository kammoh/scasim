use fst_reader::{FstFilter, FstReader, FstSignalValue};

use gxhash::{HashMap, HashMapExt};
use itertools::Itertools;
use log::info;
use miette::{Context, IntoDiagnostic};
use ndarray::{Array1, Array2};

use std::io::BufReader;
use std::path::Path;

use crate::{Hamming, markers_to_time_indices, power_model};

impl<'a> Hamming for &'a [u8] {
    #[inline(always)]
    fn hamming_weight(&self) -> u32 {
        self.into_iter().fold(0, |w, byte| w + byte.count_ones()) as u32
    }

    #[inline(always)]
    fn hamming_distance(&self, other: &Self) -> u32 {
        self.into_iter()
            .zip(other.into_iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }
}

pub fn traces_from_fst<P: AsRef<Path>, F: Fn(u64) -> bool>(
    filename: P,
    meta_markers: &[(u64, u64, u16)],
    time_filter: F,
) -> miette::Result<(Array2<f32>, Array1<u16>, Vec<u64>)> {
    let file_reader = BufReader::new(std::fs::File::open(filename).into_diagnostic()?);
    let mut fst_reader = FstReader::open_and_read_time_table(file_reader).into_diagnostic()?;
    let time_table = fst_reader
        .get_time_table()
        .wrap_err("Failed to read time table from FST file. Is the file valid?")?
        .to_vec();
    println!("Converting markers to time indices...");
    let start_time = std::time::Instant::now();
    let time_indices_and_labels = markers_to_time_indices(meta_markers, &time_table);
    println!(
        "Converted markers to time indices in {:.2}s",
        start_time.elapsed().as_secs_f32()
    );

    let num_traces = meta_markers.len();

    let (min_len, max_len) = time_indices_and_labels
        .iter()
        .map(|(lo, hi, _)| hi - lo)
        .minmax()
        .into_option()
        .expect("No time indices found");

    info!("Num traces: {num_traces}, Max length of traces: {max_len}, Min length: {min_len}");
    assert!(
        num_traces > 0,
        "No traces found in the FST file. Please check the file and markers."
    );
    assert!(
        num_traces == time_indices_and_labels.len(),
        "Number of traces ({num_traces}) does not match number of time indices ({})",
        time_indices_and_labels.len()
    );

    let header = fst_reader.get_header();

    // let mut traces_array = Array2::<f32>::zeros((time_indices_and_labels.len(), max_len));
    // let mut labels_array = Array1::<u16>::zeros(time_indices_and_labels.len());
    // let (num_traces, cur_samples_per_trace) = traces_array.dim();

    let mut all_traces = Array2::<f32>::zeros((num_traces, max_len));
    let mut labels = Array1::<u16>::zeros(num_traces);

    meta_markers
        .into_iter()
        .enumerate()
        .for_each(|(trace_index, (start, end, label))| {
            labels[trace_index] = *label;

            let low_index = time_table.binary_search(&start).unwrap_or_else(|x| x);
            // find the end index
            let high_index = time_table.binary_search(&end).unwrap_or_else(|x| x);

            let time_table_slice = &time_table[low_index..high_index];
            let mut last_values: HashMap<u32, Vec<u8>> =
                HashMap::with_capacity(header.var_count as usize);
            let filter = FstFilter::filter_time(*start, *end - 1);
            fst_reader
                .read_signals(
                    &filter,
                    |time, signal_handle, signal_value| match signal_value {
                        FstSignalValue::String(signal_value) => {
                            let time_index =
                                time_table_slice.binary_search(&time).unwrap_or_else(|x| x);

                            // if time_filter.as_ref().map(|f| f(time)).unwrap_or(true) {
                            if time_filter(time) {
                                let sig = signal_handle.get_index() as u32;

                                // println!(
                                //     "Processing signal: {}, time: {}, time_index: {}",
                                //     sig, time, time_index
                                // );

                                if let Some(last_value) = last_values.get(&sig) {
                                    all_traces[[trace_index, time_index]] +=
                                        power_model(&last_value.as_slice(), &signal_value);
                                }

                                last_values.insert(sig, signal_value.to_vec());
                            }
                        }
                        _ => {}
                    },
                )
                .expect("Failed to read signals from FST file");
        });

    // let mut leftover = 0.0;

    // let (left, right): (Vec<u64>, Vec<f32>) = time_table
    //     .iter()
    //     .zip(power_table)
    //     .opt_filter(filter_predicate, do_filter) // filter out zero power values
    //     .tuple_windows()
    //     .filter_map(|((t0, p0), (t1, _))| {
    //         if t0 == t1 {
    //             // If two consecutive time points are the same, average their power values
    //             leftover = p0;
    //             None
    //         } else {
    //             let p: f32 = p0 + leftover;
    //             leftover = 0.0; // reset leftover
    //             Some((t0, p))
    //         }
    //     })
    //     .unzip(); // .collect::<Vec<_>>();

    Ok((all_traces, labels, time_table))
}
