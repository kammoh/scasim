//! Compares the traces of a real batch with the `traces.npz` that an earlier version wrote.
//! Run with:
//!   SCASIM_REAL_BATCH=/path/to/batch cargo test --release --test real_data -- --ignored
//! The batch directory must contain the metadata file, the waveform, and `traces.npz`.
//! The test only reads the directory.

use ndarray::Array1;
use ndarray_npz::NpzReader;
use scasim::hierarchy::Selection;
use std::path::PathBuf;

#[test]
#[ignore = "needs SCASIM_REAL_BATCH"]
fn real_batch_matches_reference_npz() {
    let dir = PathBuf::from(std::env::var("SCASIM_REAL_BATCH").expect("set SCASIM_REAL_BATCH"));
    let meta_path = ["meta.json.gz", "meta.json"]
        .iter()
        .map(|n| dir.join(n))
        .find(|p| p.exists())
        .expect("no meta.json(.gz) in SCASIM_REAL_BATCH");
    let meta = scasim::batch::read_batch_meta(&meta_path).unwrap();
    let (traces, labels, diagnostics) =
        scasim::batch::batch_traces(&meta, &Selection::all()).unwrap();
    eprintln!(
        "toggles: {} in total, {} kept, {} inside segments; {:?}",
        diagnostics.total_toggles,
        diagnostics.kept_toggles,
        diagnostics.segment_toggles,
        diagnostics.info
    );

    let mut npz = NpzReader::new(std::fs::File::open(dir.join("traces.npz")).unwrap()).unwrap();
    let names = npz.names().unwrap();
    let ref_labels: Array1<u16> = npz.by_name("labels").unwrap();
    assert_eq!(labels, ref_labels, "labels differ");
    let trace_names: Vec<_> = names
        .iter()
        .filter(|n| n.starts_with("trace_"))
        .cloned()
        .collect();
    assert_eq!(
        trace_names.len(),
        traces.nrows(),
        "number of traces differs"
    );
    for name in trace_names {
        let index: usize = name
            .trim_start_matches("trace_")
            .trim_end_matches(".npy")
            .parse()
            .unwrap();
        let reference: Array1<f32> = npz.by_name(&name).unwrap();
        assert_eq!(traces.row(index), reference, "{name} differs");
    }
}
