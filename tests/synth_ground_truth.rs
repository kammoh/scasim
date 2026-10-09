//! Ground truth for `tvla`: waveforms with planted leaks from `scasim::synth`, and the results
//! that `tvla` must give. Each test runs the `tvla` binary on generated files in a temp dir.

use ndarray::{Array1, Array2};
use ndarray_npz::NpzReader;
use scasim::batch::read_batch_meta;
use scasim::power::edges::EdgeKind;
use scasim::stats::threshold::{bonferroni, family_size, t_bonferroni};
use scasim::synth::*;
use std::path::{Path, PathBuf};
use std::process::Command;

const ALPHA: f64 = 1e-5;
/// The leak is in the first channel: `--per-scope tb.dut` with the edges of `tb.clk`.
const EDGES: [&str; 6] = [
    "--clock",
    "tb.clk",
    "--include",
    "scope:tb.dut",
    "--per-scope",
    "tb.dut",
];

/// The result files of one `tvla` run.
struct Run {
    out: PathBuf,
    stderr: String,
}

fn tvla(list: &Path, out: &Path, args: &[&str]) -> Run {
    let output = Command::new(env!("CARGO_BIN_EXE_tvla"))
        .env_remove("RUST_LOG")
        .arg("--meta-list")
        .arg(list)
        .arg("--ttest-output-dir")
        .arg(out)
        .arg("--plot=false")
        .args(args)
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    assert!(output.status.success(), "{stderr}");
    Run {
        out: out.to_path_buf(),
        stderr,
    }
}

fn read<T: ndarray::Dimension>(path: &Path, name: &str) -> ndarray::Array<f64, T> {
    NpzReader::new(std::fs::File::open(path).unwrap())
        .unwrap()
        .by_name(name)
        .unwrap()
}

impl Run {
    /// `t_values.npz` of the whole selection, shape `(orders, samples)`.
    fn t(&self) -> Array2<f64> {
        read(&self.out.join("t_values.npz"), "t_values")
    }

    fn chi2(&self) -> Array1<f64> {
        read(&self.out.join("chi2.npz"), "neg_log10_p")
    }

    /// The names of the channels, in the order of the arrays.
    fn channels(&self) -> Vec<String> {
        let text = std::fs::read_to_string(self.out.join("channels.txt")).unwrap();
        text.lines().map(String::from).collect()
    }

    fn channel_index(&self, name: &str) -> usize {
        self.channels().iter().position(|c| c == name).unwrap()
    }

    /// `t` of one channel, shape `(orders, samples)`.
    fn channel_t(&self, name: &str) -> Array2<f64> {
        let i = self.channel_index(name);
        read(&self.out.join("t_values_channels.npz"), &format!("t_{i}"))
    }

    fn channel_chi2(&self, name: &str) -> Array1<f64> {
        let i = self.channel_index(name);
        read(&self.out.join("chi2_channels.npz"), &format!("chi2_{i}"))
    }

    /// The rows of `channels.tsv` as `(column name -> value)` maps, best channel first.
    fn ranking(&self) -> Vec<std::collections::HashMap<String, String>> {
        let text = std::fs::read_to_string(self.out.join("channels.tsv")).unwrap();
        let mut lines = text.lines();
        let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
        lines
            .map(|l| {
                header
                    .iter()
                    .zip(l.split('\t'))
                    .map(|(h, v)| (h.to_string(), v.to_string()))
                    .collect()
            })
            .collect()
    }

    /// The toggles inside the bins, before the first edge, and after the last edge.
    fn edge_toggles(&self) -> [u64; 3] {
        let line = self
            .stderr
            .lines()
            .find(|l| l.contains("inside the bins"))
            .unwrap_or_else(|| panic!("no edge report in\n{}", self.stderr));
        let numbers = numbers_after(line, "toggles: ");
        [numbers[0], numbers[1], numbers[2]]
    }

    /// The toggles in total and at the sampled time points (legacy sampling).
    fn legacy_toggles(&self) -> [u64; 2] {
        let line = self
            .stderr
            .lines()
            .find(|l| l.contains("at the sampled time points"))
            .unwrap();
        let numbers = numbers_after(line, "toggles: ");
        [numbers[0], numbers[1]]
    }
}

/// The unsigned integers in the text after `marker`.
fn numbers_after(text: &str, marker: &str) -> Vec<u64> {
    let rest = &text[text.find(marker).unwrap() + marker.len()..];
    rest.split(|c: char| !c.is_ascii_digit())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap())
        .collect()
}

/// The samples where `|t| > threshold`.
fn exceedances(row: ndarray::ArrayView1<f64>, threshold: f64) -> Vec<usize> {
    row.iter()
        .enumerate()
        .filter(|(_, t)| t.abs() > threshold)
        .map(|(i, _)| i)
        .collect()
}

/// A spec with three scopes of one register each. The leaked register is the only register of
/// its scope, so the channel of that scope sees the leak without other noise.
fn spec() -> SynthSpec {
    SynthSpec {
        scopes: 3,
        registers: 1,
        traces: 1000,
        cycles: 6,
        seed: 11,
        ..SynthSpec::default()
    }
}

fn leak(kind: LeakKind, strength: f64) -> Leak {
    Leak {
        scope: 1,
        register: 0,
        cycle: 3,
        kind,
        strength,
    }
}

fn generate_into(spec: &SynthSpec, dir: &tempfile::TempDir) -> SynthOutput {
    generate(spec, &dir.path().join("data")).unwrap()
}

#[test]
fn a_mean_leak_is_found_by_order_1_at_the_planted_sample() {
    let spec = SynthSpec {
        leaks: vec![leak(LeakKind::Mean, 1.0)],
        ..spec()
    };
    let dir = tempfile::tempdir().unwrap();
    let data = generate_into(&spec, &dir);
    let run = tvla(
        &data.meta_list,
        &dir.path().join("out"),
        &["-d", "2"]
            .iter()
            .chain(&EDGES)
            .copied()
            .collect::<Vec<_>>(),
    );
    let sample = spec.leak_sample(&spec.leaks[0], EdgeKind::Rising).unwrap();
    assert_eq!(sample, 3);
    // Whole selection: the only order-1 exceedance is at the planted sample.
    assert_eq!(exceedances(run.t().row(0), 4.5), [sample]);
    // Channels: the scope with the leak ranks first, and its exceedance is at the sample.
    let ranking = run.ranking();
    assert_eq!(ranking[0]["channel"], "tb.dut.s1");
    assert_eq!(ranking[0]["sample_d1"], sample.to_string());
    assert_eq!(
        exceedances(run.channel_t("tb.dut.s1").row(0), 4.5),
        [sample]
    );
    for other in ["tb.dut.s0", "tb.dut.s2"] {
        assert!(exceedances(run.channel_t(other).row(0), 4.5).is_empty());
    }
}

#[test]
fn a_variance_leak_is_found_by_order_2_and_not_by_order_1() {
    let spec = SynthSpec {
        leaks: vec![leak(LeakKind::Variance, 3.0)],
        ..spec()
    };
    let dir = tempfile::tempdir().unwrap();
    let data = generate_into(&spec, &dir);
    let args: Vec<&str> = ["-d", "2"].iter().chain(&EDGES).copied().collect();
    let run = tvla(&data.meta_list, &dir.path().join("out"), &args);
    let t = run.channel_t("tb.dut.s1");
    assert!(exceedances(t.row(0), 4.5).is_empty(), "order 1: {t:?}");
    assert_eq!(exceedances(t.row(1), 4.5), [3], "order 2: {t:?}");
    assert!(exceedances(run.t().row(0), 4.5).is_empty());
    assert_eq!(run.ranking()[0]["channel"], "tb.dut.s1");
    assert_eq!(run.ranking()[0]["sample_d2"], "3");
}

#[test]
fn an_equal3_leak_is_missed_by_orders_1_to_3_and_found_by_order_4_and_chi2() {
    let spec = SynthSpec {
        leaks: vec![leak(LeakKind::Equal3, 0.0)],
        ..spec()
    };
    let dir = tempfile::tempdir().unwrap();
    let data = generate_into(&spec, &dir);
    let args: Vec<&str> = ["-d", "4"].iter().chain(&EDGES).copied().collect();
    let run = tvla(&data.meta_list, &dir.path().join("out"), &args);
    let t = run.channel_t("tb.dut.s1");
    for order in 0..3 {
        assert!(
            exceedances(t.row(order), 4.5).is_empty(),
            "order {}: {t:?}",
            order + 1
        );
    }
    assert_eq!(exceedances(t.row(3), 4.5), [3], "order 4: {t:?}");
    let chi2 = run.channel_chi2("tb.dut.s1");
    let above: Vec<usize> = (0..chi2.len()).filter(|&i| chi2[i] > 5.0).collect();
    assert_eq!(above, [3], "chi2: {chi2:?}");
    assert_eq!(run.ranking()[0]["channel"], "tb.dut.s1");
}

#[test]
fn a_deterministic_leak_gives_an_infinite_t_and_ranks_first() {
    // No noise: the other samples of the channel are constant, and the leak sample has no
    // variance in either class.
    let spec = SynthSpec {
        density: 0.0,
        leaks: vec![leak(LeakKind::Deterministic, 1.0)],
        ..spec()
    };
    let dir = tempfile::tempdir().unwrap();
    let data = generate_into(&spec, &dir);
    let args: Vec<&str> = ["-d", "2"].iter().chain(&EDGES).copied().collect();
    let run = tvla(&data.meta_list, &dir.path().join("out"), &args);
    let t = run.channel_t("tb.dut.s1");
    assert!(t[[0, 3]].is_infinite(), "{t:?}");
    let ranking = run.ranking();
    assert_eq!(ranking[0]["channel"], "tb.dut.s1");
    assert!(ranking[0]["infinite_t"].parse::<usize>().unwrap() >= 1);
    assert_eq!(ranking[1]["infinite_t"], "0");
}

/// The number of samples above the Bonferroni thresholds of `tvla`: `(t, chi2)`.
fn bonferroni_exceedances(run: &Run, order: usize) -> (usize, usize) {
    let t = run.t();
    let samples = t.ncols() as u64;
    let family = family_size(1, samples, order as u64).unwrap();
    let t_limit = t_bonferroni(ALPHA, family);
    let chi2_limit = bonferroni(ALPHA, samples);
    let t_count = t.iter().filter(|v| v.abs() > t_limit).count();
    let chi2_count = run.chi2().iter().filter(|&&p| p > chi2_limit).count();
    (t_count, chi2_count)
}

#[test]
fn no_leak_and_shuffled_runs_have_no_bonferroni_exceedance() {
    // Under the null hypothesis, the chance of any exceedance of a Bonferroni threshold is at
    // most 1e-5 for each family (the t-values, the chi-squared values). The seeds are fixed, so
    // the test cannot flake. With 1000 traces or more, the t-statistic of a sum of bit flips is
    // close to normal. Margin at these seeds: the largest |t| is 1.5 (no leak) and 1.7
    // (shuffled), against the threshold 4.93. The largest chi-squared value is 1.5 and 0.4,
    // against 5.78.
    let dir = tempfile::tempdir().unwrap();
    let plain = SynthSpec {
        registers: 2,
        traces: 2000,
        ..spec()
    };
    let leaky = SynthSpec {
        leaks: vec![leak(LeakKind::Mean, 1.0)],
        ..spec()
    };
    let data = generate(&plain, &dir.path().join("plain")).unwrap();
    let run = tvla(
        &data.meta_list,
        &dir.path().join("out1"),
        &["-d", "2", "--include", "scope:tb.dut", "--clock", "tb.clk"],
    );
    assert_eq!(bonferroni_exceedances(&run, 2), (0, 0));
    // A leak, with the labels shuffled: the association is gone.
    let data = generate(&leaky, &dir.path().join("leaky")).unwrap();
    let args = ["-d", "2", "--include", "scope:tb.dut", "--clock", "tb.clk"];
    let real = tvla(&data.meta_list, &dir.path().join("out2"), &args);
    assert!(bonferroni_exceedances(&real, 2).0 >= 1);
    let mut shuffled_args = args.to_vec();
    shuffled_args.extend(["--shuffle-labels", "5"]);
    let shuffled = tvla(&data.meta_list, &dir.path().join("out3"), &shuffled_args);
    assert_eq!(bonferroni_exceedances(&shuffled, 2), (0, 0));
}

/// Checks that every toggle of the selection is inside a bin or before or after all bins.
fn assert_conservation(run: &Run, data: &SynthOutput) {
    let [inside, before, after] = run.edge_toggles();
    assert_eq!(
        inside + before + after,
        data.stats.dut_toggles(),
        "{}",
        run.stderr
    );
}

fn edge_args(kind: &str) -> Vec<&str> {
    let mut args = vec!["-d", "1", "--edges", kind];
    args.extend(EDGES);
    args
}

#[test]
fn two_clock_domains_conserve_the_toggles_and_the_leak_is_at_its_time() {
    // Scope s1 runs on a clock with period 12. Its cycle 2 is at tick 24 of the segment, so it
    // is in the bin of the edge 2 of the clock with period 10 (ticks 20 to 30).
    let spec = SynthSpec {
        clock2: Some(Clock2 {
            period: 12,
            phase: 0,
        }),
        leaks: vec![leak(LeakKind::Mean, 1.0).at_cycle(2)],
        ..spec()
    };
    let dir = tempfile::tempdir().unwrap();
    let data = generate_into(&spec, &dir);
    assert_eq!(spec.leak_sample(&spec.leaks[0], EdgeKind::Rising), Some(2));
    let run = tvla(
        &data.meta_list,
        &dir.path().join("edges"),
        &edge_args("rising"),
    );
    assert_conservation(&run, &data);
    assert_eq!(run.ranking()[0]["channel"], "tb.dut.s1");
    assert_eq!(run.ranking()[0]["sample_d1"], "2");
    // Legacy sampling keeps only the ticks that are multiples of 10 and drops the rest.
    let legacy = tvla(
        &data.meta_list,
        &dir.path().join("legacy"),
        &["-d", "1", "--include", "scope:tb.dut"],
    );
    let [total, kept] = legacy.legacy_toggles();
    assert_eq!(total, data.stats.dut_toggles());
    assert!(kept < total, "{total} {kept}");
}

#[test]
fn a_gated_clock_has_fewer_samples_and_the_leak_keeps_its_place() {
    // The clock stops for 2 cycles in every 4: the cycles 2, 3, 6, and 7 of 8 have no edge.
    // Cycle 4 is the third edge, so the sample is 2.
    let spec = SynthSpec {
        cycles: 8,
        gating: Some(Gating { stop: 2, every: 4 }),
        leaks: vec![leak(LeakKind::Mean, 1.0).at_cycle(4)],
        ..spec()
    };
    let dir = tempfile::tempdir().unwrap();
    let data = generate_into(&spec, &dir);
    assert_eq!(spec.samples(EdgeKind::Rising), 4);
    assert_eq!(spec.leak_sample(&spec.leaks[0], EdgeKind::Rising), Some(2));
    let run = tvla(
        &data.meta_list,
        &dir.path().join("out"),
        &edge_args("rising"),
    );
    assert_eq!(run.t().ncols(), 4);
    assert_conservation(&run, &data);
    assert_eq!(run.ranking()[0]["channel"], "tb.dut.s1");
    assert_eq!(run.ranking()[0]["sample_d1"], "2");
}

#[test]
fn glitches_inside_the_cycle_are_kept_by_edges_and_dropped_by_legacy() {
    let spec = SynthSpec {
        glitch: Some(Glitch { delta: 3, bits: 2 }),
        ..spec()
    };
    let dir = tempfile::tempdir().unwrap();
    let data = generate_into(&spec, &dir);
    // Each register flips 2 bits and flips them back, in every cycle.
    let glitch = 4 * (spec.scopes * spec.registers * spec.cycles * spec.traces) as u64;
    let edges = tvla(
        &data.meta_list,
        &dir.path().join("edges"),
        &edge_args("rising"),
    );
    assert_conservation(&edges, &data);
    assert_eq!(edges.edge_toggles()[0], data.stats.dut_toggles());
    let legacy = tvla(
        &data.meta_list,
        &dir.path().join("legacy"),
        &["-d", "1", "--include", "scope:tb.dut"],
    );
    let [total, kept] = legacy.legacy_toggles();
    assert_eq!(total, data.stats.dut_toggles());
    assert_eq!(kept, total - glitch);
}

#[test]
fn both_edges_and_falling_edges_conserve_the_toggles() {
    let dir = tempfile::tempdir().unwrap();
    // Registers update on both edges. The leak is at the rising edge of cycle 1, which is the
    // third of the 12 edges (rise 0, fall 0, rise 1).
    let both = SynthSpec {
        edge: EdgeKind::Both,
        leaks: vec![leak(LeakKind::Mean, 1.0).at_cycle(1)],
        ..spec()
    };
    let data = generate(&both, &dir.path().join("both")).unwrap();
    assert_eq!(both.samples(EdgeKind::Both), 12);
    assert_eq!(both.leak_sample(&both.leaks[0], EdgeKind::Both), Some(2));
    let run = tvla(
        &data.meta_list,
        &dir.path().join("out1"),
        &edge_args("both"),
    );
    assert_eq!(run.t().ncols(), 12);
    assert_conservation(&run, &data);
    assert_eq!(run.ranking()[0]["channel"], "tb.dut.s1");
    assert_eq!(run.ranking()[0]["sample_d1"], "2");
    // Registers update on the falling edge, at tick 6 + 10 k with the phase 1. Cycle 2 is at tick 26.
    let falling = SynthSpec {
        edge: EdgeKind::Falling,
        phase: 1,
        leaks: vec![leak(LeakKind::Mean, 1.0).at_cycle(2)],
        ..spec()
    };
    let data = generate(&falling, &dir.path().join("falling")).unwrap();
    assert_eq!(falling.samples(EdgeKind::Falling), 6);
    assert_eq!(
        falling.leak_sample(&falling.leaks[0], EdgeKind::Falling),
        Some(2)
    );
    let run = tvla(
        &data.meta_list,
        &dir.path().join("out2"),
        &edge_args("falling"),
    );
    assert_conservation(&run, &data);
    assert_eq!(run.ranking()[0]["channel"], "tb.dut.s1");
    assert_eq!(run.ranking()[0]["sample_d1"], "2");
}

#[test]
fn fst_and_vcd_of_the_same_spec_give_the_same_t_values() {
    let dir = tempfile::tempdir().unwrap();
    let base = SynthSpec {
        leaks: vec![leak(LeakKind::Mean, 1.0)],
        glitch: Some(Glitch { delta: 3, bits: 1 }),
        traces: 300,
        ..spec()
    };
    let mut results = Vec::new();
    for (name, format) in [("fst", Format::Fst), ("vcd", Format::Vcd)] {
        let spec = SynthSpec {
            format,
            ..base.clone()
        };
        let data = generate(&spec, &dir.path().join(name)).unwrap();
        let edges = tvla(
            &data.meta_list,
            &dir.path().join(format!("{name}-e")),
            &edge_args("rising"),
        );
        let legacy = tvla(
            &data.meta_list,
            &dir.path().join(format!("{name}-l")),
            &["-d", "2", "--use-existing=false"],
        );
        let bits = |a: Array2<f64>| a.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        results.push((bits(edges.t()), bits(legacy.t()), data.stats));
    }
    assert_eq!(results[0], results[1]);
}

#[test]
fn the_output_is_deterministic_and_the_memory_does_not_grow_with_the_traces() {
    let dir = tempfile::tempdir().unwrap();
    let spec = SynthSpec {
        traces: 50,
        leaks: vec![leak(LeakKind::Variance, 2.0)],
        ..spec()
    };
    for format in [Format::Fst, Format::Vcd] {
        let spec = SynthSpec {
            format,
            ..spec.clone()
        };
        let a = generate(&spec, &dir.path().join("a")).unwrap();
        let b = generate(&spec, &dir.path().join("b")).unwrap();
        for (x, y) in [(&a.waves[0], &b.waves[0]), (&a.metas[0], &b.metas[0])] {
            assert_eq!(std::fs::read(x).unwrap(), std::fs::read(y).unwrap());
        }
        assert_eq!(a.stats, b.stats);
    }
    // The generator keeps the events of one segment, not of all traces.
    let small = generate(&spec, &dir.path().join("small")).unwrap();
    let large = generate(
        &SynthSpec {
            traces: 2000,
            ..spec.clone()
        },
        &dir.path().join("large"),
    )
    .unwrap();
    assert_eq!(small.stats.peak_events, large.stats.peak_events);
}

#[test]
fn the_metadata_matches_the_spec() {
    let dir = tempfile::tempdir().unwrap();
    let spec = SynthSpec {
        batches: 2,
        traces: 40,
        ..spec()
    };
    let data = generate(&spec, &dir.path().join("data")).unwrap();
    assert_eq!(data.metas.len(), 2);
    let list = std::fs::read_to_string(&data.meta_list).unwrap();
    assert_eq!(list.lines().count(), 2);
    let mut classes = [0u64; 2];
    for (batch, meta) in data.metas.iter().enumerate() {
        let meta = read_batch_meta(meta).unwrap();
        assert_eq!(meta.clock_period, Some(spec.period));
        assert!(meta.trace_path.exists());
        assert_eq!(meta.markers.len(), 40);
        for (i, &(start, end, label)) in meta.markers.iter().enumerate() {
            assert_eq!(end - start, spec.cycles as u64 * spec.period);
            assert_eq!(label, spec.label(batch, i));
            assert_eq!(start % spec.period, 0);
            classes[label as usize] += 1;
        }
    }
    assert_eq!(classes, data.stats.class_counts);
    assert!(classes.iter().all(|&c| c > 10), "{classes:?}");
}

#[test]
fn invalid_specs_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let bad = [
        SynthSpec {
            width: 4,
            leaks: vec![leak(LeakKind::Equal3, 0.0)],
            ..spec()
        },
        SynthSpec {
            leaks: vec![Leak {
                scope: 9,
                ..leak(LeakKind::Mean, 1.0)
            }],
            ..spec()
        },
        SynthSpec {
            period: 7,
            ..spec()
        },
        SynthSpec {
            gating: Some(Gating { stop: 2, every: 6 }),
            leaks: vec![leak(LeakKind::Mean, 1.0).at_cycle(4)],
            ..spec()
        },
    ];
    for spec in bad {
        assert!(
            generate(&spec, &dir.path().join("bad")).is_err(),
            "{spec:?}"
        );
    }
}

#[test]
fn strengths_that_plant_no_difference_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let with = |kind, strength| SynthSpec {
        leaks: vec![leak(kind, strength)],
        ..spec()
    };
    // The strength 1 (or anything that rounds to 1) gives class 1 the distribution of class 0.
    for (kind, strength) in [
        (LeakKind::Variance, 1.0),
        (LeakKind::Variance, 1.4),
        (LeakKind::Mean, 0.0),
        (LeakKind::Deterministic, 0.0),
        (LeakKind::Deterministic, 0.4),
    ] {
        assert!(
            with(kind, strength).validate().is_err(),
            "{kind:?} {strength}"
        );
    }
    // The smallest valid strengths.
    for (kind, strength) in [
        (LeakKind::Variance, 2.0),
        (LeakKind::Variance, 1.5),
        (LeakKind::Mean, 0.01),
        (LeakKind::Deterministic, 1.0),
    ] {
        assert!(
            generate(&with(kind, strength), &dir.path().join("ok")).is_ok(),
            "{kind:?} {strength}"
        );
    }
}

/// The value changes of each signal of a VCD file (the signals of the generator use one
/// character as identifier): `signal index -> [(time, value)]`.
fn vcd_changes(path: &Path) -> std::collections::BTreeMap<usize, Vec<(u64, String)>> {
    let text = std::fs::read_to_string(path).unwrap();
    let body = text.split("$enddefinitions $end").nth(1).unwrap();
    let mut changes = std::collections::BTreeMap::<usize, Vec<(u64, String)>>::new();
    let mut time = 0;
    for line in body.lines() {
        if let Some(t) = line.strip_prefix('#') {
            time = t.parse().unwrap();
        } else if let Some(vector) = line.strip_prefix('b') {
            let (value, id) = vector.split_once(' ').unwrap();
            let sig = usize::from(id.as_bytes()[0] - b'!');
            changes.entry(sig).or_default().push((time, value.into()));
        } else if line.len() == 2 && !line.starts_with('$') {
            let sig = usize::from(line.as_bytes()[1] - b'!');
            changes
                .entry(sig)
                .or_default()
                .push((time, line[..1].into()));
        }
    }
    changes
}

#[test]
fn the_noise_of_the_other_registers_does_not_depend_on_the_leak_or_the_labels() {
    // The leak draws its random numbers from its own stream, and the noise of the leaking
    // register is drawn and replaced. So no draw count depends on the class, and every other
    // signal has the same value changes with and without a leak, for every kind and strength.
    let dir = tempfile::tempdir().unwrap();
    let plain = SynthSpec {
        registers: 2,
        traces: 200,
        format: Format::Vcd,
        glitch: Some(Glitch { delta: 3, bits: 1 }),
        ..spec()
    };
    let reference = generate(&plain, &dir.path().join("plain")).unwrap();
    let reference = vcd_changes(&reference.waves[0]);
    // Signals: tb.clk, tb.t0, then tb.dut.s<scope>.r<register>.
    let leaked = 2 + plain.registers;
    let leaks = [
        (LeakKind::Mean, 0.5),
        (LeakKind::Mean, 1.0),
        (LeakKind::Variance, 2.0),
        (LeakKind::Equal3, 0.0),
        (LeakKind::Deterministic, 1.0),
    ];
    for (i, (kind, strength)) in leaks.into_iter().enumerate() {
        let spec = SynthSpec {
            leaks: vec![leak(kind, strength)],
            ..plain.clone()
        };
        let data = generate(&spec, &dir.path().join(format!("leak{i}"))).unwrap();
        let changes = vcd_changes(&data.waves[0]);
        assert!(!changes[&leaked].is_empty());
        for (sig, list) in &reference {
            // The leaking register itself differs (the planted cycle, and the values after it).
            if *sig != leaked {
                assert_eq!(&changes[sig], list, "{kind:?} {strength}: signal {sig}");
            }
        }
    }
}
