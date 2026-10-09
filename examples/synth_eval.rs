//! Evaluates the `tvla` binary on synthetic waveforms with planted leaks (ground truth).
//!
//! ```text
//! cargo build --release --bin tvla
//! cargo run --release --example synth_eval -- SCRATCH_DIR [--tvla PATH] [--sweep FILE.json]
//! ```
//!
//! For each case of the sweep the harness generates a waveform with `scasim::synth`, runs `tvla`
//! and records:
//! - detection: the planted scope ranks first (`--per-scope tb.dut`), the exceedance of the
//!   planted order is at the planted sample, lower orders stay clear, and the trace count at
//!   which the max-|t| curve of the whole selection first passes 4.5;
//! - false alarms: the exceedances above 4.5 and above the Bonferroni value, for specs without
//!   a leak and for runs with `--shuffle-labels`, against the expected counts;
//! - sampling: legacy sampling against edges mode, and the activity outside the bins.
//!
//! `SCRATCH_DIR` receives `synth_eval.tsv`, `synth_eval.md`, and the files of the cases. It must
//! not be inside the repository. The default sweep is built in; `--print-sweep` prints it as
//! JSON, which `--sweep` reads back (a list of cases, see [`Case`]).

use clap::Parser;
use ndarray::{Array1, Array2};
use ndarray_npz::NpzReader;
use scasim::power::edges::EdgeKind;
use scasim::stats::threshold::{bonferroni, family_size, t_bonferroni};
use scasim::synth::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

const ALPHA: f64 = 1e-5;
const T_LIMIT: f64 = 4.5;
const CHI2_LIMIT: f64 = 5.0;

#[derive(Parser)]
struct Args {
    /// Directory for the results and the generated files (not inside the repository).
    out_dir: Option<PathBuf>,
    /// The `tvla` binary.
    #[arg(long, default_value = "target/release/tvla")]
    tvla: PathBuf,
    /// A JSON file with a list of cases instead of the built-in sweep.
    #[arg(long)]
    sweep: Option<PathBuf>,
    /// Print the built-in sweep as JSON and exit.
    #[arg(long)]
    print_sweep: bool,
    /// Keep the generated waveforms (they are deleted after each case by default).
    #[arg(long)]
    keep: bool,
}

/// One planted leak.
#[derive(Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct LeakCfg {
    /// `mean`, `variance`, `equal3`, or `deterministic`.
    kind: String,
    scope: usize,
    register: usize,
    cycle: usize,
    strength: f64,
}

impl Default for LeakCfg {
    fn default() -> Self {
        LeakCfg {
            kind: "mean".into(),
            scope: 1,
            register: 0,
            cycle: 3,
            strength: 1.0,
        }
    }
}

/// One case of the sweep: a spec and what to run. Every field has a default.
#[derive(Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Case {
    name: String,
    /// A label for the tables.
    group: String,
    leak: Option<LeakCfg>,
    scopes: usize,
    registers: usize,
    width: u32,
    density: f64,
    cycles: usize,
    /// Traces in each batch.
    traces: usize,
    batches: usize,
    seed: u64,
    period: u64,
    phase: u64,
    /// `rising`, `falling`, or `both`.
    edge: String,
    /// `[period, phase]` of the second clock.
    clock2: Option<[u64; 2]>,
    /// `[stop, every]`.
    gating: Option<[u64; 2]>,
    /// `[delta, bits]`.
    glitch: Option<[u64; 2]>,
    /// `fst` or `vcd`.
    format: String,
    /// Also run the legacy sampling.
    legacy: bool,
    /// Also run with shuffled labels (the false-alarm columns then come from that run).
    shuffle: bool,
}

impl Default for Case {
    fn default() -> Self {
        Case {
            name: "case".into(),
            group: "power".into(),
            leak: None,
            scopes: 3,
            registers: 1,
            width: 8,
            density: 0.5,
            cycles: 6,
            traces: 400,
            batches: 10,
            seed: 1,
            period: 10,
            phase: 0,
            edge: "rising".into(),
            clock2: None,
            gating: None,
            glitch: None,
            format: "fst".into(),
            legacy: true,
            shuffle: false,
        }
    }
}

impl Case {
    fn edge_kind(&self) -> EdgeKind {
        match self.edge.as_str() {
            "falling" => EdgeKind::Falling,
            "both" => EdgeKind::Both,
            _ => EdgeKind::Rising,
        }
    }

    fn spec(&self) -> Result<SynthSpec, String> {
        let leaks = match &self.leak {
            None => vec![],
            Some(l) => vec![Leak {
                scope: l.scope,
                register: l.register,
                cycle: l.cycle,
                strength: l.strength,
                kind: match l.kind.as_str() {
                    "mean" => LeakKind::Mean,
                    "variance" => LeakKind::Variance,
                    "equal3" => LeakKind::Equal3,
                    "deterministic" => LeakKind::Deterministic,
                    other => return Err(format!("unknown leak kind {other}")),
                },
            }],
        };
        Ok(SynthSpec {
            scopes: self.scopes,
            registers: self.registers,
            width: self.width,
            period: self.period,
            phase: self.phase,
            edge: self.edge_kind(),
            clock2: self.clock2.map(|[period, phase]| Clock2 { period, phase }),
            gating: self.gating.map(|[stop, every]| Gating { stop, every }),
            glitch: self.glitch.map(|[delta, bits]| Glitch {
                delta,
                bits: bits as u32,
            }),
            density: self.density,
            cycles: self.cycles,
            traces: self.traces,
            batches: self.batches,
            seed: self.seed,
            leaks,
            format: if self.format == "vcd" {
                Format::Vcd
            } else {
                Format::Fst
            },
            ..SynthSpec::default()
        })
    }

    /// The order of the t-test that finds the planted leak.
    fn leak_order(&self) -> usize {
        match self.leak.as_ref().map(|l| l.kind.as_str()) {
            Some("variance") => 2,
            Some("equal3") => 4,
            _ => 1,
        }
    }
}

/// The built-in sweep: power (kind, strength, trace count), noise, geometry, structure, and
/// false alarms.
fn default_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    let leak = |kind: &str, strength: f64| {
        Some(LeakCfg {
            kind: kind.into(),
            strength,
            ..LeakCfg::default()
        })
    };
    let kinds = [
        ("mean", 0.25),
        ("mean", 0.5),
        ("mean", 1.0),
        ("variance", 2.0),
        ("variance", 3.0),
        ("equal3", 0.0),
        ("deterministic", 1.0),
    ];
    for (kind, strength) in kinds {
        for total in [1000, 4000, 16000, 64000] {
            // Without noise, a deterministic leak has no variance in the planted channel.
            let density = if kind == "deterministic" { 0.0 } else { 0.5 };
            cases.push(Case {
                name: format!("{kind}-{strength}-n{total}"),
                group: "power".into(),
                leak: leak(kind, strength),
                // Small batches: the curve of the max |t| has one point per batch.
                traces: 100,
                batches: total / 100,
                density,
                ..Case::default()
            });
        }
    }
    let base = Case {
        leak: leak("mean", 1.0),
        traces: 400,
        ..Case::default()
    };
    for density in [0.1, 0.25, 0.5, 0.75] {
        cases.push(Case {
            name: format!("density-{density}"),
            group: "noise".into(),
            // Three noisy registers in the scope of the leak.
            registers: 4,
            density,
            ..base.clone()
        });
    }
    let geometry = [
        ("scopes8-regs4", 8, 4, 8, 6, 3),
        ("width16", 3, 1, 16, 6, 3),
        ("cycles40", 3, 1, 8, 40, 20),
    ];
    for (name, scopes, registers, width, cycles, cycle) in geometry {
        cases.push(Case {
            name: name.into(),
            group: "geometry".into(),
            scopes,
            registers,
            width,
            cycles,
            leak: Some(LeakCfg {
                cycle,
                ..LeakCfg::default()
            }),
            ..base.clone()
        });
    }
    let structure: Vec<(&str, Case)> = vec![
        (
            "clock2",
            Case {
                clock2: Some([12, 0]),
                leak: Some(LeakCfg {
                    cycle: 2,
                    ..LeakCfg::default()
                }),
                ..base.clone()
            },
        ),
        (
            "gating",
            Case {
                cycles: 8,
                gating: Some([2, 4]),
                leak: Some(LeakCfg {
                    cycle: 4,
                    ..LeakCfg::default()
                }),
                ..base.clone()
            },
        ),
        (
            "glitch",
            Case {
                glitch: Some([3, 2]),
                ..base.clone()
            },
        ),
        (
            "both-edges",
            Case {
                edge: "both".into(),
                ..base.clone()
            },
        ),
        (
            "falling-edge",
            Case {
                edge: "falling".into(),
                ..base.clone()
            },
        ),
        (
            "phase2",
            Case {
                phase: 2,
                ..base.clone()
            },
        ),
        (
            "vcd",
            Case {
                format: "vcd".into(),
                ..base.clone()
            },
        ),
    ];
    for (name, case) in structure {
        cases.push(Case {
            name: name.into(),
            group: "structure".into(),
            ..case
        });
    }
    // No leak: false alarms on the plain spec. Many samples per run for more tests.
    for seed in 1..=20 {
        cases.push(Case {
            name: format!("null-s{seed}"),
            group: "null".into(),
            leak: None,
            registers: 2,
            cycles: 50,
            traces: 200,
            seed,
            legacy: false,
            ..Case::default()
        });
    }
    // A leak with shuffled labels: the association is gone, the traces are the same.
    for seed in 1..=10 {
        cases.push(Case {
            name: format!("shuffle-s{seed}"),
            group: "shuffle".into(),
            leak: Some(LeakCfg {
                cycle: 25,
                ..LeakCfg::default()
            }),
            registers: 2,
            cycles: 50,
            traces: 200,
            seed: 100 + seed,
            legacy: false,
            shuffle: true,
            ..Case::default()
        });
    }
    cases
}

/// The user and system CPU time of all finished child processes, in seconds.
fn children_cpu() -> f64 {
    // SAFETY: `getrusage` writes into the struct that we give it.
    let usage = unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_CHILDREN, &mut usage);
        usage
    };
    let seconds = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 * 1e-6;
    seconds(usage.ru_utime) + seconds(usage.ru_stime)
}

fn own_cpu() -> f64 {
    // SAFETY: as above.
    let usage = unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut usage);
        usage
    };
    let seconds = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 * 1e-6;
    seconds(usage.ru_utime) + seconds(usage.ru_stime)
}

/// The result files of one `tvla` run.
struct Run {
    out: PathBuf,
    stderr: String,
    cpu: f64,
}

fn run_tvla(tvla: &Path, list: &Path, out: &Path, args: &[String]) -> Result<Run, String> {
    let before = children_cpu();
    let output = Command::new(tvla)
        .env_remove("RUST_LOG")
        .arg("--meta-list")
        .arg(list)
        .arg("--ttest-output-dir")
        .arg(out)
        .args(args)
        .output()
        .map_err(|e| format!("cannot run {}: {e}", tvla.display()))?;
    let cpu = children_cpu() - before;
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    if !output.status.success() {
        let first = stderr.lines().find(|l| !l.trim().is_empty());
        return Err(format!("tvla failed: {}", first.unwrap_or("no message")));
    }
    Ok(Run {
        out: out.to_path_buf(),
        stderr,
        cpu,
    })
}

fn read<T: ndarray::Dimension>(path: &Path, name: &str) -> Result<ndarray::Array<f64, T>, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    NpzReader::new(file)
        .and_then(|mut npz| npz.by_name(name))
        .map_err(|e| format!("{} {name}: {e}", path.display()))
}

impl Run {
    fn t(&self) -> Result<Array2<f64>, String> {
        read(&self.out.join("t_values.npz"), "t_values")
    }

    fn chi2(&self) -> Result<Array1<f64>, String> {
        read(&self.out.join("chi2.npz"), "neg_log10_p")
    }

    fn channel_names(&self) -> Result<Vec<String>, String> {
        let path = self.out.join("channels.txt");
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(text.lines().map(String::from).collect())
    }

    fn channel_t(&self, name: &str) -> Result<Array2<f64>, String> {
        let i = self.channel_names()?.iter().position(|c| c == name);
        let i = i.ok_or_else(|| format!("no channel {name}"))?;
        read(&self.out.join("t_values_channels.npz"), &format!("t_{i}"))
    }

    fn channel_chi2(&self, name: &str) -> Result<Array1<f64>, String> {
        let i = self.channel_names()?.iter().position(|c| c == name);
        let i = i.ok_or_else(|| format!("no channel {name}"))?;
        read(&self.out.join("chi2_channels.npz"), &format!("chi2_{i}"))
    }

    /// The name of the best channel.
    fn best_channel(&self) -> Result<String, String> {
        let path = self.out.join("channels.tsv");
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        text.lines()
            .nth(1)
            .and_then(|l| l.split('\t').nth(1))
            .map(String::from)
            .ok_or_else(|| "empty channels.tsv".to_string())
    }

    /// The numbers after `marker` in the first log line that contains `needle`.
    fn log_numbers(&self, needle: &str, marker: &str) -> Option<Vec<u64>> {
        let line = self.stderr.lines().find(|l| l.contains(needle))?;
        let rest = &line[line.find(marker)? + marker.len()..];
        Some(
            rest.split(|c: char| !c.is_ascii_digit())
                .filter(|s| !s.is_empty())
                .filter_map(|s| s.parse().ok())
                .collect(),
        )
    }

    /// The fraction of the toggles outside all bins (edges mode).
    fn outside_fraction(&self) -> Option<f64> {
        let n = self.log_numbers("inside the bins", "toggles: ")?;
        let total = n[0] + n[1] + n[2];
        (total > 0).then(|| (n[1] + n[2]) as f64 / total as f64)
    }

    /// The fraction of the toggles that the legacy sampling keeps.
    fn kept_fraction(&self) -> Option<f64> {
        let n = self.log_numbers("at the sampled time points", "toggles: ")?;
        (n[0] > 0).then(|| n[1] as f64 / n[0] as f64)
    }

    /// The first number of traces at which the max-|t| curve of `order` is above `limit`.
    fn first_detection(&self, order: usize, limit: f64) -> Option<u64> {
        let text = std::fs::read_to_string(self.out.join("max_t_values.json")).ok()?;
        let json: serde_json::Value = serde_json::from_str(&text).ok()?;
        let series = json["data"]
            .as_array()?
            .iter()
            .find(|s| s["name"] == format!("d={order}"))?;
        let (x, y) = (series["x"].as_array()?, series["y"].as_array()?);
        x.iter()
            .zip(y)
            .find(|(_, y)| y.as_f64().is_some_and(|y| y > limit))
            .and_then(|(x, _)| x.as_f64())
            .map(|x| x as u64)
    }
}

fn exceed(row: ndarray::ArrayView1<f64>, limit: f64) -> Vec<usize> {
    row.iter()
        .enumerate()
        .filter(|(_, t)| t.abs() > limit)
        .map(|(i, _)| i)
        .collect()
}

/// False-alarm counts of one run (all orders and samples of the whole selection).
#[derive(Default, Clone)]
struct Null {
    t_tests: u64,
    t_conv: u64,
    t_bonf: u64,
    chi2_tests: u64,
    chi2_conv: u64,
    chi2_bonf: u64,
    exp_t_conv: f64,
    exp_chi2_conv: f64,
    max_t: f64,
}

fn null_counts(run: &Run, order: usize) -> Result<Null, String> {
    let t = run.t()?;
    let chi2 = run.chi2()?;
    let samples = t.ncols() as u64;
    let family = family_size(1, samples, order as u64).ok_or("family too large")?;
    let t_bonf = t_bonferroni(ALPHA, family);
    let chi2_bonf = bonferroni(ALPHA, samples);
    let tests = t.len() as u64;
    // P(|Z| > 4.5) for a standard normal Z.
    let p_conv = libm::erfc(T_LIMIT / std::f64::consts::SQRT_2);
    Ok(Null {
        t_tests: tests,
        t_conv: t.iter().filter(|v| v.abs() > T_LIMIT).count() as u64,
        t_bonf: t.iter().filter(|v| v.abs() > t_bonf).count() as u64,
        chi2_tests: chi2.len() as u64,
        chi2_conv: chi2.iter().filter(|&&p| p > CHI2_LIMIT).count() as u64,
        chi2_bonf: chi2.iter().filter(|&&p| p > chi2_bonf).count() as u64,
        exp_t_conv: tests as f64 * p_conv,
        exp_chi2_conv: chi2.len() as f64 * 10f64.powf(-CHI2_LIMIT),
        max_t: t
            .iter()
            .filter(|v| v.is_finite())
            .fold(0.0, |a, v| a.max(v.abs())),
    })
}

/// The recorded values of one case. Empty strings are not applicable.
#[derive(Default)]
struct Row {
    case: String,
    group: String,
    kind: String,
    strength: String,
    total_traces: u64,
    density: f64,
    order: usize,
    error: String,
    // Edges mode, detection.
    rank1: String,
    hit: String,
    extra_exceed: String,
    lower_clear: String,
    chi2_hit: String,
    t_at_leak: String,
    first_detect: String,
    // Sampling.
    outside_frac: String,
    legacy_rank1: String,
    legacy_hit: String,
    legacy_kept_frac: String,
    // False alarms.
    null: Option<Null>,
    // CPU seconds.
    cpu_gen: f64,
    cpu_edges: f64,
    cpu_legacy: f64,
    cpu_shuffle: f64,
}

fn yes(b: bool) -> String {
    if b { "yes" } else { "no" }.into()
}

fn fraction(f: Option<f64>) -> String {
    f.map(|f| format!("{f:.4}")).unwrap_or_default()
}

/// Runs one case.
fn run_case(case: &Case, tvla: &Path, dir: &Path, keep: bool) -> Result<Row, String> {
    let spec = case.spec()?;
    let order = case.leak_order();
    let run_order = order.max(2);
    let mut row = Row {
        case: case.name.clone(),
        group: case.group.clone(),
        kind: case
            .leak
            .as_ref()
            .map_or("none", |l| l.kind.as_str())
            .into(),
        strength: case
            .leak
            .as_ref()
            .map_or(String::new(), |l| l.strength.to_string()),
        total_traces: (case.traces * case.batches) as u64,
        density: case.density,
        order,
        ..Row::default()
    };
    let case_dir = dir.join(&case.name);
    let before = own_cpu();
    let data = generate(&spec, &case_dir.join("data")).map_err(|e| e.to_string())?;
    row.cpu_gen = own_cpu() - before;
    let kind = case.edge_kind();
    let edge_name = match kind {
        EdgeKind::Rising => "rising",
        EdgeKind::Falling => "falling",
        EdgeKind::Both => "both",
    };
    let mut common = vec![
        "--include".to_string(),
        "scope:tb.dut".into(),
        "--per-scope".into(),
        "tb.dut".into(),
        "-d".into(),
        run_order.to_string(),
    ];
    let mut edge_args = vec![
        "--clock".to_string(),
        SynthSpec::CLOCK.into(),
        "--edges".into(),
        edge_name.into(),
    ];
    edge_args.extend(common.clone());
    // The curve of the max |t| is a plot file. Plot only when there is a curve to read.
    let plot = case.leak.is_some() && case.batches > 1 && !case.shuffle;
    edge_args.push(format!("--plot={plot}"));
    let edges = run_tvla(tvla, &data.meta_list, &case_dir.join("edges"), &edge_args)?;
    row.cpu_edges = edges.cpu;
    row.outside_frac = fraction(edges.outside_fraction());
    let planted = case.leak.as_ref().map(|l| SynthSpec::scope_path(l.scope));
    let expected = spec.leaks.first().and_then(|l| spec.leak_sample(l, kind));
    if let (Some(name), Some(sample)) = (&planted, expected) {
        row.rank1 = yes(&edges.best_channel()? == name);
        let t = edges.channel_t(name)?;
        let hits = exceed(t.row(order - 1), T_LIMIT);
        row.hit = yes(hits.contains(&sample));
        row.extra_exceed = hits.iter().filter(|&&s| s != sample).count().to_string();
        row.lower_clear = if order > 1 {
            yes((0..order - 1).all(|o| exceed(t.row(o), T_LIMIT).is_empty()))
        } else {
            String::new()
        };
        row.t_at_leak = format!("{:.2}", t[[order - 1, sample]].abs());
        row.chi2_hit = yes(edges.channel_chi2(name)?[sample] > CHI2_LIMIT);
        row.first_detect = edges
            .first_detection(order, T_LIMIT)
            .map(|n| n.to_string())
            .unwrap_or_default();
    }
    if case.leak.is_none() {
        row.null = Some(null_counts(&edges, run_order)?);
    }
    if case.legacy {
        common.push("--plot=false".into());
        let legacy = run_tvla(tvla, &data.meta_list, &case_dir.join("legacy"), &common);
        match legacy {
            Ok(legacy) => {
                row.cpu_legacy = legacy.cpu;
                row.legacy_kept_frac = fraction(legacy.kept_fraction());
                if let Some(name) = &planted {
                    row.legacy_rank1 = yes(&legacy.best_channel()? == name);
                    let t = legacy.channel_t(name)?;
                    // The sample numbers differ from the edges mode, so any sample counts.
                    row.legacy_hit = yes(!exceed(t.row(order - 1), T_LIMIT).is_empty());
                }
            }
            Err(e) => row.legacy_hit = format!("error: {e}"),
        }
    }
    if case.shuffle {
        let mut args = edge_args.clone();
        args.retain(|a| !a.starts_with("--plot"));
        args.extend([
            "--plot=false".into(),
            "--shuffle-labels".into(),
            (case.seed + 7).to_string(),
        ]);
        let shuffled = run_tvla(tvla, &data.meta_list, &case_dir.join("shuffled"), &args)?;
        row.cpu_shuffle = shuffled.cpu;
        row.null = Some(null_counts(&shuffled, run_order)?);
    }
    if !keep {
        let _ = std::fs::remove_dir_all(case_dir.join("data"));
    }
    Ok(row)
}

const HEADER: [&str; 30] = [
    "case",
    "group",
    "kind",
    "strength",
    "traces",
    "density",
    "order",
    "error",
    "rank1",
    "hit",
    "extra_exceed",
    "lower_clear",
    "chi2_hit",
    "t_at_leak",
    "first_detect_traces",
    "outside_frac",
    "legacy_rank1",
    "legacy_hit",
    "legacy_kept_frac",
    "null_t_tests",
    "null_t_gt4.5",
    "null_t_gt_bonf",
    "null_t_expected",
    "null_chi2_tests",
    "null_chi2_gt5",
    "null_chi2_gt_bonf",
    "null_max_t",
    "cpu_gen_s",
    "cpu_tvla_s",
    "cpu_shuffle_s",
];

fn tsv_line(row: &Row) -> String {
    let n = row.null.clone();
    let count = |f: fn(&Null) -> u64| n.as_ref().map(|n| f(n).to_string()).unwrap_or_default();
    let fields: Vec<String> = vec![
        row.case.clone(),
        row.group.clone(),
        row.kind.clone(),
        row.strength.clone(),
        row.total_traces.to_string(),
        row.density.to_string(),
        row.order.to_string(),
        row.error.clone(),
        row.rank1.clone(),
        row.hit.clone(),
        row.extra_exceed.clone(),
        row.lower_clear.clone(),
        row.chi2_hit.clone(),
        row.t_at_leak.clone(),
        row.first_detect.clone(),
        row.outside_frac.clone(),
        row.legacy_rank1.clone(),
        row.legacy_hit.clone(),
        row.legacy_kept_frac.clone(),
        count(|n| n.t_tests),
        count(|n| n.t_conv),
        count(|n| n.t_bonf),
        n.as_ref()
            .map(|n| format!("{:.4}", n.exp_t_conv))
            .unwrap_or_default(),
        count(|n| n.chi2_tests),
        count(|n| n.chi2_conv),
        count(|n| n.chi2_bonf),
        n.as_ref()
            .map(|n| format!("{:.2}", n.max_t))
            .unwrap_or_default(),
        format!("{:.2}", row.cpu_gen),
        format!("{:.2}", row.cpu_edges + row.cpu_legacy),
        format!("{:.2}", row.cpu_shuffle),
    ];
    fields.join("\t")
}

/// A short markdown summary of all rows.
fn summary(rows: &[Row], total_cpu: f64) -> String {
    let mut md = String::from("# synth_eval summary\n\n");
    let _ = writeln!(
        md,
        "Cases: {}. CPU time of the whole run: {total_cpu:.1} s.\n",
        rows.len()
    );
    let errors: Vec<&Row> = rows.iter().filter(|r| !r.error.is_empty()).collect();
    for r in &errors {
        let _ = writeln!(md, "- ERROR {}: {}", r.case, r.error);
    }
    // Detection: for each kind and strength, the smallest trace count at which the planted
    // channel ranks first and the exceedance is at the planted sample.
    md.push_str("## Detection (group `power`)\n\n");
    md.push_str("Cell: `ok` = the planted scope ranks first and the exceedance of the planted order is at the planted sample (lower orders clear); `-` = missed. In brackets: the trace count at which the max-|t| curve of the whole selection first passes 4.5.\n\n");
    let power: Vec<&Row> = rows.iter().filter(|r| r.group == "power").collect();
    let mut totals: Vec<u64> = power.iter().map(|r| r.total_traces).collect();
    totals.sort_unstable();
    totals.dedup();
    let _ = write!(md, "| leak | order |");
    for t in &totals {
        let _ = write!(md, " {t} |");
    }
    md.push_str("\n|---|---|");
    for _ in &totals {
        md.push_str("---|");
    }
    md.push('\n');
    let mut kinds: BTreeMap<(String, String), Vec<&Row>> = BTreeMap::new();
    for r in &power {
        kinds
            .entry((r.kind.clone(), r.strength.clone()))
            .or_default()
            .push(r);
    }
    for ((kind, strength), rs) in &kinds {
        let _ = write!(md, "| {kind} {strength} | {} |", rs[0].order);
        for t in &totals {
            let cell = rs
                .iter()
                .find(|r| r.total_traces == *t)
                .map_or("n/a".to_string(), |r| {
                    let found = r.rank1 == "yes" && r.hit == "yes" && r.lower_clear != "no";
                    let curve = if r.first_detect.is_empty() {
                        String::new()
                    } else {
                        format!(" [{}]", r.first_detect)
                    };
                    format!("{}{curve}", if found { "ok" } else { "-" })
                });
            let _ = write!(md, " {cell} |");
        }
        md.push('\n');
    }
    md.push_str("\n## Other cases with a leak\n\n");
    md.push_str("`legacy kept`: the share of the toggles that the legacy sampling keeps; `outside`: the share of the toggles outside all bins in edges mode (conservation report); `legacy hit`: the planted channel has an exceedance at any sample.\n\n");
    md.push_str("| case | group | rank1 | hit | extra | lower clear | chi2 hit | t at leak | outside | legacy hit | legacy kept |\n|---|---|---|---|---|---|---|---|---|---|---|\n");
    for r in rows
        .iter()
        .filter(|r| r.group != "power" && r.group != "shuffle" && r.kind != "none")
    {
        let _ = writeln!(
            md,
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            r.case,
            r.group,
            r.rank1,
            r.hit,
            r.extra_exceed,
            r.lower_clear,
            r.chi2_hit,
            r.t_at_leak,
            r.outside_frac,
            r.legacy_hit,
            r.legacy_kept_frac
        );
    }
    md.push_str("\n## False alarms\n\n");
    md.push_str("Counts over all orders and samples of the whole selection. Expected at 4.5: tests x 2 P(Z > 4.5). Expected above the Bonferroni value: at most 1e-5 per family (one family per run).\n\n");
    md.push_str("| group | runs | t tests | t > 4.5 | expected | t > Bonferroni | chi2 tests | chi2 > 5 | expected | chi2 > Bonferroni | largest t |\n|---|---|---|---|---|---|---|---|---|---|---|\n");
    for group in ["null", "shuffle"] {
        let nulls: Vec<&Null> = rows
            .iter()
            .filter(|r| r.group == group)
            .filter_map(|r| r.null.as_ref())
            .collect();
        if nulls.is_empty() {
            continue;
        }
        let sum = |f: fn(&Null) -> u64| nulls.iter().map(|n| f(n)).sum::<u64>();
        let sumf = |f: fn(&Null) -> f64| nulls.iter().map(|n| f(n)).sum::<f64>();
        let _ = writeln!(
            md,
            "| {group} | {} | {} | {} | {:.4} | {} | {} | {} | {:.4} | {} | {:.2} |",
            nulls.len(),
            sum(|n| n.t_tests),
            sum(|n| n.t_conv),
            sumf(|n| n.exp_t_conv),
            sum(|n| n.t_bonf),
            sum(|n| n.chi2_tests),
            sum(|n| n.chi2_conv),
            sumf(|n| n.exp_chi2_conv),
            sum(|n| n.chi2_bonf),
            nulls.iter().map(|n| n.max_t).fold(0.0, f64::max),
        );
    }
    let generator: f64 = rows.iter().map(|r| r.cpu_gen).sum();
    let tvla: f64 = rows
        .iter()
        .map(|r| r.cpu_edges + r.cpu_legacy + r.cpu_shuffle)
        .sum();
    let _ = writeln!(
        md,
        "\nCPU time: generator {generator:.1} s, tvla {tvla:.1} s."
    );
    md
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    let cases = match &args.sweep {
        Some(path) => {
            let text =
                std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?
        }
        None => default_cases(),
    };
    if args.print_sweep {
        println!(
            "{}",
            serde_json::to_string_pretty(&cases).map_err(|e| e.to_string())?
        );
        return Ok(());
    }
    let out_dir = args
        .out_dir
        .ok_or("give the scratch directory as an argument")?;
    // Check before creating anything: the scratch directory must not be in the repository.
    let out_dir = std::path::absolute(&out_dir).map_err(|e| e.to_string())?;
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
    if out_dir.starts_with(repo) {
        return Err("the scratch directory must not be inside the repository".into());
    }
    std::fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;
    let start = (own_cpu(), children_cpu());
    let mut rows = Vec::new();
    for case in &cases {
        let mut row = match run_case(case, &args.tvla, &out_dir, args.keep) {
            Ok(row) => row,
            Err(e) => Row {
                case: case.name.clone(),
                group: case.group.clone(),
                error: e,
                ..Row::default()
            },
        };
        if !row.error.is_empty() {
            row.kind = case
                .leak
                .as_ref()
                .map_or("none", |l| l.kind.as_str())
                .into();
        }
        eprintln!("{} {}", row.case, row.error);
        rows.push(row);
    }
    let total_cpu = own_cpu() - start.0 + children_cpu() - start.1;
    let mut tsv = HEADER.join("\t");
    for row in &rows {
        tsv.push('\n');
        tsv.push_str(&tsv_line(row));
    }
    tsv.push('\n');
    std::fs::write(out_dir.join("synth_eval.tsv"), tsv).map_err(|e| e.to_string())?;
    let md = summary(&rows, total_cpu);
    std::fs::write(out_dir.join("synth_eval.md"), &md).map_err(|e| e.to_string())?;
    println!("{md}");
    Ok(())
}
