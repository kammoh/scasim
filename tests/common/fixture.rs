//! Fixtures: write FST and VCD files with known content. No code from `src/` is used.

use fst_writer::*;
use std::path::{Path, PathBuf};

/// One signal of a fixture.
#[derive(Debug, Clone)]
pub struct FixtureSignal {
    /// Scope path, for example `tb.dut`. Signals must be sorted by scope.
    pub scope: String,
    pub name: String,
    pub width: u32,
}

/// One test file: signals, aliases, value changes per time step, and section boundaries.
#[derive(Debug, Clone)]
pub struct Fixture {
    pub signals: Vec<FixtureSignal>,
    /// Module (component) name per scope path; scopes not listed get an empty name. Only FST
    /// files can store module names.
    pub modules: Vec<(String, String)>,
    /// Extra variables (scope, name, index of the aliased signal). Written after all signals.
    pub aliases: Vec<(String, String, usize)>,
    /// Value of each signal before the first time step (ASCII state characters).
    pub initial: Vec<String>,
    /// Strictly increasing times (all > 0), each with (signal index, new value) changes. The
    /// changes of a step are written in order, so one signal can change several times in a
    /// step (a glitch).
    pub steps: Vec<(u64, Vec<(usize, String)>)>,
    /// Indices into `steps` before which a new section starts. Only FST files have sections.
    pub flush_before: Vec<usize>,
    /// The timescale is 10^`timescale_exponent` seconds.
    pub timescale_exponent: i8,
}

impl Fixture {
    /// Signals in scope `tb` with the given widths, named `s0`, `s1`, ...
    pub fn flat(widths: &[u32]) -> Fixture {
        Fixture {
            signals: widths
                .iter()
                .enumerate()
                .map(|(i, &w)| FixtureSignal {
                    scope: "tb".into(),
                    name: format!("s{i}"),
                    width: w,
                })
                .collect(),
            modules: vec![],
            aliases: vec![],
            initial: widths.iter().map(|&w| "0".repeat(w as usize)).collect(),
            steps: vec![],
            flush_before: vec![],
            timescale_exponent: -12,
        }
    }

    /// The paths of signal `index`: its own path, then the path of each alias.
    fn paths(&self, index: usize) -> Vec<String> {
        let join = |scope: &str, name: &str| {
            if scope.is_empty() {
                name.to_string()
            } else {
                format!("{scope}.{name}")
            }
        };
        let s = &self.signals[index];
        std::iter::once(join(&s.scope, &s.name))
            .chain(
                self.aliases
                    .iter()
                    .filter(|(_, _, target)| *target == index)
                    .map(|(scope, name, _)| join(scope, name)),
            )
            .collect()
    }
}

/// Moves the open scope stack to `target`, closing and opening scopes as needed.
fn enter_scope(
    header: &mut FstHeaderWriter<std::io::BufWriter<std::fs::File>>,
    open: &mut Vec<String>,
    target: &str,
    modules: &[(String, String)],
) {
    let parts: Vec<&str> = if target.is_empty() {
        vec![]
    } else {
        target.split('.').collect()
    };
    let common = open.iter().zip(&parts).take_while(|(a, b)| a == *b).count();
    while open.len() > common {
        header.up_scope().unwrap();
        open.pop();
    }
    for part in &parts[common..] {
        open.push(part.to_string());
        let path = open.join(".");
        let module = modules
            .iter()
            .find(|(s, _)| *s == path)
            .map(|(_, m)| m.as_str())
            .unwrap_or("");
        header.scope(*part, module, FstScopeType::Module).unwrap();
    }
}

pub fn write_fst(path: &Path, fx: &Fixture) {
    let info = FstInfo {
        start_time: 0,
        timescale_exponent: fx.timescale_exponent,
        version: "scasim test".into(),
        date: "2026-10-08".into(),
        file_type: FstFileType::Verilog,
    };
    let mut header = open_fst(path, &info).unwrap();
    let mut open: Vec<String> = Vec::new();
    let mut ids = Vec::new();
    for s in &fx.signals {
        enter_scope(&mut header, &mut open, &s.scope, &fx.modules);
        let id = header
            .var(
                &s.name,
                FstSignalType::bit_vec(s.width),
                FstVarType::Wire,
                FstVarDirection::Implicit,
                None,
            )
            .unwrap();
        ids.push(id);
    }
    for (scope, name, target) in &fx.aliases {
        enter_scope(&mut header, &mut open, scope, &fx.modules);
        header
            .var(
                name,
                FstSignalType::bit_vec(fx.signals[*target].width),
                FstVarType::Wire,
                FstVarDirection::Implicit,
                Some(ids[*target]),
            )
            .unwrap();
    }
    enter_scope(&mut header, &mut open, "", &fx.modules);
    let mut body = header.finish().unwrap();
    for (i, v) in fx.initial.iter().enumerate() {
        body.signal_change(ids[i], v.as_bytes()).unwrap();
    }
    for (k, (time, changes)) in fx.steps.iter().enumerate() {
        if fx.flush_before.contains(&k) {
            body.flush().unwrap();
        }
        body.time_change(*time).unwrap();
        for (sig, v) in changes {
            body.signal_change(ids[*sig], v.as_bytes()).unwrap();
        }
    }
    body.finish().unwrap();
}

/// The VCD timescale text for 10^`exponent` seconds, for example `1 ps` for -12 and `10 ns`
/// for -8.
fn vcd_timescale(exponent: i8) -> String {
    let units = ["fs", "ps", "ns", "us", "ms", "s"];
    let unit_index = (exponent.clamp(-15, 0) + 15) / 3;
    let unit_exponent = unit_index * 3 - 15;
    format!(
        "{} {}",
        10u32.pow((exponent - unit_exponent) as u32),
        units[unit_index as usize]
    )
}

/// The VCD identifier code of variable number `n`: a string of printable characters.
fn vcd_id(mut n: usize) -> String {
    let mut id = String::new();
    loop {
        id.push(char::from(b'!' + (n % 94) as u8));
        n /= 94;
        if n == 0 {
            return id;
        }
    }
}

/// Writes the fixture as a VCD file. VCD has four states, so the values may use only `0`, `1`,
/// `x`, and `z`. The initial values are at time 0. A VCD time table always contains time 0.
pub fn write_vcd(path: &Path, fx: &Fixture) {
    let mut out = format!("$timescale {} $end\n", vcd_timescale(fx.timescale_exponent));
    let mut open: Vec<String> = Vec::new();
    let enter = |out: &mut String, open: &mut Vec<String>, target: &str| {
        let parts: Vec<&str> = if target.is_empty() {
            vec![]
        } else {
            target.split('.').collect()
        };
        let common = open.iter().zip(&parts).take_while(|(a, b)| a == *b).count();
        while open.len() > common {
            out.push_str("$upscope $end\n");
            open.pop();
        }
        for part in &parts[common..] {
            out.push_str(&format!("$scope module {part} $end\n"));
            open.push(part.to_string());
        }
    };
    for (i, s) in fx.signals.iter().enumerate() {
        enter(&mut out, &mut open, &s.scope);
        out.push_str(&format!(
            "$var wire {} {} {} $end\n",
            s.width,
            vcd_id(i),
            s.name
        ));
    }
    for (scope, name, target) in &fx.aliases {
        enter(&mut out, &mut open, scope);
        out.push_str(&format!(
            "$var wire {} {} {} $end\n",
            fx.signals[*target].width,
            vcd_id(*target),
            name
        ));
    }
    enter(&mut out, &mut open, "");
    out.push_str("$enddefinitions $end\n");
    let value = |sig: usize, v: &str| {
        assert!(
            v.bytes().all(|c| b"01xz".contains(&c)),
            "VCD has only the states 0, 1, x, and z"
        );
        if fx.signals[sig].width == 1 {
            format!("{v}{}\n", vcd_id(sig))
        } else {
            format!("b{v} {}\n", vcd_id(sig))
        }
    };
    out.push_str("#0\n$dumpvars\n");
    for (sig, v) in fx.initial.iter().enumerate() {
        out.push_str(&value(sig, v));
    }
    out.push_str("$end\n");
    for (time, changes) in &fx.steps {
        out.push_str(&format!("#{time}\n"));
        for (sig, v) in changes {
            out.push_str(&value(*sig, v));
        }
    }
    std::fs::write(path, out).unwrap();
}

/// A temporary directory with one FST file of the fixture.
pub fn temp_fst(fx: &Fixture) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("f.fst");
    write_fst(&path, fx);
    (dir, path)
}

/// A temporary directory with one VCD file of the fixture.
pub fn temp_vcd(fx: &Fixture) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("f.vcd");
    write_vcd(&path, fx);
    (dir, path)
}

/// Sets the end time in the header block of an FST file. The header block has the block type
/// 0, then the section length, the start time, and the end time (each a big-endian `u64`).
pub fn patch_header_end_time(path: &Path, end_time: u64) {
    let mut bytes = std::fs::read(path).unwrap();
    assert_eq!(bytes[0], 0, "the first block is the header");
    bytes[17..25].copy_from_slice(&end_time.to_be_bytes());
    std::fs::write(path, bytes).unwrap();
}

/// `fst-writer` 0.3.1 stores a zlib-compressed time table as if it were uncompressed when both
/// have the same length, so the reader decodes garbage. Returns false for such files.
pub fn time_table_is_readable(path: &Path, fx: &Fixture) -> bool {
    let want: Vec<u64> = if fx.steps.is_empty() {
        vec![]
    } else {
        std::iter::once(0)
            .chain(fx.steps.iter().map(|s| s.0))
            .collect()
    };
    let file = std::fs::File::open(path).unwrap();
    match fst_reader::FstReader::open_and_read_time_table(std::io::BufReader::new(file)) {
        Ok(reader) => reader.get_time_table() == Some(&want[..]),
        Err(_) => false,
    }
}

/// Every `.fst` file of the `fst-reader` test corpus, sorted.
pub fn corpus_files() -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|e| e == "fst") {
                out.push(path);
            }
        }
    }
    let mut out = Vec::new();
    walk(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("fst-reader/fsts"),
        &mut out,
    );
    out.sort();
    out
}
