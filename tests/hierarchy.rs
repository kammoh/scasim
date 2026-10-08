mod common;

use common::*;
use scasim::hierarchy::*;
use std::io::BufReader;
use std::path::Path;

fn fixture() -> Fixture {
    let sig = |scope: &str, name: &str, width| FixtureSignal {
        scope: scope.into(),
        name: name.into(),
        width,
    };
    let mut fx = Fixture::flat(&[]);
    fx.signals = vec![
        sig("tb", "clk", 1),
        sig("tb.dut", "data [3:0]", 4),
        sig("tb.dut.u_a", "sum", 8),
        sig("tb.dut.u_b", "sum", 8),
    ];
    fx.modules = vec![
        ("tb.dut.u_a".into(), "adder".into()),
        ("tb.dut.u_b".into(), "adder".into()),
    ];
    fx.aliases = vec![
        ("tb.dut".into(), "clk".into(), 0),
        ("tb.dut.u_a".into(), "ck".into(), 0),
    ];
    fx.initial = vec!["0".into(), "0000".into(), "0".repeat(8), "0".repeat(8)];
    fx
}

fn fst_index(path: &Path) -> HierarchyIndex {
    let mut reader =
        fst_reader::FstReader::open(BufReader::new(std::fs::File::open(path).unwrap())).unwrap();
    HierarchyIndex::from_fst(&mut reader).unwrap()
}

fn wellen_index(path: &Path) -> HierarchyIndex {
    let header =
        wellen::viewers::read_header_from_file(path, &wellen::LoadOptions::default()).unwrap();
    HierarchyIndex::from_wellen(&header.hierarchy)
}

fn sorted_paths(index: &HierarchyIndex, handle: usize) -> Vec<&str> {
    let mut p: Vec<&str> = index.paths[handle]
        .iter()
        .map(|p| p.path.as_str())
        .collect();
    p.sort();
    p
}

fn select(index: &HierarchyIndex, rules: &[&str]) -> Vec<bool> {
    Selection::parse(rules)
        .unwrap()
        .resolve(index)
        .unwrap()
        .selected
}

#[test]
fn fst_index_lists_every_alias_and_strips_bit_ranges() {
    let (_dir, path) = temp_fst(&fixture());
    let index = fst_index(&path);
    assert!(index.has_module_names);
    assert_eq!(index.paths.len(), 4);
    assert_eq!(
        sorted_paths(&index, 0),
        vec!["tb.clk", "tb.dut.clk", "tb.dut.u_a.ck"]
    );
    assert_eq!(sorted_paths(&index, 1), vec!["tb.dut.data"]);
    assert_eq!(
        index.paths[2][0].modules,
        vec!["".to_string(), "".into(), "adder".into()]
    );
    assert_eq!(index.paths[2][0].scope, "tb.dut.u_a");
}

#[test]
fn fst_index_marks_aliases() {
    let (_dir, path) = temp_fst(&fixture());
    let index = fst_index(&path);
    let mut flags: Vec<(&str, bool)> = index.paths[0]
        .iter()
        .map(|p| (p.path.as_str(), p.is_alias))
        .collect();
    flags.sort();
    assert_eq!(
        flags,
        vec![
            ("tb.clk", false),
            ("tb.dut.clk", true),
            ("tb.dut.u_a.ck", true)
        ]
    );
    assert!(index.paths[1..].iter().flatten().all(|p| !p.is_alias));
}

#[test]
fn fst_index_skips_events_strings_and_reals() {
    use fst_writer::*;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("types.fst");
    let info = FstInfo {
        start_time: 0,
        timescale_exponent: -12,
        version: "test".into(),
        date: "2026-10-08".into(),
        file_type: FstFileType::Verilog,
    };
    let mut header = open_fst(&path, &info).unwrap();
    header.scope("tb", "", FstScopeType::Module).unwrap();
    let mut var = |name: &str, signal: FstSignalType, tpe| {
        header
            .var(name, signal, tpe, FstVarDirection::Implicit, None)
            .unwrap();
    };
    var("bits", FstSignalType::bit_vec(4), FstVarType::Wire);
    var("ev", FstSignalType::bit_vec(1), FstVarType::Event);
    var("name", FstSignalType::bit_vec(1), FstVarType::GenericString);
    var("r", FstSignalType::real(), FstVarType::Real);
    var("rt", FstSignalType::real(), FstVarType::RealTime);
    var("sr", FstSignalType::real(), FstVarType::ShortReal);
    // A VHDL array with a null range has the width 0.
    var("empty", FstSignalType::bit_vec(0), FstVarType::Logic);
    var("last", FstSignalType::bit_vec(1), FstVarType::Logic);
    header.up_scope().unwrap();
    let mut body = header.finish().unwrap();
    body.time_change(0).unwrap();
    body.finish().unwrap();

    let index = fst_index(&path);
    let with_path: Vec<usize> = (0..index.paths.len())
        .filter(|&h| !index.paths[h].is_empty())
        .collect();
    assert_eq!(with_path, vec![0, 7]);
    assert_eq!(sorted_paths(&index, 0), vec!["tb.bits"]);
    assert_eq!(sorted_paths(&index, 7), vec!["tb.last"]);
    // Handles without a path are never selected, whatever the rules say.
    assert_eq!(
        select(&index, &["-signal:tb.bits"]),
        vec![false, false, false, false, false, false, false, true]
    );
}

#[test]
fn selection_on_a_real_file() {
    let (_dir, path) = temp_fst(&fixture());
    let index = fst_index(&path);
    // The clock is selected through its alias inside u_a.
    assert_eq!(
        select(&index, &["+module:adder"]),
        vec![true, false, true, true]
    );
    assert_eq!(
        select(&index, &["+module:adder", "-signal:tb.clk"]),
        vec![false, false, true, true]
    );
    assert_eq!(
        select(&index, &["-scope:tb.dut.u_a", "+signal:tb.dut.u_a.sum"]),
        vec![false, true, true, true]
    );
}

#[test]
fn wellen_index_matches_the_fst_index_for_simple_names() {
    let (_dir, path) = temp_fst(&fixture());
    let fst = fst_index(&path);
    let wellen = wellen_index(&path);
    assert!(wellen.has_module_names);
    assert_eq!(wellen.paths.len(), fst.paths.len());
    for h in 0..fst.paths.len() {
        let key = |p: &SignalPath| (p.path.clone(), p.scope.clone(), p.modules.clone());
        let mut from_fst: Vec<_> = fst.paths[h].iter().map(key).collect();
        let mut from_wellen: Vec<_> = wellen.paths[h].iter().map(key).collect();
        from_fst.sort();
        from_wellen.sort();
        assert_eq!(from_wellen, from_fst, "handle {h}");
        assert!(wellen.paths[h].iter().all(|p| !p.is_alias));
    }
}

#[test]
fn vcd_files_have_no_module_names() {
    let mut fx = fixture();
    fx.modules.clear();
    fx.initial = vec!["0".into(), "0000".into(), "0".repeat(8), "0".repeat(8)];
    let (_dir, path) = temp_vcd(&fx);
    let index = wellen_index(&path);
    assert!(!index.has_module_names);
    assert!(index.paths.iter().all(|p| !p.is_empty()));
    assert_eq!(
        sorted_paths(&index, 0),
        vec!["tb.clk", "tb.dut.clk", "tb.dut.u_a.ck"]
    );
    assert_eq!(sorted_paths(&index, 1), vec!["tb.dut.data"]);
    let rules = Selection::parse(&["+module:x"]).unwrap();
    assert!(matches!(
        rules.resolve(&index),
        Err(SelectionError::NoModuleNames)
    ));
}

#[test]
fn vcd_scope_names_with_dots_stay_whole() {
    let vcd = "$timescale 1ps $end\n\
               $var wire 1 ! top_sig $end\n\
               $scope module tb $end\n\
               $scope module u.x $end\n\
               $var wire 1 \" s $end\n\
               $var wire 1 # a.b $end\n\
               $upscope $end\n\
               $upscope $end\n\
               $enddefinitions $end\n\
               #0\n$dumpvars\n0!\n0\"\n0#\n$end\n";
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dots.vcd");
    std::fs::write(&path, vcd).unwrap();
    let index = wellen_index(&path);
    let mut all: Vec<_> = index
        .paths
        .iter()
        .flatten()
        .map(|p| (p.path.as_str(), p.scope.as_str(), p.modules.len()))
        .collect();
    all.sort();
    assert_eq!(
        all,
        vec![
            ("tb.u.x.a.b", "tb.u.x", 2),
            ("tb.u.x.s", "tb.u.x", 2),
            ("top_sig", "", 0),
        ]
    );
    // The scope rule uses the whole scope name.
    let selected = select(&index, &["+scope:tb.u.x"]);
    assert_eq!(selected.iter().filter(|&&s| s).count(), 2);
}

#[test]
fn vcd_variables_without_bit_vector_values_get_no_path() {
    let vcd = "$timescale 1ps $end\n\
               $scope module tb $end\n\
               $var wire 4 ! bits $end\n\
               $var real 64 \" r $end\n\
               $var event 1 # e $end\n\
               $var wire 1 $ last $end\n\
               $upscope $end\n\
               $enddefinitions $end\n\
               #0\n$dumpvars\nb0000 !\nr0 \"\n0$\n$end\n#5\nb0001 !\nr1.5 \"\n1#\n1$\n";
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("types.vcd");
    std::fs::write(&path, vcd).unwrap();
    let index = wellen_index(&path);
    let names: Vec<_> = index
        .paths
        .iter()
        .flatten()
        .map(|p| p.path.as_str())
        .collect();
    assert_eq!(names.len(), 2, "{names:?}");
    assert!(names.contains(&"tb.bits") && names.contains(&"tb.last"));
}
