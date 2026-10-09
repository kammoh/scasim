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

/// An FST file with a scope named `a.b` (an escaped identifier) next to the scope `a` that
/// contains the scope `b`. Handle 0 is `x` in the scope `a.b`, handle 1 is `v` in `a`, and handle
/// 2 is `y` in `a` then `b`. A variable `top` outside every scope is handle 3.
fn dotted_scope_file(dir: &Path) -> std::path::PathBuf {
    use fst_writer::*;
    let path = dir.join("dotted.fst");
    let info = FstInfo {
        start_time: 0,
        timescale_exponent: -12,
        version: "test".into(),
        date: "2026-10-08".into(),
        file_type: FstFileType::Verilog,
    };
    let mut header = open_fst(&path, &info).unwrap();
    let var = |header: &mut FstHeaderWriter<_>, name: &str| {
        header
            .var(
                name,
                FstSignalType::bit_vec(1),
                FstVarType::Wire,
                FstVarDirection::Implicit,
                None,
            )
            .unwrap();
    };
    header.scope("a.b", "", FstScopeType::Module).unwrap();
    var(&mut header, "x");
    header.up_scope().unwrap();
    header.scope("a", "", FstScopeType::Module).unwrap();
    var(&mut header, "v");
    header.scope("b", "", FstScopeType::Module).unwrap();
    var(&mut header, "y");
    header.up_scope().unwrap();
    header.up_scope().unwrap();
    var(&mut header, "top");
    let mut body = header.finish().unwrap();
    body.time_change(0).unwrap();
    body.finish().unwrap();
    path
}

#[test]
fn a_dotted_scope_name_stays_whole_and_scope_rules_respect_it() {
    let dir = tempfile::tempdir().unwrap();
    let path = dotted_scope_file(dir.path());
    let fst = fst_index(&path);
    let wellen = wellen_index(&path);
    for index in [&fst, &wellen] {
        let names = |h: usize| index.paths[h][0].scope_names.clone();
        assert_eq!(names(0), ["a.b"]);
        assert_eq!(names(1), ["a"]);
        assert_eq!(names(2), ["a", "b"]);
        assert!(names(3).is_empty());
        assert_eq!(sorted_paths(index, 0), ["a.b.x"]);
        assert_eq!(sorted_paths(index, 3), ["top"]);
        // `scope:a` does not select the signals of the scope `a.b`.
        assert_eq!(select(index, &["+scope:a"]), [false, true, true, false]);
        // The text `a.b` names the literal scope and the nested scope.
        assert_eq!(select(index, &["+scope:a.b"]), [true, false, true, false]);
        assert_eq!(select(index, &["-scope:a.b"]), [false, true, false, true]);
        assert_eq!(
            select(index, &["+signal:a.b.x"]),
            [true, false, false, false]
        );
    }
    for h in 0..4 {
        assert_eq!(fst.paths[h], wellen.paths[h]);
    }
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

/// A vector that the dumper writes bit by bit as `d [0]`, `d [1]`, and `d [2]`, and the same
/// bits again in a second scope. `wellen` merges adjacent bits with the same name into one
/// variable that is derived from the bit signals. Handles 0 to 2 are the bits, and handle 3 is
/// an ordinary signal.
fn bit_by_bit_fixture() -> Fixture {
    let sig = |name: &str, width| FixtureSignal {
        scope: "tb".into(),
        name: name.into(),
        width,
    };
    let mut fx = Fixture::flat(&[]);
    fx.signals = vec![
        sig("d [0]", 1),
        sig("d [1]", 1),
        sig("d [2]", 1),
        sig("e", 4),
    ];
    fx.aliases = (0..3)
        .map(|bit| ("tb.u".to_string(), format!("d [{bit}]"), bit))
        .collect();
    fx.initial = vec!["0".into(), "0".into(), "0".into(), "0000".into()];
    fx
}

/// The input handle indices of every derived signal that `wellen` reports for the file.
fn derived_inputs(path: &Path) -> Vec<Vec<usize>> {
    let header =
        wellen::viewers::read_header_from_file(path, &wellen::LoadOptions::default()).unwrap();
    header
        .hierarchy
        .all_derived_signals()
        .map(|(_, derived)| {
            let mut inputs: Vec<usize> = derived.inputs().iter().map(|i| i.index()).collect();
            inputs.sort();
            inputs
        })
        .collect()
}

/// Checks the index that `from_wellen` builds for the file of `bit_by_bit_fixture`.
fn check_merged_vector_paths(path: &Path) {
    // The test is only useful if `wellen` merges the bits into one derived signal.
    assert_eq!(derived_inputs(path), vec![vec![0, 1, 2]]);
    let index = wellen_index(path);
    // The derived signal has a handle after the four real handles. No path may use it.
    assert_eq!(index.paths.len(), 4);
    // Every bit gets the paths of the merged variable in both scopes.
    for bit in 0..3 {
        assert_eq!(
            sorted_paths(&index, bit),
            vec!["tb.d", "tb.u.d"],
            "bit {bit}"
        );
        assert!(index.paths[bit].iter().all(|p| !p.is_alias));
    }
    assert_eq!(sorted_paths(&index, 3), vec!["tb.e"]);
    // A rule that names the merged vector selects all of its bits.
    assert_eq!(
        select(&index, &["+signal:tb.u.d"]),
        vec![true, true, true, false]
    );
    assert_eq!(
        select(&index, &["-signal:tb.d"]),
        vec![false, false, false, true]
    );
}

#[test]
fn wellen_gives_every_bit_of_a_merged_vector_the_paths_of_the_vector() {
    let (_dir, path) = temp_fst(&bit_by_bit_fixture());
    check_merged_vector_paths(&path);
    // The FST index lists the same paths for every bit handle.
    let fst = fst_index(&path);
    let wellen = wellen_index(&path);
    assert_eq!(wellen.paths.len(), fst.paths.len());
    for h in 0..fst.paths.len() {
        let key = |p: &SignalPath| (p.path.clone(), p.scope.clone(), p.modules.clone());
        let mut from_fst: Vec<_> = fst.paths[h].iter().map(key).collect();
        let mut from_wellen: Vec<_> = wellen.paths[h].iter().map(key).collect();
        from_fst.sort();
        from_wellen.sort();
        assert_eq!(from_wellen, from_fst, "handle {h}");
    }
}

#[test]
fn wellen_maps_a_merged_vector_in_a_vcd_file_to_its_bit_handles() {
    let (_dir, path) = temp_vcd(&bit_by_bit_fixture());
    check_merged_vector_paths(&path);
}

#[test]
fn derived_signals_of_simulator_files_map_to_the_same_input_handles_as_fst() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("fst-reader/fsts");
    for name in [
        "questa-sim/dump.vcd.fst",
        "riviera-pro/dump.vcd.fst",
        "vcs/processor.vcd.fst",
    ] {
        let path = root.join(name);
        let header =
            wellen::viewers::read_header_from_file(&path, &wellen::LoadOptions::default()).unwrap();
        let derived: Vec<_> = header.hierarchy.all_derived_signals().collect();
        assert!(!derived.is_empty(), "{name} has no derived signal");
        let fst = fst_index(&path);
        let wellen = HierarchyIndex::from_wellen(&header.hierarchy);
        // The handles of derived signals come after the real handles. No path may use them.
        assert!(wellen.paths.len() <= fst.paths.len(), "{name}");
        for (_, d) in derived {
            for input in d.inputs() {
                let h = input.index();
                let key = |p: &SignalPath| (p.path.clone(), p.scope.clone(), p.modules.clone());
                let mut from_fst: Vec<_> = fst.paths[h].iter().map(key).collect();
                let mut from_wellen: Vec<_> = wellen.paths[h].iter().map(key).collect();
                from_fst.sort();
                from_wellen.sort();
                assert!(!from_fst.is_empty(), "{name}: handle {h} has no path");
                assert_eq!(from_wellen, from_fst, "{name}: handle {h}");
            }
        }
    }
}
