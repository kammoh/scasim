use ndarray::Array2;
use ndarray_npz::NpzReader;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

struct Batch {
    dir: tempfile::TempDir,
}
impl Batch {
    fn new(id: &str, v1: bool) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let mut waveform = String::from(
            "$timescale 1ps $end\n$scope module tb $end\n$var wire 1 ! clk $end\n$scope module unit $end\n$var wire 4 \" data $end\n$upscope $end\n$upscope $end\n$enddefinitions $end\n#0\n0!\nb0000 \"\n",
        );
        let mut segments = Vec::new();
        let mut markers = Vec::new();
        for i in 0..12u64 {
            let label = if v1 { (i % 3) as u16 } else { (i % 2) as u16 };
            for j in 0..2 {
                let t = 10 + i * 20 + j * 10;
                waveform.push_str(&format!(
                    "#{t}\n1!\nb{:04b} \"\n#{}\n0!\n",
                    (i * 3 + j + u64::from(label)) % 16,
                    t + 5
                ));
            }
            segments.push(
                serde_json::json!({"id":i,"start":10+i*20,"end":30+i*20,"label":label,"group":i/6}),
            );
            markers.push(serde_json::json!([10 + i * 20, 30 + i * 20, label]));
        }
        waveform.push_str("#250\n1!\n");
        std::fs::write(dir.path().join("w.vcd"), waveform).unwrap();
        let json = if v1 {
            serde_json::json!({"scasim_meta":1,"waveform":"w.vcd","time":{"mantissa":1,"exponent":-12},"batch":{"id":id,"status":"committed","seeds":{"base":17},"design_random":{"requested":"off","applied":"off","how":"hook"}},"segments":segments,"labels":{"0":"a","1":"b","2":"c"},"groups":{"0":"g0","1":"g1"},"extensions":{}})
        } else {
            serde_json::json!({"trace_filename":"w.vcd","clock_period":10,"markers":markers})
        };
        std::fs::write(
            dir.path().join("meta.json"),
            serde_json::to_vec(&json).unwrap(),
        )
        .unwrap();
        Self { dir }
    }
    fn meta(&self) -> PathBuf {
        self.dir.path().join("meta.json")
    }
    fn run(&self, flags: &[&str]) -> Output {
        run(
            &["--meta-json", self.meta().to_str().unwrap()],
            &self.dir.path().join("out"),
            flags,
        )
    }
}
fn run(input: &[&str], out: &Path, flags: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_tvla"))
        .args(input)
        .args([
            "--use-existing=false",
            "--num-threads",
            "2",
            "-d",
            "2",
            "--ttest-output-dir",
        ])
        .arg(out)
        .args(if flags.iter().any(|s| s.starts_with("--plot")) {
            vec![]
        } else {
            vec!["--plot=false"]
        })
        .args(flags)
        .output()
        .unwrap()
}
fn ok(output: Output) {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
fn bits(path: &Path, name: &str) -> Vec<u64> {
    let mut npz = NpzReader::new(std::fs::File::open(path).unwrap()).unwrap();
    let a: Array2<f64> = npz.by_name(name).unwrap();
    a.iter().map(|x| x.to_bits()).collect()
}
#[test]
fn select_three_labels_and_two_groups() {
    let b = Batch::new("b0", true);
    let fail = b.run(&["--clock", "tb.clk"]);
    assert!(!fail.status.success());
    let err = String::from_utf8_lossy(&fail.stderr);
    assert!(
        err.contains("--group") && err.contains("--pool-groups"),
        "{err}"
    );
    ok(b.run(&["--clock", "tb.clk", "--pair", "1", "2", "--group", "0"]));
    let first = bits(&b.dir.path().join("out/t_values.npz"), "t_values");
    ok(b.run(&["--clock", "tb.clk", "--pair", "1", "2", "--pool-groups"]));
    assert_ne!(
        first,
        bits(&b.dir.path().join("out/t_values.npz"), "t_values")
    );
    assert!(
        !b.run(&["--clock", "tb.clk", "--pair", "2", "2", "--group", "0"])
            .status
            .success()
    );
}
#[test]
fn caches_match_normal_run_and_reject_duplicate_and_corruption() {
    for (v1, edges, scopes) in [
        (false, false, false),
        (false, true, true),
        (true, true, true),
    ] {
        let a = Batch::new("a", v1);
        let b = Batch::new("b", v1);
        let flags: Vec<&str> = if edges {
            vec![
                "--clock",
                "tb.clk",
                "--length-policy",
                "pad",
                "--per-scope",
                "tb",
                "--pool-groups",
            ]
        } else {
            vec![]
        };
        let mut flags = flags;
        if !scopes {
            flags.retain(|f| *f != "--per-scope");
        }
        let list = a.dir.path().join("meta.list");
        std::fs::write(
            &list,
            format!("{}\n{}\n", a.meta().display(), b.meta().display()),
        )
        .unwrap();
        let normal = a.dir.path().join("normal");
        ok(run(
            &["--meta-list", list.to_str().unwrap()],
            &normal,
            &flags,
        ));
        let ca = a.dir.path().join("a.stats");
        let cb = b.dir.path().join("b.stats");
        for (batch, path) in [(&a, &ca), (&b, &cb)] {
            let mut f = flags.clone();
            f.extend(["--stats-out", path.to_str().unwrap()]);
            ok(batch.run(&f));
        }
        let merged = a.dir.path().join("merged");
        ok(run(
            &["--merge-stats", ca.to_str().unwrap(), cb.to_str().unwrap()],
            &merged,
            if v1 { &["--pool-groups"] } else { &[] },
        ));
        assert_eq!(
            bits(&normal.join("t_values.npz"), "t_values"),
            bits(&merged.join("t_values.npz"), "t_values")
        );
        for name in ["statistic", "neg_log10_p", "min_expected"] {
            let read = |path: &Path| {
                let mut npz = NpzReader::new(std::fs::File::open(path).unwrap()).unwrap();
                let a: ndarray::Array1<f64> = npz.by_name(name).unwrap();
                a.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            };
            assert_eq!(
                read(&normal.join("chi2.npz")),
                read(&merged.join("chi2.npz"))
            );
        }
        for name in ["dof", "columns", "merged"] {
            let read = |path: &Path| {
                let mut npz = NpzReader::new(std::fs::File::open(path).unwrap()).unwrap();
                let values: ndarray::Array1<u32> = npz.by_name(name).unwrap();
                values.to_vec()
            };
            assert_eq!(
                read(&normal.join("chi2.npz")),
                read(&merged.join("chi2.npz"))
            );
        }
        let read_n = |path: &Path| {
            let mut npz = NpzReader::new(std::fs::File::open(path).unwrap()).unwrap();
            let values: ndarray::Array1<u64> = npz.by_name("n").unwrap();
            values.to_vec()
        };
        assert_eq!(
            read_n(&normal.join("chi2.npz")),
            read_n(&merged.join("chi2.npz"))
        );
        if scopes {
            for filename in ["t_values_channels.npz", "chi2_channels.npz"] {
                let mut left =
                    NpzReader::new(std::fs::File::open(normal.join(filename)).unwrap()).unwrap();
                let mut right =
                    NpzReader::new(std::fs::File::open(merged.join(filename)).unwrap()).unwrap();
                let names = left.names().unwrap();
                assert_eq!(names, right.names().unwrap());
                for name in names {
                    let l: ndarray::ArrayD<f64> = left.by_name(&name).unwrap();
                    let r: ndarray::ArrayD<f64> = right.by_name(&name).unwrap();
                    assert_eq!(l.shape(), r.shape());
                    assert_eq!(
                        l.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        r.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                    );
                }
            }
        }
        assert!(
            !run(
                &["--merge-stats", ca.to_str().unwrap(), ca.to_str().unwrap()],
                &merged,
                &[]
            )
            .status
            .success()
        );
        std::fs::write(&ca, b"corrupt").unwrap();
        let failed = run(&["--merge-stats", ca.to_str().unwrap()], &merged, &[]);
        assert!(!failed.status.success());
        assert!(!String::from_utf8_lossy(&failed.stderr).contains("panicked"));
    }
}
#[test]
fn curve_final_preserves_final_results() {
    let b = Batch::new("b", false);
    ok(b.run(&[]));
    let want = bits(&b.dir.path().join("out/t_values.npz"), "t_values");
    ok(b.run(&["--curve", "final"]));
    assert_eq!(
        want,
        bits(&b.dir.path().join("out/t_values.npz"), "t_values")
    );
    assert!(!b.run(&["--curve", "every:0"]).status.success());
}

#[test]
fn changed_seed_segment_and_preprocessing_change_the_cache_key() {
    let b = Batch::new("b", true);
    let out = b.dir.path().join("batch.stats");
    let flags = [
        "--clock",
        "tb.clk",
        "--pool-groups",
        "--stats-out",
        out.to_str().unwrap(),
    ];
    ok(b.run(&flags));
    let original = std::fs::read(&out).unwrap();
    let mut meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(b.meta()).unwrap()).unwrap();
    meta["batch"]["seeds"]["base"] = 18.into();
    std::fs::write(b.meta(), serde_json::to_vec(&meta).unwrap()).unwrap();
    ok(b.run(&flags));
    assert_ne!(original, std::fs::read(&out).unwrap());
    meta["batch"]["seeds"]["base"] = 17.into();
    meta["segments"][0]["id"] = 99.into();
    std::fs::write(b.meta(), serde_json::to_vec(&meta).unwrap()).unwrap();
    ok(b.run(&flags));
    assert_ne!(original, std::fs::read(&out).unwrap());
    meta["segments"][0]["id"] = 0.into();
    std::fs::write(b.meta(), serde_json::to_vec(&meta).unwrap()).unwrap();
    let mut changed = flags.to_vec();
    changed.extend(["--exclude", "signal:tb.unit.data"]);
    ok(b.run(&changed));
    assert_ne!(original, std::fs::read(&out).unwrap());
    let mut choice = flags.to_vec();
    choice.extend(["--pair", "1", "2"]);
    ok(b.run(&choice));
    assert_eq!(original, std::fs::read(&out).unwrap());
}

#[test]
fn curve_every_k_records_the_last_partial_checkpoint() {
    let b = Batch::new("b", false);
    let list = b.dir.path().join("meta.list");
    std::fs::write(&list, format!("{}\n", b.meta().display()).repeat(5)).unwrap();
    let out = b.dir.path().join("spaced");
    ok(run(
        &["--meta-list", list.to_str().unwrap()],
        &out,
        &["--curve", "every:2"],
    ));
    let curve = std::fs::read_to_string(out.join("curves.tsv")).unwrap();
    let rows: Vec<_> = curve.lines().filter(|l| !l.starts_with('#')).collect();
    assert_eq!(rows.len(), 5); // Header, origin, batches 2 and 4, and the final batch.
    let counts: Vec<_> = rows
        .iter()
        .skip(1)
        .map(|l| l.split('\t').next().unwrap())
        .collect();
    assert_eq!(counts, ["0", "24", "48", "60"]);
    let final_out = b.dir.path().join("final");
    ok(run(
        &["--meta-list", list.to_str().unwrap()],
        &final_out,
        &["--curve", "final", "--plot=true"],
    ));
    assert!(!final_out.join("curves.tsv").exists());
    assert!(!final_out.join("max_t_values.html").exists());
    assert!(!final_out.join("max_chi2.html").exists());
}

#[test]
fn merge_rejects_preprocessing_flags_that_cannot_change_cached_traces() {
    let b = Batch::new("b", false);
    let cache = b.dir.path().join("batch.stats");
    ok(b.run(&["--stats-out", cache.to_str().unwrap()]));
    let result = run(
        &["--merge-stats", cache.to_str().unwrap()],
        &b.dir.path().join("merged"),
        &["--include", "scope:tb"],
    );
    assert!(!result.status.success());
}

#[test]
fn unequal_cache_lengths_follow_the_saved_policy() {
    for edges in [false, true] {
        for policy in ["pad", "truncate", "error"] {
            let a = Batch::new("a", false);
            let b = Batch::new("b", false);
            let mut meta: serde_json::Value =
                serde_json::from_slice(&std::fs::read(b.meta()).unwrap()).unwrap();
            for marker in meta["markers"].as_array_mut().unwrap() {
                marker[1] = (marker[0].as_u64().unwrap() + 10).into();
            }
            std::fs::write(b.meta(), serde_json::to_vec(&meta).unwrap()).unwrap();
            let mut flags = vec!["--length-policy", policy];
            if edges {
                flags.extend(["--clock", "tb.clk"]);
            }
            let list = a.dir.path().join("meta.list");
            std::fs::write(
                &list,
                format!("{}\n{}\n", a.meta().display(), b.meta().display()),
            )
            .unwrap();
            let normal = a.dir.path().join("normal");
            let normal_result = run(&["--meta-list", list.to_str().unwrap()], &normal, &flags);
            let ca = a.dir.path().join("a.stats");
            let cb = b.dir.path().join("b.stats");
            for (batch, path) in [(&a, &ca), (&b, &cb)] {
                let mut f = flags.clone();
                f.extend(["--stats-out", path.to_str().unwrap()]);
                ok(batch.run(&f));
            }
            let merged = a.dir.path().join("merged");
            let merge_result = run(
                &["--merge-stats", ca.to_str().unwrap(), cb.to_str().unwrap()],
                &merged,
                &[],
            );
            if policy == "error" {
                assert!(!normal_result.status.success());
                assert!(!merge_result.status.success());
            } else {
                ok(normal_result);
                ok(merge_result);
                assert_eq!(
                    bits(&normal.join("t_values.npz"), "t_values"),
                    bits(&merged.join("t_values.npz"), "t_values")
                );
            }
        }
    }
}

#[test]
fn shuffled_separate_caches_match_batch_list_order_and_group_counts() {
    let a = Batch::new("a", true);
    let b = Batch::new("b", true);
    let flags = [
        "--clock",
        "tb.clk",
        "--shuffle-labels",
        "19",
        "--pool-groups",
        "--pair",
        "1",
        "2",
    ];
    let list = a.dir.path().join("meta.list");
    std::fs::write(
        &list,
        format!("{}\n{}\n", a.meta().display(), b.meta().display()),
    )
    .unwrap();
    let normal = a.dir.path().join("normal");
    ok(run(
        &["--meta-list", list.to_str().unwrap()],
        &normal,
        &flags,
    ));
    let ca = a.dir.path().join("a.stats");
    let cb = b.dir.path().join("b.stats");
    for (batch, path) in [(&a, &ca), (&b, &cb)] {
        let mut f = flags.to_vec();
        f.extend(["--stats-out", path.to_str().unwrap()]);
        ok(batch.run(&f));
    }
    let merged = a.dir.path().join("merged");
    ok(run(
        &["--merge-stats", ca.to_str().unwrap(), cb.to_str().unwrap()],
        &merged,
        &["--pool-groups", "--pair", "1", "2"],
    ));
    assert_eq!(
        bits(&normal.join("t_values.npz"), "t_values"),
        bits(&merged.join("t_values.npz"), "t_values")
    );
    assert_eq!(
        std::fs::read(normal.join("curves.tsv")).unwrap(),
        std::fs::read(merged.join("curves.tsv")).unwrap()
    );
}
