# scalib-fixtures

This crate makes `tests/fixtures/stats/scalib_ttest.json`. The file holds deterministic traces
and labels, and the t-values that the SCALib t-test (`scalib::ttest::Ttest`) computed for them.
Tests of the scasim t-test engine read the file. The tests do not need SCALib.

The crate is its own workspace (`[workspace]` in `Cargo.toml`). It is not a member of the scasim
workspace. It uses SCALib as a git dependency (`https://github.com/kammoh/SCALib.git`, commit
`f1e2d9d`), so it keeps working after scasim drops the `deps/SCALib` submodule.
`Cargo.lock` pins the exact commit.

## Run

Set `CARGO_TARGET_DIR` to a directory outside the repository. Then the build files do not appear
in the repository.

```
CARGO_TARGET_DIR=/some/scratch/dir \
  cargo run --release --manifest-path scripts/scalib-fixtures/Cargo.toml [OUTPUT_FILE]
```

The default output file is `tests/fixtures/stats/scalib_ttest.json`. The program is
deterministic: two runs write the same bytes. A run takes less than one second (after the build).

## What the file holds

- `scalib`: the repository and the commit of SCALib.
- `generator_version`: the version of the data recipes, the cases, and the file format.
  Change `GENERATOR_VERSION` in `src/main.rs` when you change any of them.
- `datasets`: name, `dtype` (`u32` or `f32`), a description with the seeds, `labels`, and `traces`.
  Traces are integers. For `f32`, convert each value to `f32` (this is exact).
  The data comes from a SplitMix64 generator with fixed seeds.
- `cases`: name, dataset, order `d`, `ns`, `batch_sizes` (the batches that were passed to
  `update`, in order), and `t_values` (shape `(d, ns)`, classes 0 and 1).
  A number is written with the shortest text that reads back to the same f64.
  A non-finite value is `null`. `non_finite` then lists its `order`, `sample`, and `kind`
  (`"nan"`, `"inf"`, or `"-inf"`).

Cases: u32 traces with orders 1 to 4 and six batch splits (including batches of 7), integer-valued
f32 traces, orders 1 to 3, unbalanced classes with small `ns`, and two degenerate datasets. The
degenerate cases record what SCALib returns for constant samples, zero samples, and a class with
one trace. For these, SCALib gives NaN and infinite values. Do not assume that the new engine must
copy every one of them; check ruling A2 of the A2a design.

SCALib reads floating-point traces as integers (it truncates them). All traces here are integers.

## License

The file holds numbers that SCALib computed. SCALib is licensed under AGPL-3.0. This generator
only calls the public API of SCALib. It does not copy SCALib code.
