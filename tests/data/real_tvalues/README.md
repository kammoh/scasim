# Real t-value oracles

These files hold the t-values that the SCALib-based `tvla` computed for real waveforms. The new
statistics engine must reproduce them. They are the oracles of the ignored tests in
`tests/real_data.rs`.

## How they were made

- Program: `tvla` built from scasim commit `cbceeb5` (SCALib t-test), `cargo build --release --bin tvla`.
- Options: `-d 4 --plot=false`. The six single runs add `--use-existing=false`, so `tvla` parses
  the waveform. The multi run reads the cached `traces.npz` files.
- Every run used copies of the input files in a scratch directory. The source directory
  `/Volumes/src/krystals_hw/cocotb/tvla_run/` is read-only and `tvla` writes `traces.npz` next to
  the metadata file.
- Each oracle is the `t_values` array (f64, shape `(4, ns)`) of the `t_values.npz` that `tvla`
  wrote. Orders 1 to 4 are rows 0 to 3.
- The values are NaN where SCALib divides 0 by 0 or where a class has too few traces. They have
  no infinite value. SCALib gives NaN for 0/0 and +inf or -inf for x/0. The new engine must give a
  non-finite value of the same kind at the same place.

Single runs (one batch each). `R` is `/Volumes/src/krystals_hw/cocotb/tvla_run`:

```
for B in <batch dir below>; do
  cp $R/$B/meta.json.gz $R/$B/tvla.fst $WORK/
  tvla -d 4 --plot=false --use-existing=false --ttest-output-dir $WORK/out --meta-json $WORK/meta.json.gz
done
```

Multi run (50 batches, cache only, no waveform):

```
# $WORK/meta.list has one line per batch: <hash>/meta.json.gz (the file multi50.meta.list)
tvla -d 4 --plot=false --ttest-output-dir $WORK/out --meta-list $WORK/meta.list
```

## Files

Batch directories are relative to `R`. The sha256 values are of the oracle file in this directory.

| Oracle file | Batch directory | Shape | NaN |
|---|---|---|---|
| `QuadR2NttShuffledStall-multi_no_random-058a00d90.npz` | `QuadR2NttShuffledStall_61cb8c6f/ntt_test/multi_no_random/058a00d90` | (4, 371) | 672 |
| `QuadR2NttShuffledStall-multi-245beb1812.npz` | `QuadR2NttShuffledStall_61cb8c6f/ntt_test/multi/245beb1812` | (4, 371) | 74 |
| `KyberR4Ntt-multi_no_random-39003d6087.npz` | `KyberR4Ntt_shuffled_dcc30f1c/ntt_test/multi_no_random/39003d6087` | (4, 414) | 648 |
| `KyberR4Ntt-multi-577e5775de.npz` | `KyberR4Ntt_shuffled_dcc30f1c/ntt_test/multi/577e5775de` | (4, 414) | 92 |
| `QuadR2NttShuffledDynamicExtraCycle-multi_no_random-8055b71277.npz` | `QuadR2NttShuffledDynamicExtraCycle_5f7421e0/ntt_test/multi_no_random/8055b71277` | (4, 305) | 76 |
| `QuadR2NttShuffledDynamicExtraCycle-multi-83de1065a.npz` | `QuadR2NttShuffledDynamicExtraCycle_5f7421e0/ntt_test/multi/83de1065a` | (4, 360) | 8 |
| `multi50.npz` (+ `multi50.meta.list`) | `QuadR2NttShuffledStall_61cb8c6f/ntt_test/multi/<hash>` for the 50 hashes in the list | (4, 371) | 74 |

SHA-256 of the files in this directory:

```
cefcf5ac183711ecbfd6d7ce7d3c73bb6b616e5cc13289a5a65ddcda60adacbb  KyberR4Ntt-multi_no_random-39003d6087.npz
dfe61c7e369e341befdb8c7330ed3c1beb5523794d7188dfb067b0dff2a7bd91  KyberR4Ntt-multi-577e5775de.npz
d524700c094f6058882dd76fbdbf8b9898d7cf6304d850cee9c5a1ec01ce26d7  QuadR2NttShuffledDynamicExtraCycle-multi_no_random-8055b71277.npz
08ba648604fb2ad27c989013abe35802bda3b6b49559f3de5916df6a372571e4  QuadR2NttShuffledDynamicExtraCycle-multi-83de1065a.npz
ff65c8c6c80d146c1742cd1125bacaccbdeade586c5bcec93aaf191d9f4afa73  QuadR2NttShuffledStall-multi_no_random-058a00d90.npz
e8b929b6d3b665b13d0b82b9c21360c3344e79e24b68db16b59f1ef7f2ea8f33  QuadR2NttShuffledStall-multi-245beb1812.npz
e8e0365af959f82fb553cf59731a2726494f7046350e135453abc92665d8385b  multi50.npz
f7c90bed3ee22da78d6efc66b4f846ab328cd88311164230795e8c47896571a3  multi50.meta.list
```

SHA-256 of the input files (to detect a changed source):

```
3f8e945f2bd940b9f9b278ec937a2553840e1b5fd8c704f2b157aee98b9321d8  QuadR2NttShuffledStall_61cb8c6f/ntt_test/multi_no_random/058a00d90/meta.json.gz
ab25172a1f76f1d427d1389e51c32b1e64d271faf30584649ef1c17f767513e1  QuadR2NttShuffledStall_61cb8c6f/ntt_test/multi_no_random/058a00d90/tvla.fst
02c056d7bc16acdab338ebb7190d8fbcbef13a2f0c8e4657dbac451b2962ab65  QuadR2NttShuffledStall_61cb8c6f/ntt_test/multi/245beb1812/meta.json.gz
134ab967b52140ef5541e9c588885f3944072e65b1c54a8381edeceef184070c  QuadR2NttShuffledStall_61cb8c6f/ntt_test/multi/245beb1812/tvla.fst
bea63844f486d49d313bce588d8a031fc8c2427084a84ae5b56e99e2100d7da7  KyberR4Ntt_shuffled_dcc30f1c/ntt_test/multi_no_random/39003d6087/meta.json.gz
ea4f66a2ea49ebbea401b95c67fc607f86571a9e6d62aa7f8cb83019118b1393  KyberR4Ntt_shuffled_dcc30f1c/ntt_test/multi_no_random/39003d6087/tvla.fst
75eb13f0224df98006f0c8c7de432bb30635c988422732cc2c15e7d0db226d5e  KyberR4Ntt_shuffled_dcc30f1c/ntt_test/multi/577e5775de/meta.json.gz
9f3bd36299f8ddd68af0f0fd0049e37fbab69cd8cba1bb0fa9894303be49a862  KyberR4Ntt_shuffled_dcc30f1c/ntt_test/multi/577e5775de/tvla.fst
77ac950a76d77e689c8869f6e13fd78e5998af8ccc463b2bc1b08b6488940476  QuadR2NttShuffledDynamicExtraCycle_5f7421e0/ntt_test/multi_no_random/8055b71277/meta.json.gz
c5033c22d651c16e7ec35c3d60188cd05265f43d5cc86df4a8dd0e301398d895  QuadR2NttShuffledDynamicExtraCycle_5f7421e0/ntt_test/multi_no_random/8055b71277/tvla.fst
e50e0ae110cc1c594e9c5b06e983ac8b2bf6806ee19bf5e2141d2db963b17ff9  QuadR2NttShuffledDynamicExtraCycle_5f7421e0/ntt_test/multi/83de1065a/meta.json.gz
1a48f88a9640c4bdfa4c4c63ce70803b0b3c651c5d6c05e7f9ad991908f80b26  QuadR2NttShuffledDynamicExtraCycle_5f7421e0/ntt_test/multi/83de1065a/tvla.fst
```

The multi run reads `meta.json.gz` and `traces.npz` of 50 batches (100 files). The SHA-256 of
the list of their own SHA-256 lines (`shasum -a 256 <hash>/meta.json.gz <hash>/traces.npz`, in
the order of `multi50.meta.list`, run in the multi run directory) is
`0ba44d1dd231dc4af1960bb8d088483f9e974d12f1dfcc962d65067e49eb8cee`.

## How to run the tests

```
R=/Volumes/src/krystals_hw/cocotb/tvla_run
SCASIM_REAL_TVALUES=QuadR2NttShuffledStall-multi-245beb1812 \
SCASIM_REAL_BATCH=$R/QuadR2NttShuffledStall_61cb8c6f/ntt_test/multi/245beb1812 \
  cargo test --release --test real_data tvla_t_values_match -- --ignored --nocapture

SCASIM_REAL_MULTI=$R/QuadR2NttShuffledStall_61cb8c6f/ntt_test/multi \
  cargo test --release --test real_data tvla_multi_batch -- --ignored --nocapture
```

Each test copies the inputs to a temporary directory and does not write to the source directory.
The tolerance is a relative difference of 1e-9 (`|a-b| / max(1, |a|, |b|)`).
