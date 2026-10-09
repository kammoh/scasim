# scasim-tvla

Turns a cocotb 2.1 testbench into a TVLA leakage-assessment harness for scasim.

- `scasim_tvla.meta`: writes `scasim_meta` version 1 files. It uses only the standard library.
- `scasim_tvla.session`: the `Tvla` session. It schedules the classes, opens and closes
  segments, and writes the metadata. It needs `cocotb>=2.1`.
- `scasim_tvla.test`: the `@tvla_test` decorator.

Install for development:

    pip install -e 'python/scasim-tvla[cocotb,test]'
    pytest python/scasim-tvla

The command `scasim-tvla` is reserved for the runner (not implemented yet).
