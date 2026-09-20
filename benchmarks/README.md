# tslearn benchmarks

This directory holds an [airspeed velocity](https://asv.readthedocs.io/)
benchmark suite for the performance-sensitive paths in tslearn: DTW
cdists, LB_Keogh top-k, Soft-DTW, GAK, Frechet, KMeans/KShape, the STOMP
matrix profile, and the preprocessing transforms that have a vectorized
fast path.

Each module exposes:

- One or more **asv classes** (`params`, `param_names`, `setup`, `time_*`).
  These are picked up automatically by `asv run`.
- A **`__main__`** block that prints a small comparison table when the
  module is run directly. This is the right tool when you just want a
  quick local check without setting up asv.

## Quick local check (no asv)

```
python -m benchmarks.bench_dtw
python -m benchmarks.bench_dtw_topk
python -m benchmarks.bench_clustering
python -m benchmarks.bench_matrix_profile
```

Each prints one line per `(time_*, param-tuple)` with the best of three
runs in milliseconds, after a warm-up call to amortize numba JIT.

## Tracking changes over time (asv)

```
pip install asv
asv machine --yes               # one-time, names this machine
asv run                         # benchmark the current branch
asv run main..HEAD              # benchmark every commit since main
asv publish && asv preview      # serve an HTML report
```

The asv config lives in `asv.conf.json` at the repo root. Results land in
`.asv/` (gitignored).

## Adding a new benchmark

1. Drop a `bench_<area>.py` into this directory.
2. Define a class with `params`, `param_names`, `setup(self, *params)`,
   and at least one `time_<name>(self, *params)` method.
3. Add a `__main__` block that calls `run_standalone(MyClass, "label")`
   from `_common.py` so the module also works as a script.

Keep parameter grids small — asv runs the full cartesian product per
commit, so doubling a grid doubles the wall-clock cost.
