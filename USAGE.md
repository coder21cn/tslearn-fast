# Usage guide for the performance branch

This branch keeps the public tslearn API mostly unchanged, but adds several
Numba-backed fast paths and benchmark utilities. Most users should continue to
call the standard tslearn functions and estimators; the accelerated code is
selected automatically for supported NumPy inputs.

## Install and import

Use the repository checkout directly while developing:

```bash
python -m pip install -e .
```

For local review without an editable install, put the checkout on
`PYTHONPATH`:

```bash
# Linux / macOS
export PYTHONPATH=/path/to/tslearn
# Windows (cmd)
set PYTHONPATH=path\to\tslearn
```

The usual tslearn imports still apply:

```python
import numpy as np
from tslearn.metrics import cdist_dtw, cdist_soft_dtw, cdist_gak, cdist_frechet
from tslearn.clustering import TimeSeriesKMeans, KShape
from tslearn.neighbors import KNeighborsTimeSeries
from tslearn.matrix_profile import MatrixProfile
from tslearn.preprocessing import TimeSeriesResampler
```

## Input format

tslearn expects time series datasets as arrays of shape
`(n_ts, max_sz, d)`, where:

- `n_ts` is the number of series
- `max_sz` is the padded time dimension
- `d` is the number of features

Variable-length series use trailing `NaN` rows as padding. Mid-series `NaN`
or infinite values are data-quality issues for many metrics and should be
tested carefully because some legacy paths propagate them while others raise.

```python
from tslearn.utils import to_time_series_dataset

X = to_time_series_dataset([
    [1.0, 2.0, 3.0],
    [1.0, 2.0],
])
```

## Accelerated metrics

The following public functions now dispatch to fused NumPy/Numba kernels when
the inputs are supported:

- `cdist_dtw`
- `cdist_soft_dtw`
- `cdist_gak`
- `cdist_frechet`
- `dtw` and `dtw_path` for Sakoe-Chiba NumPy inputs

Example:

```python
rng = np.random.RandomState(0)
X = rng.randn(64, 128, 1)
Y = rng.randn(32, 128, 1)

D = cdist_dtw(X, Y)
D_band = cdist_dtw(
    X,
    Y,
    global_constraint="sakoe_chiba",
    sakoe_chiba_radius=5,
)
S = cdist_soft_dtw(X, Y, gamma=1.0)
G = cdist_gak(X, Y, sigma=1.0)
F = cdist_frechet(X, Y)
```

The Itakura DTW fast path requires equal valid lengths within each dataset.
If variable lengths make a shared Itakura parallelogram impossible, the code
falls back to the generic path.

## Top-k DTW neighbors

`KNeighborsTimeSeries(metric="dtw")` can use an LB_Keogh top-k fast path when
the effective DTW constraint is Sakoe-Chiba and all valid lengths match.

```python
knn = KNeighborsTimeSeries(
    n_neighbors=5,
    metric="dtw",
    metric_params={"sakoe_chiba_radius": 4},
    n_jobs=4,
)
knn.fit(X)
dists, indices = knn.kneighbors(Y)
```

The top-k path is intentionally narrow. It should fall back when inputs are
variable-length, unsupported by the Sakoe-only kernel, or when the requested
neighbor count must be handled by the sklearn fallback.

## Clustering

`TimeSeriesKMeans` now has faster DTW assignment and parallel barycenter
updates for DTW and Soft-DTW when `n_jobs` is greater than 1.

```python
km = TimeSeriesKMeans(
    n_clusters=4,
    metric="dtw",
    metric_params={"sakoe_chiba_radius": 4},
    n_jobs=4,
    random_state=0,
)
labels = km.fit_predict(X)
```

`KShape` accepts `n_jobs` for parallel per-cluster shape extraction:

```python
ks = KShape(n_clusters=4, n_jobs=4, random_state=0)
labels = ks.fit_predict(X)
```

## Matrix profile

The NumPy implementation of `MatrixProfile` uses a STOMP-style kernel instead
of materializing all subsequence distances.

```python
mp = MatrixProfile(subsequence_length=32, implementation="numpy", scale=True)
profile = mp.fit_transform(X)
```

Matrix profiles remain univariate (`d == 1`) for drop-in compatibility with
`main` and the stumpy-backed implementations.

## Preprocessing and PAA

`TimeSeriesResampler` and `PiecewiseAggregateApproximation` include vectorized
paths for common equal-length NumPy inputs:

```python
from tslearn.piecewise import PiecewiseAggregateApproximation

X_resampled = TimeSeriesResampler(sz=64).fit_transform(X)
X_paa = PiecewiseAggregateApproximation(n_segments=16).fit_transform(X)
```

The slower legacy-style path is still used when variable length or non-finite
values require the older semantics.

## n_jobs behavior

The new metric kernels map `n_jobs` to Numba thread counts:

- `None` and `1` use one Numba thread
- positive integers request that many Numba threads, **clamped to
  `numba.config.NUMBA_NUM_THREADS`** (the configured pool size, usually
  the CPU count). Requests above that are silently capped — Numba's
  `set_num_threads` rejects oversize requests, so the branch clamps
  rather than crashes. The legacy joblib-backed API accepted
  oversubscription, so this is a behavior cap, not a hard error
- `-1` leaves Numba's current/default thread count in place
- `0` is invalid and raises, matching joblib behavior
- negative values below `-1` are resolved through joblib-style effective jobs

Be cautious when nesting parallel calls. The branch tries to avoid the worst
oversubscription cases, but benchmark real workloads rather than assuming
larger `n_jobs` is always faster.

Cold-start cost: every fast-path numba kernel is decorated with
`cache=True`, so the JIT compile cost (typically 0.5–2 s per kernel) is
paid only on the first run after install. Subsequent processes load
from Numba's on-disk cache. The matrix-profile STOMP kernel is the one
exception (it uses `numba.get_thread_id()` internally, which Numba
treats as a dynamic global and refuses to cache); it recompiles every
process.

## Measured speedups vs upstream base

These measurements of the corrected implementation replace the earlier
performance figures. It remains faster than upstream commit `99cf640` in
all 11 scenarios. Earlier figures should not be treated as measurements of
this corrected implementation; differences from historical results do not
isolate the overhead of the fixes without matching all benchmark conditions.

### Method and environment

- Measured on 2026-09-21 against upstream commit
  `99cf640bcc759fdf65dc47ad9c0ec1080768de1e`.
- Measured implementation: `tslearn/` Git tree
  `ad86b6d03fd8a774685c38ede322f7bd73459cbc` (unchanged by this documentation update).
- Intel Core i5-11400 @ 2.60 GHz, 6 physical cores / 12 logical processors;
  Windows 11.
- Python 3.13.14, NumPy 2.5.1, Numba 0.67.0, SciPy 1.18.0,
  scikit-learn 1.9.0, and joblib 1.5.3; Numba threading layer `omp` (OpenMP).
- Existing scenarios from `tests/bench_vs_main.py`, with helpers from
  `tests/_bench_runner.py`; identical deterministic inputs (seed 0).
- Median of five timed calls after one untimed JIT warm-up per scenario;
  startup and compilation excluded. Baseline ran before the fork for each profile.
- `NUMBA_NUM_THREADS`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
  `MKL_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` were all set to 1 or 2,
  identically for both versions. These are limits, not forced thread counts;
  default API `n_jobs` settings were preserved.

All 22 benchmark-output comparisons matched at `rtol=1e-7, atol=1e-9`.
This checks the benchmark helpers' return values (for example, labels for
k-means), not every estimator attribute.

### One-thread limit

| Scenario | 99cf640 (ms) | tslearn-fast (ms) | Speedup |
|---|---:|---:|---:|
| `cdist_dtw[N=100, L=32]` | 56.016 | 7.438 | 7.53× |
| `cdist_dtw[N=200, L=64]` | 487.905 | 117.762 | 4.14× |
| `cdist_dtw[N=80, L=200, sakoe=10]` | 161.027 | 34.846 | 4.62× |
| `cdist_dtw[N=40, L=400, multivariate d=5]` | 794.707 | 248.753 | 3.19× |
| `cdist_soft_dtw[N=100, L=32]` | 1420.777 | 216.345 | 6.57× |
| `cdist_gak[N=100, L=32]` | 186.412 | 36.425 | 5.12× |
| `cdist_frechet[N=100, L=32]` | 54.708 | 10.008 | 5.47× |
| `kneighbors_dtw[N_train=200, N_test=50, L=64, sakoe=4, k=3]` | 122.428 | 16.502 | 7.42× |
| `kmeans_dtw[N=80, L=32, k=4, sakoe=3]` | 148.621 | 42.900 | 3.46× |
| `softdtw_barycenter[N=20, L=24]` | 244.235 | 24.064 | 10.15× |
| `matrix_profile[N=1, L=1000, m=32]` | 28.830 | 4.027 | 7.16× |

### Two-thread limit

| Scenario | 99cf640 (ms) | tslearn-fast (ms) | Speedup |
|---|---:|---:|---:|
| `cdist_dtw[N=100, L=32]` | 58.657 | 7.429 | 7.90× |
| `cdist_dtw[N=200, L=64]` | 489.441 | 117.122 | 4.18× |
| `cdist_dtw[N=80, L=200, sakoe=10]` | 161.043 | 34.851 | 4.62× |
| `cdist_dtw[N=40, L=400, multivariate d=5]` | 799.504 | 248.266 | 3.22× |
| `cdist_soft_dtw[N=100, L=32]` | 1423.356 | 165.173 | 8.62× |
| `cdist_gak[N=100, L=32]` | 185.269 | 36.422 | 5.09× |
| `cdist_frechet[N=100, L=32]` | 55.444 | 10.277 | 5.39× |
| `kneighbors_dtw[N_train=200, N_test=50, L=64, sakoe=4, k=3]` | 123.415 | 16.541 | 7.46× |
| `kmeans_dtw[N=80, L=32, k=4, sakoe=3]` | 149.901 | 44.464 | 3.37× |
| `softdtw_barycenter[N=20, L=24]` | 243.140 | 19.868 | 12.24× |
| `matrix_profile[N=1, L=1000, m=32]` | 28.940 | 2.907 | 9.96× |

The k-means scenario uses `max_iter=3`, `n_init=1`, and `random_state=0`;
the Soft-DTW barycenter uses `gamma=1.0` and `max_iter=20`; matrix profile
uses `implementation="numpy"` and `scale=True`.

These synthetic, warm-run measurements on one machine are not guarantees for
all workloads. They do not measure cold starts, memory use, the workqueue
backend, or numerical edge cases that deliberately use slower fallbacks.

### Reproduce the timings

From the repository root, in PowerShell:

```powershell
$env:NUMBA_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
python tests/bench_vs_main.py --main-rev 99cf640 --repeats 5
```

Repeat with all five limits set to `"2"`. Use the dependency versions above
for a comparable environment. The command reports timings; output parity was
checked separately during this rerun.

## Benchmarks

asv-style benchmark scripts live in `benchmarks/` for cross-commit
tracking:

```bash
python -m benchmarks.bench_dtw
python -m benchmarks.bench_dtw_topk
python -m benchmarks.bench_soft_dtw
python -m benchmarks.bench_gak
python -m benchmarks.bench_frechet
python -m benchmarks.bench_clustering
python -m benchmarks.bench_matrix_profile
python -m benchmarks.bench_preprocessing
```

For commit-to-commit tracking, use asv:

```bash
python -m pip install asv
asv machine --yes
asv run main..HEAD
asv publish
asv preview
```

## Correctness checks

Run the focused fast-path tests before relying on a performance change:

```bash
python -m pytest tests/test_fast_paths.py tests/test_metrics.py tests/test_neighbors.py -q
python -m pytest tests/test_matrixprofile.py tests/test_preprocessing.py tests/test_piecewise.py -q
```

For broader coverage:

```bash
python -m pytest -q
```

When changing a fast path, compare against `main` on:

- empty datasets and empty cross-distance outputs
- variable-length padded datasets
- all-NaN or zero-length members
- mid-series `NaN` and `Inf` values
- Sakoe-Chiba, Itakura, and ambiguous constraint combinations
- `n_jobs=None`, `1`, positive values, `0`, `-1`, and lower negative values

These are the cases most likely to drift when replacing per-pair Python
dispatch with fused kernels.

## Optional Keras dependency

Keras is still only needed for shapelet learning workflows such as
`LearningShapelets`. The metric, clustering, preprocessing, matrix-profile,
and benchmark changes in this branch do not require Keras.
