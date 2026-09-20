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

Side-by-side timing against upstream commit `99cf640` using `tests/bench_vs_main.py`. Values are the median of three runs after a JIT warm-up on an Intel Core i5-11400 with Python 3.13.14, NumPy 2.5.1, and Numba 0.67.0:

| scenario | upstream | tslearn-fast | speedup |
|---|---:|---:|---:|
| `cdist_dtw[N=100, L=32]` | 0.0549s | 0.0073s | **7.48×** |
| `cdist_dtw[N=200, L=64]` | 0.4742s | 0.1265s | **3.75×** |
| `cdist_dtw[N=80, L=200, sakoe=10]` | 0.1591s | 0.0354s | **4.50×** |
| `cdist_dtw[N=40, L=400, multivariate d=5]` | 0.8484s | 0.2560s | **3.31×** |
| `cdist_soft_dtw[N=100, L=32]` | 1.4102s | 0.0476s | **29.61×** |
| `cdist_gak[N=100, L=32]` | 0.1816s | 0.0395s | **4.59×** |
| `cdist_frechet[N=100, L=32]` | 0.0552s | 0.0103s | **5.37×** |
| `KNeighborsTimeSeries(metric='dtw').kneighbors[N=200×50, sakoe=4, k=3]` | 0.1225s | 0.0192s | **6.39×** |
| `TimeSeriesKMeans(metric='dtw').fit_predict[N=80, k=4, sakoe=3]` | 0.1504s | 0.0433s | **3.47×** |
| `softdtw_barycenter[N=20, L=24, max_iter=20]` | 0.4381s | 0.0293s | **14.97×** |
| `MatrixProfile.fit_transform[L=1000, m=32]` | 0.0285s | 0.0017s | **16.82×** |

Results vary with data shape, hardware, threading, and dependency versions. Reproduce with:

```bash
python tests/bench_vs_main.py --main-rev 99cf640bcc759fdf65dc47ad9c0ec1080768de1e --repeats 3
```

K-Means now matches the rest of the suite after fusing DBA's per-
iteration assignment+update into a vectorized `numpy.add.at` +
`numpy.bincount` scatter (the legacy per-position `numpy.average` loop
was the dominant cost on small problems, not the DTW kernel itself).

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
