<div align="center">
  <p><a href="https://github.com/tslearn-team/tslearn"><img src="https://raw.githubusercontent.com/tslearn-team/tslearn/main/docs/_static/tslearn_logo_white_background.png?cache-control=no-cache" width="20%" alt="tslearn logo"/></a></p>
  <h1>tslearn-fast</h1>
  <p>An unofficial, performance-focused fork of tslearn for time-series machine learning in Python.</p>
</div>

<p align="center">
  <a href="https://www.python.org/downloads/"><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10+-blue.svg"></a>
  <a href="https://github.com/tslearn-team/tslearn"><img alt="Upstream tslearn" src="https://img.shields.io/badge/upstream-tslearn-blue"></a>
  <a href="LICENSE"><img alt="BSD-2-Clause license" src="https://img.shields.io/badge/license-BSD--2--Clause-green.svg"></a>
</p>

> [!IMPORTANT]
> `tslearn-fast` is an independent fork, not an official release of the
> [tslearn project](https://github.com/tslearn-team/tslearn). It aims to retain
> the public tslearn API while accelerating supported NumPy workloads.

## Why tslearn-fast

The fork adds fused or vectorized implementations for performance-sensitive
operations while retaining upstream-compatible fallbacks for unsupported
inputs and backends. The main areas include:

- DTW, Soft-DTW, Global Alignment Kernel, and Frechet distance matrices
- Sakoe-Chiba DTW paths and LB_Keogh-assisted nearest-neighbor search
- DBA and Soft-DTW barycenters
- STOMP matrix profiles and selected preprocessing transforms
- Clustering, nearest-neighbor, and SVM estimator integration

Correctness is covered by parity, regression, serialization, and edge-case
tests. An ASV benchmark suite is included so results can be measured on your
own data and hardware; no universal speedup is assumed.

- [Usage guide](USAGE.md)
- [Performance best practices](BEST_PRACTICES.md)
- [Benchmark guide](benchmarks/README.md)

## Measured performance

Benchmarked on 2026-09-21 against upstream commit
[`99cf640`](https://github.com/tslearn-team/tslearn/commit/99cf640bcc759fdf65dc47ad9c0ec1080768de1e)
on an Intel Core i5-11400 using Windows 11, Python 3.13.14, NumPy 2.5.1,
and Numba 0.67.0 with OpenMP. Values are the median of five runs after one
untimed JIT warm-up, using identical inputs and thread limits for both versions.

| Scenario | Upstream (ms, 1 thread) | tslearn-fast (ms, 1 thread) | Speedup (1 thread) | Speedup (2-thread cap) |
|---|---:|---:|---:|---:|
| DTW matrix, `N=100, L=32` | 56.016 | 7.438 | 7.53× | 7.90× |
| DTW matrix, `N=200, L=64` | 487.905 | 117.762 | 4.14× | 4.18× |
| Banded DTW, `N=80, L=200` | 161.027 | 34.846 | 4.62× | 4.62× |
| Multivariate DTW, `N=40, L=400, d=5` | 794.707 | 248.753 | 3.19× | 3.22× |
| Soft-DTW matrix, `N=100, L=32` | 1420.777 | 216.345 | 6.57× | 8.62× |
| Global Alignment Kernel, `N=100, L=32` | 186.412 | 36.425 | 5.12× | 5.09× |
| Frechet matrix, `N=100, L=32` | 54.708 | 10.008 | 5.47× | 5.39× |
| DTW k-neighbors, `200×50, L=64` | 122.428 | 16.502 | 7.42× | 7.46× |
| DTW k-means, `N=80, L=32, k=4` | 148.621 | 42.900 | 3.46× | 3.37× |
| Soft-DTW barycenter, `N=20, L=24` | 244.235 | 24.064 | 10.15× | 12.24× |
| Matrix profile, `L=1000, m=32` | 28.830 | 4.027 | 7.16× | 9.96× |

Observed speedups ranged from **3.19× to 10.15×** with a one-thread limit and
**3.22× to 12.24×** with a two-thread limit. Absolute timings above use the
one-thread limit; the last column is a separate two-thread-limit comparison.
All 22 benchmark-output comparisons matched within `rtol=1e-7, atol=1e-9`.

Thread limits cap Numba, OpenMP, and BLAS pools; they do not force each algorithm
to use that many threads. Default API `n_jobs` settings were preserved.
These are warm-run results for the listed synthetic workloads, not universal
speedup guarantees. Startup costs and numerical edge cases that select slower
fallbacks are not represented.

Reproduce the one-thread comparison from the repository root in PowerShell:

```powershell
$env:NUMBA_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
python tests/bench_vs_main.py --main-rev 99cf640bcc759fdf65dc47ad9c0ec1080768de1e --repeats 5
```

Repeat with all five limits set to `"2"`. See the
[usage guide](USAGE.md#measured-speedups-vs-upstream-base) for both timing tables,
full environment details, and limitations, and the
[benchmark guide](benchmarks/README.md) for other benchmark options.

## Upstream base and maintenance

The optimized branch is deliberately based on upstream tslearn commit
[`99cf640`](https://github.com/tslearn-team/tslearn/commit/99cf640bcc759fdf65dc47ad9c0ec1080768de1e),
the revision against which these changes were developed and tested. It does
not claim to include later upstream changes. The `upstream` remote can be used
to evaluate future upgrades separately.

This repository preserves the original BSD-2-Clause license and attribution.
For the official package, documentation, and support channels, visit
[tslearn-team/tslearn](https://github.com/tslearn-team/tslearn).

<hr>

<!-- Table of content -->

| Section | Description |
|-|-|
| [Measured performance](#measured-performance) | Speedups against upstream commit 99cf640 |
| [Installation](#installation) | Installing the dependencies and tslearn |
| [Getting started](#getting-started) | A quick introduction on how to use tslearn |
| [Available features](#available-features) | An extensive overview of tslearn's functionalities |
| [Documentation](#documentation) | A link to our API reference and a gallery of examples |
| [Contributing](#contributing) | A guide for heroes willing to contribute |
| [Citation](#referencing-tslearn) | A citation for tslearn for scholarly articles |

## Installation<a id="installation"></a>

Install the public `main` branch directly from GitHub:

```bash
python -m pip install "git+https://github.com/coder21cn/tslearn-fast.git@main"
```

For development, clone the repository and use an editable install:

```bash
git clone https://github.com/coder21cn/tslearn-fast.git
cd tslearn-fast
python -m pip install -e .
```

The distribution and import name remain `tslearn`, so install this fork in a
dedicated environment rather than alongside the official PyPI package. If you
want the official upstream release, use `python -m pip install tslearn` or
`conda install -c conda-forge tslearn`.

## Getting started<a id="getting-started"></a>

### 1. Getting the data in the right format
tslearn expects a time series dataset to be formatted as a 3D `numpy` array. The three dimensions correspond to the number of time series, the number of measurements per time series and the number of dimensions respectively (`n_ts, max_sz, d`). In order to get the data in the right format, different solutions exist:
* [You can use the utility functions such as `to_time_series_dataset`.](https://tslearn.readthedocs.io/en/stable/gen_modules/tslearn.utils.html#module-tslearn.utils)
* [You can convert from other popular time series toolkits in Python.](https://tslearn.readthedocs.io/en/stable/integration_other_software.html)
* [You can load any of the UCR datasets in the required format.](https://tslearn.readthedocs.io/en/stable/gen_modules/tslearn.datasets.html#module-tslearn.datasets)
* [You can generate synthetic data using the `generators` module.](https://tslearn.readthedocs.io/en/stable/gen_modules/tslearn.generators.html#module-tslearn.generators)

It should further be noted that tslearn [supports variable-length timeseries](https://tslearn.readthedocs.io/en/stable/variablelength.html).

```python3
>>> from tslearn.utils import to_time_series_dataset
>>> my_first_time_series = [1, 3, 4, 2]
>>> my_second_time_series = [1, 2, 4, 2]
>>> my_third_time_series = [1, 2, 4, 2, 2]
>>> X = to_time_series_dataset([my_first_time_series,
                                my_second_time_series,
                                my_third_time_series])
>>> y = [0, 1, 1]
```

### 2. Data preprocessing and transformations
Optionally, tslearn has several utilities to preprocess the data. In order to facilitate the convergence of different algorithms, you can [scale time series](https://tslearn.readthedocs.io/en/stable/gen_modules/tslearn.preprocessing.html#module-tslearn.preprocessing). Alternatively, in order to speed up training times, one can [resample](https://tslearn.readthedocs.io/en/stable/gen_modules/preprocessing/tslearn.preprocessing.TimeSeriesResampler.html#tslearn.preprocessing.TimeSeriesResampler) the data or apply a [piece-wise transformation](https://tslearn.readthedocs.io/en/stable/gen_modules/tslearn.piecewise.html#module-tslearn.piecewise).

```python3
>>> from tslearn.preprocessing import TimeSeriesScalerMinMax
>>> X_scaled = TimeSeriesScalerMinMax().fit_transform(X)
>>> print(X_scaled)
[[[0.] [0.667] [1.] [0.333] [nan]]
 [[0.] [0.333] [1.] [0.333] [nan]]
 [[0.] [0.333] [1.] [0.333] [0.333]]]
```

### 3. Training a model

After getting the data in the right format, a model can be trained. Depending on the use case, tslearn supports different tasks: classification, clustering and regression. For an extensive overview of possibilities, check out our [gallery of examples](https://tslearn.readthedocs.io/en/stable/auto_examples/index.html).

```python3
>>> from tslearn.neighbors import KNeighborsTimeSeriesClassifier
>>> knn = KNeighborsTimeSeriesClassifier(n_neighbors=1)
>>> knn.fit(X_scaled, y)
>>> print(knn.predict(X_scaled))
[0 1 1]
```

As can be seen, the models in tslearn follow the same API as those of the well-known scikit-learn. Moreover, they are fully compatible with it, allowing to use different scikit-learn utilities such as [hyper-parameter tuning and pipelines](https://tslearn.readthedocs.io/en/stable/auto_examples/neighbors/plot_knnts_sklearn.html).

### 4. More analyses

tslearn further allows to perform all different types of analysis. Examples include [calculating barycenters](https://tslearn.readthedocs.io/en/stable/gen_modules/tslearn.barycenters.html#module-tslearn.barycenters) of a group of time series or calculate the distances between time series using a [variety of distance metrics](https://tslearn.readthedocs.io/en/stable/gen_modules/tslearn.metrics.html#module-tslearn.metrics).

## Available features<a id="available-features"></a>

| data                                                                                                                                                                                         | processing                                                                                                              | clustering                                                                                                                                                       | classification                                                                                                                                                                          | regression                                                                                                                                                                           | metrics                                                                                                                              |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| [UCR Datasets](https://tslearn.readthedocs.io/en/stable/gen_modules/tslearn.datasets.html#module-tslearn.datasets)                                                                           | [Scaling](https://tslearn.readthedocs.io/en/stable/gen_modules/tslearn.preprocessing.html#module-tslearn.preprocessing) | [TimeSeriesKMeans](https://tslearn.readthedocs.io/en/stable/gen_modules/clustering/tslearn.clustering.TimeSeriesKMeans.html#tslearn.clustering.TimeSeriesKMeans) | [KNN Classifier](https://tslearn.readthedocs.io/en/stable/gen_modules/neighbors/tslearn.neighbors.KNeighborsTimeSeriesClassifier.html#tslearn.neighbors.KNeighborsTimeSeriesClassifier) | [KNN Regressor](https://tslearn.readthedocs.io/en/stable/gen_modules/neighbors/tslearn.neighbors.KNeighborsTimeSeriesRegressor.html#tslearn.neighbors.KNeighborsTimeSeriesRegressor) | [Dynamic Time Warping](https://tslearn.readthedocs.io/en/stable/gen_modules/metrics/tslearn.metrics.dtw.html#tslearn.metrics.dtw)    |
| [Generators](https://tslearn.readthedocs.io/en/stable/gen_modules/tslearn.generators.html#module-tslearn.generators)                                                                         | [Piecewise](https://tslearn.readthedocs.io/en/stable/gen_modules/tslearn.piecewise.html#module-tslearn.piecewise)       | [KShape](https://tslearn.readthedocs.io/en/stable/gen_modules/clustering/tslearn.clustering.KShape.html#tslearn.clustering.KShape)                               | [TimeSeriesSVC](https://tslearn.readthedocs.io/en/stable/gen_modules/svm/tslearn.svm.TimeSeriesSVC.html#tslearn.svm.TimeSeriesSVC)                                                      | [TimeSeriesSVR](https://tslearn.readthedocs.io/en/stable/gen_modules/svm/tslearn.svm.TimeSeriesSVR.html#tslearn.svm.TimeSeriesSVR)                                                   | [Global Alignment Kernel](https://tslearn.readthedocs.io/en/stable/gen_modules/metrics/tslearn.metrics.gak.html#tslearn.metrics.gak) |
| Conversion([1](https://tslearn.readthedocs.io/en/stable/gen_modules/tslearn.utils.html#module-tslearn.utils), [2](https://tslearn.readthedocs.io/en/stable/integration_other_software.html)) |                                                                                                                         | [KernelKmeans](https://tslearn.readthedocs.io/en/stable/gen_modules/clustering/tslearn.clustering.KernelKMeans.html#tslearn.clustering.KernelKMeans)             | [LearningShapelets](https://tslearn.readthedocs.io/en/stable/gen_modules/shapelets/tslearn.shapelets.LearningShapelets.html)                                    | [MLP](https://tslearn.readthedocs.io/en/stable/gen_modules/tslearn.neural_network.html#module-tslearn.neural_network)                                                                | [Barycenters](https://tslearn.readthedocs.io/en/stable/gen_modules/tslearn.barycenters.html#module-tslearn.barycenters)              |
|                                                                                                                                                                                              |                                                                                                                         |                                                                                                                                                                  | [Early Classification](https://tslearn.readthedocs.io/en/stable/gen_modules/tslearn.early_classification.html#module-tslearn.early_classification)                                      |                                                                                                                                                                                      | [Matrix Profile](https://tslearn.readthedocs.io/en/stable/gen_modules/tslearn.matrix_profile.html#module-tslearn.matrix_profile)     |


## Documentation<a id="documentation"></a>

The documentation is hosted at [readthedocs](http://tslearn.readthedocs.io/en/stable/index.html). It includes an [API](https://tslearn.readthedocs.io/en/stable/reference.html), [gallery of examples](https://tslearn.readthedocs.io/en/stable/auto_examples/index.html) and a [user guide](https://tslearn.readthedocs.io/en/stable/user_guide/userguide.html).

## Contributing<a id="contributing"></a>

If you would like to contribute to `tslearn`, please have a look at [our contribution guidelines](https://github.com/tslearn-team/tslearn/blob/99cf640bcc759fdf65dc47ad9c0ec1080768de1e/CONTRIBUTING.md). A list of interesting TODO's can be found [here](https://github.com/tslearn-team/tslearn/issues?utf8=✓&q=is%3Aissue%20is%3Aopen%20label%3A%22new%20feature%22%20). **If you want other ML methods for time series to be added to this TODO list, do not hesitate to [open an issue](https://github.com/tslearn-team/tslearn/issues/new/choose)!**

## Referencing tslearn<a id="referencing-tslearn"></a>

If you use `tslearn` in a scientific publication, we would appreciate citations:

```bibtex
@article{JMLR:v21:20-091,
  author  = {Romain Tavenard and Johann Faouzi and Gilles Vandewiele and 
             Felix Divo and Guillaume Androz and Chester Holtz and 
             Marie Payne and Roman Yurchak and Marc Ru{\ss}wurm and 
             Kushal Kolar and Eli Woods},
  title   = {Tslearn, A Machine Learning Toolkit for Time Series Data},
  journal = {Journal of Machine Learning Research},
  year    = {2020},
  volume  = {21},
  number  = {118},
  pages   = {1-6},
  url     = {http://jmlr.org/papers/v21/20-091.html}
}
```

#### Acknowledgments
Authors would like to thank Mathieu Blondel for providing code for [Kernel k-means](https://gist.github.com/mblondel/6230787) and [Soft-DTW](https://github.com/mblondel/soft-dtw), and to Mehran Maghoumi for his [`torch`-compatible implementation of SoftDTW](https://github.com/Maghoumi/pytorch-softdtw-cuda).
