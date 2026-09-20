"""Differential parity harness vs the ``main`` branch.

Runs a matrix of public fast-path calls (``cdist_dtw``, ``cdist_soft_dtw``,
``cdist_gak``, ``cdist_frechet``, ``cdist_soft_dtw_normalized``,
``MatrixProfile``, ``TimeSeriesResampler``) against an edge-case input
matrix (empty, mid-NaN, mid-Inf, trailing-NaN, all-NaN-padded, zero-length,
n_jobs ∈ {None, 0, 1, -1, 99}, ambiguous + float Sakoe radius, Itakura)
and compares the outputs to ``main``'s.

Why a script (not a pytest module): the goal is one diagnostic run that
prints every mismatch in one go. Also useful: case ids are stable, so
the first failure isn't necessarily the first parametrize id, and
collecting all of them is more actionable than ``pytest -x``.

Usage::

    python tests/check_main_parity.py
    python tests/check_main_parity.py --main-rev origin/main

Exit code: 0 if all cases match, 1 if any drift, 2 if setup failed.
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


# Force the repo root onto ``sys.path`` so ``tests._main_worktree``
# resolves to ours even inside an editable install where another
# ``tests`` package may shadow it from site-packages.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from tests._main_worktree import REPO_ROOT, WORKTREE_DIR, ensure_worktree

RUNNER = REPO_ROOT / "tests" / "_main_parity_runner.py"

# Tolerance shared between ``_values_equal``'s np.allclose call and the
# ``compare`` first-mismatch locator — keep these in sync so the
# diagnostic doesn't lie about which cells differ.
_ATOL = 1e-9
_RTOL = 1e-7


# Cases where HEAD is intentionally allowed to diverge from main. Keys are
# patterns matched against case_id (only ``*`` is special — matches any
# substring; brackets are literal, since case_ids contain ``[...]`` segments
# that we don't want fnmatch to read as character classes). Values are the
# reason. Matched cases are reported under "Expected drifts" and do not
# fail the exit code.
#
# Example::
#
#     EXPECTED_DRIFTS = {
#         "some_intentional_extension[*]":
#             "branch intentionally extends main behavior for this surface",
#         "cdist_frechet[*][sakoe_r1]":
#             "Frechet Sakoe cdist now matches the single-pair API",
#     }
EXPECTED_DRIFTS: dict[str, str] = {
    # cdist_frechet on main forwards the public-string ``global_constraint``
    # to a numba-jitted kernel whose signature defaults to int 0; the
    # string-vs-int comparisons silently fall through to no-constraint.
    # We pass the int code now, so cdist matches the per-pair entrypoint
    # — divergence appears only on the configurations main was buggy on
    # (Sakoe-Chiba, Itakura, and ambiguous constraint combinations).
    "cdist_frechet[*][sakoe_r1]":
        "cdist_frechet now applies Sakoe-Chiba mask (main: silently dropped)",
    "cdist_frechet_self[*][sakoe_r1]":
        "cdist_frechet now applies Sakoe-Chiba mask (main: silently dropped)",
    "cdist_frechet[*][sakoe_r1f]":
        "cdist_frechet now applies Sakoe-Chiba mask (main: silently dropped)",
    "cdist_frechet_self[*][sakoe_r1f]":
        "cdist_frechet now applies Sakoe-Chiba mask (main: silently dropped)",
    "cdist_frechet[*][sakoe_rfrac]":
        "cdist_frechet now applies Sakoe-Chiba mask (main: silently dropped)",
    "cdist_frechet_self[*][sakoe_rfrac]":
        "cdist_frechet now applies Sakoe-Chiba mask (main: silently dropped)",
    "cdist_frechet[*][ambiguous]":
        "cdist_frechet now raises on ambiguous constraints (main: silently dropped)",
    "cdist_frechet_self[*][ambiguous]":
        "cdist_frechet now raises on ambiguous constraints (main: silently dropped)",
    "cdist_frechet[*][explicit_sakoe]":
        "cdist_frechet now applies Sakoe-Chiba mask (main: silently dropped)",
    "cdist_frechet_self[*][explicit_sakoe]":
        "cdist_frechet now applies Sakoe-Chiba mask (main: silently dropped)",
    "cdist_frechet[*][explicit_sakoe_default]":
        "cdist_frechet now applies Sakoe-Chiba mask (main: silently dropped)",
    "cdist_frechet_self[*][explicit_sakoe_default]":
        "cdist_frechet now applies Sakoe-Chiba mask (main: silently dropped)",
    "cdist_frechet[*][itakura_slope]":
        "cdist_frechet now applies inferred Itakura mask (main: silently dropped)",
    "cdist_frechet_self[*][itakura_slope]":
        "cdist_frechet now applies inferred Itakura mask (main: silently dropped)",
    "cdist_frechet[*][explicit_itakura]":
        "cdist_frechet now applies Itakura mask (main: silently dropped)",
    "cdist_frechet_self[*][explicit_itakura]":
        "cdist_frechet now applies Itakura mask (main: silently dropped)",
    "cdist_frechet[*][explicit_itakura_default]":
        "cdist_frechet now applies Itakura mask (main: silently dropped)",
    "cdist_frechet_self[*][explicit_itakura_default]":
        "cdist_frechet now applies Itakura mask (main: silently dropped)",
}


def _expected_drift_reason(case_id: str) -> str | None:
    for pattern, reason in EXPECTED_DRIFTS.items():
        regex = ".*".join(re.escape(part) for part in pattern.split("*"))
        if re.fullmatch(regex, case_id):
            return reason
    return None


# ---------------------------------------------------------------------------
# Case generation
# ---------------------------------------------------------------------------

_RNG = np.random.RandomState(0)
_SMALL_1D = _RNG.randn(3, 5, 1).astype(np.float64)
_SMALL_2D = _RNG.randn(4, 5, 2).astype(np.float64)


# Each entry: (id, dataset). Stored as numpy arrays to keep pickling cheap.
DATASETS = [
    ("small_1d", _SMALL_1D),
    ("small_2d", _SMALL_2D),
    ("empty_n", np.empty((0, 5, 1), dtype=np.float64)),
    ("empty_zero", np.empty((0, 0, 1), dtype=np.float64)),
    ("midnan", np.array([[[0.0], [np.nan], [1.0]],
                         [[0.0], [1.0], [2.0]]], dtype=np.float64)),
    ("midinf", np.array([[[0.0], [np.inf], [1.0]],
                         [[0.0], [1.0], [2.0]]], dtype=np.float64)),
    ("trailing_nan", np.array([[[1.0], [2.0], [np.nan]],
                                [[1.0], [2.0], [3.0]]], dtype=np.float64)),
    # All-NaN padded series: trims to (0, 1) on main → SquaredEuclidean raises.
    ("all_nan_padded", np.array([[[np.nan]], [[1.0]]], dtype=np.float64)),
]


# Common DTW / Frechet kwargs grid.
DTW_KWARGS = [
    ("default", {}),
    ("sakoe_r1", {"sakoe_chiba_radius": 1}),
    ("sakoe_r1f", {"sakoe_chiba_radius": 1.0}),         # float radius
    ("sakoe_rfrac", {"sakoe_chiba_radius": 0.5}),        # fractional radius
    ("sakoe_rinf", {"sakoe_chiba_radius": np.inf}),      # main: mask → no-constraint
    ("sakoe_rnan", {"sakoe_chiba_radius": np.nan}),      # main: mask handles
    ("ambiguous", {"sakoe_chiba_radius": 2,
                   "itakura_max_slope": 2.0}),          # main raises
    ("itakura_slope", {"itakura_max_slope": 2.0}),
    ("explicit_sakoe", {"global_constraint": "sakoe_chiba",
                        "sakoe_chiba_radius": 1}),
    ("explicit_sakoe_default", {"global_constraint": "sakoe_chiba"}),
    ("explicit_itakura", {"global_constraint": "itakura",
                          "itakura_max_slope": 2.0}),
    ("explicit_itakura_default", {"global_constraint": "itakura"}),
    ("njobs0", {"n_jobs": 0}),                           # main raises
    ("njobs99", {"n_jobs": 99}),                         # main accepts
    ("njobs_neg1", {"n_jobs": -1}),
    ("njobs_2", {"n_jobs": 2}),
]


def iter_cases():
    """Yield (case_id, dotted_func, args, kwargs) tuples."""
    for ds_id, ds in DATASETS:
        # cdist_dtw: cross + self
        for kw_id, kw in DTW_KWARGS:
            yield (f"cdist_dtw[{ds_id}][{kw_id}]",
                   "tslearn.metrics.cdist_dtw", (ds, ds), kw)
            yield (f"cdist_dtw_self[{ds_id}][{kw_id}]",
                   "tslearn.metrics.cdist_dtw", (ds,), kw)
            # cdist_frechet: cross + self
            yield (f"cdist_frechet[{ds_id}][{kw_id}]",
                   "tslearn.metrics.cdist_frechet", (ds, ds), kw)
            yield (f"cdist_frechet_self[{ds_id}][{kw_id}]",
                   "tslearn.metrics.cdist_frechet", (ds,), kw)

        # cdist_soft_dtw: gamma sweep. NaN gamma exercises the
        # finite-gamma fast-path gate (legacy raises ZeroDivisionError;
        # without the gate the fused kernel would surface a SystemError).
        for gamma in (0.5, 1.0, np.nan):
            yield (f"cdist_soft_dtw[{ds_id}][gamma={gamma}]",
                   "tslearn.metrics.cdist_soft_dtw", (ds, ds),
                   {"gamma": gamma})
            yield (f"cdist_soft_dtw_self[{ds_id}][gamma={gamma}]",
                   "tslearn.metrics.cdist_soft_dtw", (ds,),
                   {"gamma": gamma})
            yield (f"cdist_soft_dtw_norm[{ds_id}][gamma={gamma}]",
                   "tslearn.metrics.cdist_soft_dtw_normalized",
                   (ds, ds), {"gamma": gamma})

        # cdist_gak: sigma sweep. NaN sigma exercises the finite-sigma
        # fast-path gate (legacy propagates NaN; without the gate the
        # fused kernel surfaces ZeroDivisionError on its ``1/sigma**2``).
        for sigma in (1.0, 2.0, np.nan):
            yield (f"cdist_gak[{ds_id}][sigma={sigma}]",
                   "tslearn.metrics.cdist_gak", (ds, ds),
                   {"sigma": sigma})
            yield (f"cdist_gak_self[{ds_id}][sigma={sigma}]",
                   "tslearn.metrics.cdist_gak", (ds,),
                   {"sigma": sigma})

    # Asymmetric cross pairs: catch inactive-side bugs that the
    # symmetric ``(ds, ds)`` cases above can't reach (e.g. empty-vs-
    # invalid Soft-DTW, where one side is zero-row and the other holds
    # a NaN row). Default kwargs only — the kwargs sweep above already
    # covers per-config behavior on (ds, ds), and re-running every
    # kwargs config across N² pairs would balloon the matrix without
    # adding signal.
    cross_funcs = [
        ("cdist_dtw", "tslearn.metrics.cdist_dtw", {}),
        ("cdist_frechet", "tslearn.metrics.cdist_frechet", {}),
        ("cdist_soft_dtw", "tslearn.metrics.cdist_soft_dtw",
         {"gamma": 1.0}),
        ("cdist_soft_dtw_norm",
         "tslearn.metrics.cdist_soft_dtw_normalized",
         {"gamma": 1.0}),
        ("cdist_gak", "tslearn.metrics.cdist_gak", {"sigma": 1.0}),
    ]
    for fn_id, fn_dotted, fn_kw in cross_funcs:
        for a_id, ds_a in DATASETS:
            for b_id, ds_b in DATASETS:
                if a_id == b_id:
                    continue  # already covered by (ds, ds) above
                # Mismatched feature dim is rejected by the underlying
                # pairwise-distance call; pairing them here is just user
                # error (and HEAD vs main may handle that error differently
                # without it being a fast-path drift).
                if ds_a.shape[2] != ds_b.shape[2]:
                    continue
                yield (f"{fn_id}_cross[{a_id}|{b_id}]",
                       fn_dotted, (ds_a, ds_b), fn_kw)

    # Pytorch backend axis. Every ``be.is_numpy`` gate in the metric
    # dispatchers picks numpy-vs-torch from the input array type, so
    # the same call with torch inputs must not silently route into the
    # numba fast path. Coverage focuses on representative shapes per
    # function rather than the full kwargs sweep — the dispatch logic
    # is the same regardless of constraint kwargs, so re-running every
    # kwargs config across torch would balloon the matrix without
    # adding signal. Empty / mid-NaN datasets check that the same gate
    # decisions match across backends on edge inputs.
    torch_funcs_cdist = [
        ("cdist_dtw", "tslearn.metrics.cdist_dtw", {}),
        ("cdist_frechet", "tslearn.metrics.cdist_frechet", {}),
        ("cdist_soft_dtw", "tslearn.metrics.cdist_soft_dtw",
         {"gamma": 1.0}),
        ("cdist_soft_dtw_norm",
         "tslearn.metrics.cdist_soft_dtw_normalized",
         {"gamma": 1.0}),
        ("cdist_gak", "tslearn.metrics.cdist_gak", {"sigma": 1.0}),
    ]
    torch_datasets = [
        (did, ds) for did, ds in DATASETS
        if did in ("small_1d", "midnan", "empty_n", "all_nan_padded")
    ]
    for fn_id, fn_dotted, fn_kw in torch_funcs_cdist:
        for ds_id, ds in torch_datasets:
            yield (f"{fn_id}[torch][{ds_id}]",
                   "local:run_torch_dispatch",
                   (fn_dotted, (ds, ds), fn_kw), {})
            yield (f"{fn_id}_self[torch][{ds_id}]",
                   "local:run_torch_dispatch",
                   (fn_dotted, (ds,), fn_kw), {})

    torch_single_pair = [
        ("dtw", "tslearn.metrics.dtw"),
        ("dtw_path", "tslearn.metrics.dtw_path"),
        ("frechet", "tslearn.metrics.frechet"),
        ("soft_dtw", "tslearn.metrics.soft_dtw"),
        ("gak", "tslearn.metrics.gak"),
    ]
    sp_src = dict(DATASETS)["small_1d"]
    sp_s1, sp_s2 = sp_src[0], sp_src[1]
    for fn_id, fn_dotted in torch_single_pair:
        yield (f"{fn_id}[torch][small_1d]",
               "local:run_torch_dispatch",
               (fn_dotted, (sp_s1, sp_s2), {}), {})

    # Single-pair DTW / Frechet: pull two finite series + a NaN-bearing
    # series out of the dataset matrix (skip when the dataset has fewer
    # than 2 rows so we don't pad).
    by_id = dict(DATASETS)
    for kw_id, kw in DTW_KWARGS:
        # n_jobs is meaningless for single-pair; skip those kwargs.
        if "n_jobs" in kw:
            continue
        for src_id in ("small_1d", "small_2d", "midnan", "trailing_nan"):
            ds = by_id[src_id]
            if ds.shape[0] < 2:
                continue
            s1, s2 = ds[0], ds[1]
            yield (f"dtw[{src_id}][{kw_id}]",
                   "tslearn.metrics.dtw", (s1, s2), kw)
            yield (f"frechet[{src_id}][{kw_id}]",
                   "tslearn.metrics.frechet", (s1, s2), kw)
            yield (f"dtw_path[{src_id}][{kw_id}]",
                   "tslearn.metrics.dtw_path", (s1, s2), kw)

    # LCSS + unnormalized_gak + lb_keogh + CTW: untested public metric
    # APIs that share dispatch plumbing with the covered metrics. LCSS
    # uses the same constraint kwargs as DTW; ``unnormalized_gak`` is
    # the un-normalized sibling of ``gak``; ``lb_keogh`` is the LB used
    # by the KNN fast path (already exercised end-to-end via
    # ``kneighbors_dtw``, here pinned at the function level too); CTW
    # is iterative so we cap ``max_iter=2`` to keep the harness fast.
    lcss_kwargs = [
        ("default", {}),
        ("sakoe_r1", {"sakoe_chiba_radius": 1}),
        ("sakoe_rfrac", {"sakoe_chiba_radius": 0.5}),
        ("eps0p1", {"eps": 0.1}),
    ]
    for src_id in ("small_1d", "small_2d", "midnan", "trailing_nan"):
        ds = by_id[src_id]
        if ds.shape[0] < 2:
            continue
        s1, s2 = ds[0], ds[1]
        for kw_id, kw in lcss_kwargs:
            yield (f"lcss[{src_id}][{kw_id}]",
                   "tslearn.metrics.lcss", (s1, s2), kw)
            yield (f"lcss_path[{src_id}][{kw_id}]",
                   "tslearn.metrics.lcss_path", (s1, s2), kw)
        for sigma in (1.0, np.nan):
            yield (f"unnormalized_gak[{src_id}][sigma={sigma}]",
                   "tslearn.metrics.unnormalized_gak",
                   (s1, s2), {"sigma": sigma})
        for radius in (1, 0.5, float("inf")):
            yield (f"lb_keogh[{src_id}][radius={radius}]",
                   "tslearn.metrics.lb_keogh",
                   (s1, s2), {"radius": radius})
    # CTW is iterative (CCA + DTW alternation) and far slower per call
    # than the other metrics here. Pin the dispatch with a single
    # finite single-pair case rather than walking the full matrix -
    # CTW's correctness path runs through the same DTW kernels already
    # exhaustively covered, so this case is for surface coverage only.
    sp_ctw = by_id["small_1d"]
    yield ("ctw[small_1d][max_iter=2]",
           "tslearn.metrics.ctw", (sp_ctw[0], sp_ctw[1]),
           {"max_iter": 2})

    # TimeSeriesResampler.fit_transform — same edge-case datasets;
    # several target sizes including the size-1 special case.
    for ds_id, ds in DATASETS:
        if ds.shape[1] == 0:
            continue
        for sz in (1, 2, 5):
            yield (f"resampler[{ds_id}][sz={sz}]",
                   "local:run_resampler",
                   (ds, sz), {})

    # Scalers + piecewise transforms. Walks the same dataset matrix
    # through ``TimeSeriesScalerMinMax``, ``TimeSeriesScalerMeanVariance``,
    # ``PiecewiseAggregateApproximation``, and
    # ``SymbolicAggregateApproximation``. Skips zero-length sequences
    # (``shape[1] == 0``) since the public APIs don't define a result
    # for them. Toggles ``per_timeseries`` / ``per_feature`` to exercise
    # both the per-row stats path and the global-stats reduction path.
    for ds_id, ds in DATASETS:
        if ds.shape[1] == 0:
            continue
        for per_ts in (True, False):
            yield (f"scaler_minmax[{ds_id}][per_ts={per_ts}]",
                   "local:run_scaler_minmax",
                   (ds, (0.0, 1.0), per_ts, True), {})
            yield (f"scaler_meanvar[{ds_id}][per_ts={per_ts}]",
                   "local:run_scaler_meanvar",
                   (ds, 0.0, 1.0, per_ts, True), {})
        for n_seg in (1, 2):
            yield (f"paa[{ds_id}][n_segments={n_seg}]",
                   "local:run_paa", (ds, n_seg), {})
            for scale in (True, False):
                yield (f"sax[{ds_id}][n_segments={n_seg}][scale={scale}]",
                       "local:run_sax", (ds, n_seg, 4, scale), {})

    # TimeSeriesImputer.fit_transform — exercises every supported
    # imputation strategy across the edge-case dataset matrix. NaN /
    # Inf / trailing-NaN datasets are the interesting ones; finite
    # datasets just confirm that imputation is a no-op when there's
    # nothing to fill. Toggles ``keep_trailing_nans`` for the all-NaN-
    # padded case, where the two settings produce different results.
    for ds_id, ds in DATASETS:
        if ds.shape[1] == 0:
            continue
        for method in ("mean", "median", "ffill", "bfill",
                       "linear", "constant"):
            value = 0.0 if method == "constant" else float("nan")
            yield (f"imputer[{ds_id}][method={method}]",
                   "local:run_imputer", (ds, method, value, True), {})
        # ``keep_trailing_nans=False`` only changes behavior for inputs
        # whose padding contains all-NaN trailing rows. Pin both
        # settings on ``all_nan_padded`` to catch any drift in the
        # trailing-NaN detection.
        if ds_id == "all_nan_padded":
            yield ("imputer[all_nan_padded][method=ffill,trail=False]",
                   "local:run_imputer",
                   (ds, "ffill", float("nan"), False), {})

    # MatrixProfile - univariate only. ``MatrixProfile`` operates on a
    # single series at a time, so slice the dataset down to row 0 here
    # rather than in the runner helper.
    for ds_id, ds in DATASETS:
        if ds.shape[0] == 0 or ds.shape[1] < 4 or ds.shape[2] != 1:
            continue
        for scale in (True, False):
            for sub_len in (3,):
                yield (f"matrix_profile[{ds_id}][scale={scale}][m={sub_len}]",
                       "local:run_matrix_profile",
                       (ds[:1], sub_len, scale), {})

    # End-to-end estimator parity. These exercise the user-facing code
    # paths the metric harness above can't reach: kNN's metric-dispatch
    # plumbing, clustering estimators' assignment/update loops, SVC's GAK
    # self-diagonal caching, and softdtw_barycenter's L-BFGS objective loop.
    rng = np.random.RandomState(7)

    # Category 1: estimator surface beyond KNN-DTW + SVC-GAK. Run the
    # existing edge-case dataset matrix through the public clustering
    # estimators that this package exposes. ``TimeSeriesKMeans`` has DTW
    # and Soft-DTW metrics; GAK clustering is exposed as ``KernelKMeans``.
    # There is no ``TimeSeriesKMedoids`` class in this repo to exercise.
    for ds_id, ds in DATASETS:
        for kw_id, metric_params in [
            ("dtw_default", {}),
            ("dtw_sakoe_r1", {"sakoe_chiba_radius": 1}),
            ("dtw_sakoe_rfrac", {"sakoe_chiba_radius": 0.5}),
            ("dtw_sakoe_rinf", {"sakoe_chiba_radius": float("inf")}),
        ]:
            yield (f"ts_kmeans[{ds_id}][{kw_id}]",
                   "local:run_timeseries_kmeans",
                   (ds, "dtw", metric_params), {})
        for gamma in (1.0, np.nan):
            yield (f"ts_kmeans[{ds_id}][softdtw_gamma={gamma}]",
                   "local:run_timeseries_kmeans",
                   (ds, "softdtw", {"gamma": gamma}), {})
        for sigma in (1.0, np.nan):
            yield (f"kernel_kmeans_gak[{ds_id}][sigma={sigma}]",
                   "local:run_kernel_kmeans_gak",
                   (ds, sigma), {})
        yield (f"kshape[{ds_id}]",
               "local:run_kshape",
               (ds,), {})

    knn_train = rng.randn(8, 12, 1).astype(np.float64)
    knn_test = rng.randn(4, 12, 1).astype(np.float64)
    yield ("kneighbors_dtw[radius=2,k=3]",
           "local:run_kneighbors_dtw",
           (knn_train, knn_test, 3, 2), {})
    # Fractional / non-finite radius: ``_sakoe_radius_for_fast_path``
    # declines, KNN falls back to the legacy mask path (matching main).
    # Without the gate, downstream ``int(radius)`` would truncate 0.5 to
    # 0 and surface OverflowError on inf.
    yield ("kneighbors_dtw[radius=0.5,k=3]",
           "local:run_kneighbors_dtw",
           (knn_train, knn_test, 3, 0.5), {})
    yield ("kneighbors_dtw[radius=inf,k=3]",
           "local:run_kneighbors_dtw",
           (knn_train, knn_test, 3, float("inf")), {})

    svc_X = rng.randn(12, 10, 1).astype(np.float64)
    svc_y = numpy_array_of_classes(svc_X.shape[0])
    svc_test = rng.randn(4, 10, 1).astype(np.float64)
    yield ("svc_gak_predict[gamma=1.0]",
           "local:run_svc_gak_predict",
           (svc_X, svc_y, svc_test, 1.0), {})

    # Regression sibling of the SVC case. The kernel + self-diagonal
    # plumbing is shared between ``TimeSeriesSVC`` and
    # ``TimeSeriesSVR``; this pins the regressor predict path against
    # a continuous target so a future SVC-only fix can't silently
    # break SVR.
    svr_y = rng.randn(svc_X.shape[0]).astype(np.float64)
    yield ("svr_gak_predict[gamma=1.0]",
           "local:run_svr_gak_predict",
           (svc_X, svr_y, svc_test, 1.0), {})

    bary_X = rng.randn(5, 8, 1).astype(np.float64)
    yield ("softdtw_barycenter[gamma=1.0,max_iter=10]",
           "local:run_softdtw_barycenter",
           (bary_X, 1.0, 10), {})

    # Non-finite gamma: main raises (ZeroDivisionError for nan via the
    # ``_softdtw_func`` divide; ValueError for inf via finite-input check
    # on the resulting NaN obj). The fast-path gate routes both through
    # the legacy path so the exception types match.
    yield ("softdtw_barycenter[gamma=nan,max_iter=1]",
           "local:run_softdtw_barycenter",
           (bary_X, float("nan"), 1), {})
    yield ("softdtw_barycenter[gamma=inf,max_iter=1]",
           "local:run_softdtw_barycenter",
           (bary_X, float("inf"), 1), {})

    # max_iter=0 short-circuits to ``init`` unchanged on main, even when
    # the input contains zero-length / all-NaN members. The fast-path's
    # eager empty-member rejection is gated on ``max_iter > 0`` so this
    # case stops being a regression.
    bary_empty = np.array([[[np.nan]], [[1.0]]], dtype=np.float64)
    bary_init = np.array([[0.0]], dtype=np.float64)
    yield ("softdtw_barycenter[max_iter=0,all_nan_member]",
           "local:run_softdtw_barycenter",
           (bary_empty, 1.0, 0), {"init": bary_init})


def numpy_array_of_classes(n):
    """Two-class label vector with deterministic balance for SVC tests."""
    return np.array([i % 2 for i in range(n)], dtype=np.int64)


# ---------------------------------------------------------------------------
# Subprocess runner against `main`
# ---------------------------------------------------------------------------

def run_on_main(cases, main_dir: Path) -> dict:
    """Run all cases under ``PYTHONPATH=main_dir`` in one subprocess."""
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as fin:
        pickle.dump(cases, fin)
        in_path = fin.name
    out_path = in_path + ".out"

    env = os.environ.copy()
    # Put the worktree root first on PYTHONPATH so it shadows any editable
    # install of HEAD in site-packages.
    extra = str(main_dir)
    env["PYTHONPATH"] = (
        extra + os.pathsep + env.get("PYTHONPATH", "")
    ).rstrip(os.pathsep)

    completed = subprocess.run(
        [sys.executable, str(RUNNER), in_path, out_path],
        env=env,
        cwd=str(main_dir),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"main runner failed (rc={completed.returncode}):\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )

    with open(out_path, "rb") as f:
        results = pickle.load(f)

    os.unlink(in_path)
    os.unlink(out_path)
    return dict(results)


def run_on_head(cases) -> dict:
    """Run all cases inline against HEAD (this process).

    Delegates the resolve+invoke+capture to the runner module's
    ``run_one`` so the result-shape contract has a single source.
    """
    from tests._main_parity_runner import run_one
    return {
        cid: run_one(fn, args, kw) for cid, fn, args, kw in cases
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _values_equal(a, b):
    """Equality with NaN-equal-NaN semantics for scalars and small
    containers. Mirrors ``np.allclose(..., equal_nan=True)`` but works
    on tuples (``dtw_path`` returns ``(path, dist)``) and Python floats."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            return False
        if a.shape != b.shape:
            return False
        if a.size == 0:
            return True
        with np.errstate(invalid="ignore"):
            return bool(np.allclose(
                a, b, atol=_ATOL, rtol=_RTOL, equal_nan=True,
            ))
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        if type(a) is not type(b) or len(a) != len(b):
            return False
        return all(_values_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
        with np.errstate(invalid="ignore"):
            return bool(np.allclose(
                a, b, atol=_ATOL, rtol=_RTOL, equal_nan=True,
            ))
    return a == b


def compare(main_r: dict, head_r: dict) -> str | None:
    """Return ``None`` if equivalent, else a diff string."""
    if main_r["status"] != head_r["status"]:
        return (f"status differs (main={main_r['status']}, "
                f"head={head_r['status']}); "
                f"main.msg={main_r.get('msg', '')!r}; "
                f"head.msg={head_r.get('msg', '')!r}")
    if main_r["status"] == "err":
        if main_r["type"] != head_r["type"]:
            return (f"exception type differs "
                    f"(main={main_r['type']}, head={head_r['type']}); "
                    f"head.msg={head_r['msg']!r}")
        return None
    a, b = main_r["result"], head_r["result"]
    if _values_equal(a, b):
        return None
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray) \
            and a.shape == b.shape and a.size > 0:
        diff_mask = ~(
            (np.isnan(a) & np.isnan(b))
            | (np.isfinite(a) & np.isfinite(b)
               & (np.abs(a - b) <= _ATOL + _RTOL * np.abs(b)))
        )
        idx = np.argwhere(diff_mask)
        if len(idx) > 0:
            i = tuple(idx[0])
            return (f"values differ at {i}: "
                    f"main={a[i]!r} head={b[i]!r} "
                    f"(total {diff_mask.sum()}/{a.size} cells differ)")
    return f"values differ (main={a!r} head={b!r})"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--main-rev", default="main",
                        help="Git rev to compare against (default: main).")
    parser.add_argument("--limit", type=int, default=None,
                        help="If set, only run the first N cases. Useful "
                             "for smoke-testing the harness.")
    parser.add_argument("--show-passing", action="store_true",
                        help="Print every case, not just mismatches.")
    args = parser.parse_args(argv)

    try:
        main_dir = ensure_worktree(args.main_rev)
    except subprocess.CalledProcessError as e:
        sys.stderr.write(
            f"Failed to create worktree at {WORKTREE_DIR} for {args.main_rev}\n"
            f"stdout: {e.stdout}\nstderr: {e.stderr}\n"
        )
        return 2

    cases = list(iter_cases())
    if args.limit is not None:
        cases = cases[: args.limit]

    print(f"Running {len(cases)} cases on main ({args.main_rev}) at {main_dir}")
    main_results = run_on_main(cases, main_dir)
    print(f"Running {len(cases)} cases on HEAD")
    head_results = run_on_head(cases)

    n_ok = 0
    drifts = []
    expected = []
    for case_id, _func, _args, _kwargs in cases:
        m = main_results.get(case_id)
        h = head_results.get(case_id)
        if m is None or h is None:
            diff = "missing result"
        else:
            diff = compare(m, h)
        if diff is None:
            n_ok += 1
            if args.show_passing:
                print(f"  OK   {case_id}")
            continue
        reason = _expected_drift_reason(case_id)
        if reason is not None:
            expected.append((case_id, diff, reason))
        else:
            drifts.append((case_id, diff))

    print()
    print(f"Summary: {n_ok} match, {len(drifts)} drift, "
          f"{len(expected)} expected drift, {len(cases)} total")
    if expected and args.show_passing:
        print()
        print("Expected drifts (allowlisted):")
        for case_id, diff, reason in expected:
            print(f"  {case_id}  ({reason})")
            print(f"    {diff}")
    if drifts:
        print()
        print("Drifts:")
        for case_id, diff in drifts:
            print(f"  {case_id}")
            print(f"    {diff}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
