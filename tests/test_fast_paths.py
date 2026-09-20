"""Parity tests for the numpy-backend fast-path kernels.

These tests pin equivalence between the accelerated kernels (band-iteration,
LB-prune, STOMP, etc.) and the legacy generic implementation across a matrix
of input shapes:

- empty datasets (zero series, zero-length series)
- all-NaN padded series (variable-length representation)
- variable-length series within a dataset
- univariate AND multivariate
- self-similarity (``dataset2=None``) AND cross-dataset
- Sakoe-Chiba, Itakura, and unconstrained DTW

The point of these tests isn't to re-validate the math — it's to catch
fast-path drift after future changes (a fast path that quietly disagrees
with the slow path on, say, all-NaN rows is the worst kind of regression).
"""
import numpy as np
import pytest

from tslearn.metrics import (
    cdist_dtw,
    cdist_frechet,
    cdist_gak,
    cdist_soft_dtw,
    dtw_path,
    frechet,
    gak,
    sigma_gak,
    soft_dtw,
)
from tslearn.metrics._dtw import _njit_dtw, _njit_dtw_path
from tslearn.metrics._dtw_lb import cdist_dtw_topk_fast
from tslearn.metrics._frechet import _njit_frechet
from tslearn.metrics._masks import GLOBAL_CONSTRAINT_CODE
from tslearn.metrics.utils import _cdist_generic
from tslearn.backend import instantiate_backend
from tslearn.matrix_profile import MatrixProfile
from tslearn.utils import to_time_series_dataset

NUMPY_BE = instantiate_backend("numpy")
ATOL = 1e-9

SAKOE_R3 = pytest.param(
    {"global_constraint": "sakoe_chiba", "sakoe_chiba_radius": 3},
    id="sakoe_r3",
)
SAKOE_R2 = pytest.param(
    {"global_constraint": "sakoe_chiba", "sakoe_chiba_radius": 2},
    id="sakoe_r2",
)
ITAKURA_S2 = pytest.param(
    {"global_constraint": "itakura", "itakura_max_slope": 2.0},
    id="itakura_s2",
)
UNCONSTRAINED = pytest.param({}, id="unconstrained")


def _constraint_kwargs(d):
    """Map a public-API constraint dict (with string ``global_constraint``)
    to the int-coded form expected by the ``_njit_dtw`` / ``_njit_frechet``
    per-pair entry points used in our slow-path references."""
    return dict(
        global_constraint=GLOBAL_CONSTRAINT_CODE[d.get("global_constraint")],
        sakoe_chiba_radius=d.get("sakoe_chiba_radius"),
        itakura_max_slope=d.get("itakura_max_slope"),
    )


def _slow_cdist_dtw(X, Y, **mp):
    """Run cdist_dtw via the generic per-pair path, bypassing cdist_dtw_fast."""
    return _cdist_generic(
        dist_fun=_njit_dtw,
        dataset1=X,
        dataset2=Y,
        n_jobs=None,
        verbose=0,
        be=NUMPY_BE,
        compute_diagonal=False,
        **mp,
    )


# ------------------------------ cdist_dtw -----------------------------------


@pytest.mark.parametrize("d", [1, 5])
@pytest.mark.parametrize("constraint", [UNCONSTRAINED, SAKOE_R3, ITAKURA_S2])
def test_cdist_dtw_self_parity(d, constraint):
    rng = np.random.RandomState(0)
    X = rng.randn(8, 30, d).astype(np.float64)

    fast = cdist_dtw(X, None, **constraint)
    slow = _slow_cdist_dtw(X, None, **_constraint_kwargs(constraint))

    np.testing.assert_allclose(fast, slow, atol=ATOL)
    # Self-similarity invariants: zero diagonal, symmetric.
    np.testing.assert_array_equal(np.diag(fast), np.zeros(8))
    np.testing.assert_allclose(fast, fast.T, atol=ATOL)


@pytest.mark.parametrize("d", [1, 5])
@pytest.mark.parametrize("constraint", [UNCONSTRAINED, SAKOE_R3, ITAKURA_S2])
def test_cdist_dtw_cross_parity(d, constraint):
    rng = np.random.RandomState(1)
    X = rng.randn(6, 30, d).astype(np.float64)
    Y = rng.randn(4, 30, d).astype(np.float64)

    fast = cdist_dtw(X, Y, **constraint)
    slow = _slow_cdist_dtw(X, Y, **_constraint_kwargs(constraint))

    np.testing.assert_allclose(fast, slow, atol=ATOL)


@pytest.mark.parametrize(
    "constraint",
    # Itakura is included even though the fast path declines on
    # variable-length and falls through to the same slow reference; the
    # parametrize covers the dispatcher's None-return wiring.
    [UNCONSTRAINED, SAKOE_R2, ITAKURA_S2],
)
def test_cdist_dtw_variable_length_parity(constraint):
    rng = np.random.RandomState(2)
    X = to_time_series_dataset(
        [rng.randn(L, 1).astype(np.float64) for L in (10, 14, 12)]
    )
    Y = to_time_series_dataset(
        [rng.randn(L, 1).astype(np.float64) for L in (12, 9)]
    )

    fast = cdist_dtw(X, Y, **constraint)
    slow = _slow_cdist_dtw(X, Y, **_constraint_kwargs(constraint))
    np.testing.assert_allclose(fast, slow, atol=ATOL)


def test_cdist_dtw_empty_datasets():
    X = np.zeros((0, 10, 2), dtype=np.float64)
    Y = np.random.RandomState(0).randn(3, 10, 2).astype(np.float64)

    out = cdist_dtw(X, Y)
    assert out.shape == (0, 3)
    out = cdist_dtw(Y, X)
    assert out.shape == (3, 0)
    out = cdist_dtw(X, X)
    assert out.shape == (0, 0)
    # Self-similarity with zero rows.
    out = cdist_dtw(X, None)
    assert out.shape == (0, 0)


def test_cdist_dtw_and_frechet_reject_feature_dimension_mismatch():
    X = np.zeros((2, 3, 1), dtype=np.float64)
    Y = np.zeros((2, 3, 2), dtype=np.float64)
    for fn in (cdist_dtw, cdist_frechet):
        with pytest.raises(ValueError, match="same feature size"):
            fn(X, Y)
        with pytest.raises(ValueError, match="same feature size"):
            fn(Y, X)

    empty = np.empty((0, 3, 1), dtype=np.float64)
    for fn in (cdist_dtw, cdist_frechet):
        assert fn(empty, Y).shape == (0, 2)
        assert fn(Y, empty).shape == (2, 0)


def test_cdist_dtw_all_nan_row():
    # A row that's entirely NaN trims to valid length 0; the slow path
    # represents that pair distance as +inf (one side empty). Fast path
    # must agree.
    rng = np.random.RandomState(3)
    X = rng.randn(4, 10, 1).astype(np.float64)
    X[2, :, :] = np.nan  # row 2 has zero valid length

    fast = cdist_dtw(X, X)
    slow = _slow_cdist_dtw(X, X)
    # Both should agree (within float tolerance) including the inf
    # rows/columns. assert_allclose treats inf == inf as equal.
    np.testing.assert_allclose(fast, slow, atol=ATOL)
    # Sanity: row 2 vs itself is treated as 0 (both empty).
    assert fast[2, 2] == 0.0
    # Row 2 vs a non-empty row is +inf.
    assert np.isinf(fast[2, 0])


def test_cdist_dtw_itakura_zero_length_falls_back_to_mask_errors():
    # Itakura mask construction on zero-valid-length series raises on main.
    # The fast path must decline these inputs instead of returning 0/inf.
    all_empty = np.array([[[np.nan]], [[np.nan]]], dtype=np.float64)
    non_empty = np.array([[[1.0]], [[2.0]]], dtype=np.float64)
    kwargs = {"global_constraint": "itakura"}

    with pytest.raises(ZeroDivisionError):
        cdist_dtw(all_empty, **kwargs)
    with pytest.raises(ZeroDivisionError):
        cdist_dtw(all_empty, all_empty, **kwargs)
    with pytest.raises(RuntimeWarning, match="unfeasible"):
        cdist_dtw(all_empty, non_empty, **kwargs)
    with pytest.raises(ZeroDivisionError):
        cdist_dtw(non_empty, all_empty, **kwargs)


# ----------------------- cdist_dtw_topk_fast (LB) ---------------------------


@pytest.mark.parametrize("d", [1, 5])
def test_cdist_dtw_topk_matches_full_argsort(d):
    # The LB top-k fast path must return the same top-k by argsort as a
    # full cdist_dtw scan with the same Sakoe-Chiba radius.
    rng = np.random.RandomState(4)
    X = rng.randn(6, 40, d).astype(np.float64)
    Y = rng.randn(20, 40, d).astype(np.float64)
    radius = 4
    k = 3

    full = cdist_dtw(
        X, Y, global_constraint="sakoe_chiba", sakoe_chiba_radius=radius
    )
    ref_d = np.sort(full, axis=1)[:, :k]

    fast = cdist_dtw_topk_fast(X, Y, k=k, radius=radius)
    assert fast is not None
    fast_d, fast_i = fast
    # Compare distances after sort (avoids fragility on argsort tie order).
    np.testing.assert_allclose(fast_d, ref_d, atol=ATOL)
    # Each returned index must point to a candidate whose actual distance
    # equals the reported one — i.e. the (idx, distance) mapping is sound.
    actual = np.take_along_axis(full, fast_i, axis=1)
    np.testing.assert_allclose(actual, fast_d, atol=ATOL)


def test_cdist_dtw_topk_declines_when_inputs_unsupported():
    rng = np.random.RandomState(5)
    X = to_time_series_dataset(
        [rng.randn(L, 1).astype(np.float64) for L in (12, 10, 14)]
    )
    Y = to_time_series_dataset(
        [rng.randn(L, 1).astype(np.float64) for L in (12, 12)]
    )
    # Variable-length within a dataset → fast path returns None.
    assert cdist_dtw_topk_fast(X, Y, k=2, radius=3) is None

    # No radius (radius is None or negative) → fast path declines.
    Z = rng.randn(4, 12, 1).astype(np.float64)
    assert cdist_dtw_topk_fast(Z, Z, k=2, radius=None) is None


def test_cdist_dtw_topk_empty_query_and_oversize_k():
    # Empty query → output shapes propagate; dtypes match the kernel contract.
    Y = np.random.RandomState(6).randn(5, 12, 1).astype(np.float64)
    X = np.zeros((0, 12, 1), dtype=np.float64)
    fast = cdist_dtw_topk_fast(X, Y, k=2, radius=3)
    assert fast is not None
    d, i = fast
    assert d.shape == (0, 2) and i.shape == (0, 2)
    assert d.dtype == np.float64 and i.dtype == np.int64

    # k > n_candidates: kernel must not raise; rows are padded with +inf / -1
    # for the unused slots. This protects against future k_eff clamp drift.
    X2 = np.random.RandomState(7).randn(2, 12, 1).astype(np.float64)
    fast = cdist_dtw_topk_fast(X2, Y, k=10, radius=3)
    assert fast is not None
    d, i = fast
    assert d.shape == (2, 10) and i.shape == (2, 10)
    # First 5 slots are real candidates, last 5 are pad sentinels.
    assert (i[:, :5] >= 0).all() and (i[:, 5:] == -1).all()
    assert np.isfinite(d[:, :5]).all() and np.isinf(d[:, 5:]).all()


# ------------------------------ cdist_soft_dtw -------------------------------


@pytest.mark.parametrize("d", [1, 4])
@pytest.mark.parametrize("self_sim", [True, False])
def test_cdist_soft_dtw_parity(d, self_sim):
    # Compare the parallel kernel against single-pair soft_dtw aggregated
    # in a Python double loop. Narrow problem so the loop is tractable.
    rng = np.random.RandomState(7)
    X = rng.randn(5, 16, d).astype(np.float64)
    Y = X if self_sim else rng.randn(4, 16, d).astype(np.float64)

    fast = cdist_soft_dtw(X, None if self_sim else Y, gamma=0.5)
    n1 = X.shape[0]
    n2 = X.shape[0] if self_sim else Y.shape[0]
    slow = np.empty((n1, n2), dtype=np.float64)
    Y_ref = X if self_sim else Y
    for i in range(n1):
        for j in range(n2):
            slow[i, j] = soft_dtw(X[i], Y_ref[j], gamma=0.5)
    np.testing.assert_allclose(fast, slow, atol=ATOL)


def test_cdist_soft_dtw_empty():
    X = np.zeros((0, 8, 2), dtype=np.float64)
    Y = np.random.RandomState(8).randn(3, 8, 2).astype(np.float64)
    out = cdist_soft_dtw(X, Y, gamma=1.0)
    assert out.shape == (0, 3)
    out = cdist_soft_dtw(X, None, gamma=1.0)
    assert out.shape == (0, 0)


def test_cdist_soft_dtw_rejects_feature_dimension_mismatch():
    from tslearn.metrics import cdist_soft_dtw_normalized

    X = np.zeros((2, 3, 1), dtype=np.float64)
    Y = np.zeros((2, 3, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="Incompatible dimension"):
        cdist_soft_dtw(X, Y, gamma=1.0)
    with pytest.raises(ValueError, match="Incompatible dimension"):
        cdist_soft_dtw_normalized(X, Y, gamma=1.0)

    empty = np.empty((0, 3, 1), dtype=np.float64)
    assert cdist_soft_dtw(empty, Y, gamma=1.0).shape == (0, 2)
    assert cdist_soft_dtw(Y, empty, gamma=1.0).shape == (2, 0)
    assert cdist_soft_dtw_normalized(empty, Y, gamma=1.0).shape == (0, 2)
    assert cdist_soft_dtw_normalized(Y, empty, gamma=1.0).shape == (2, 0)


# ------------------------------- cdist_gak ----------------------------------


@pytest.mark.parametrize("d", [1, 3])
@pytest.mark.parametrize("self_sim", [True, False])
def test_cdist_gak_parity(d, self_sim):
    # Reference: per-pair gak() in a Python loop.
    rng = np.random.RandomState(9)
    X = rng.randn(5, 14, d).astype(np.float64)
    Y = X if self_sim else rng.randn(4, 14, d).astype(np.float64)
    sigma = sigma_gak(X)

    fast = cdist_gak(X, None if self_sim else Y, sigma=sigma)
    n1 = X.shape[0]
    n2 = X.shape[0] if self_sim else Y.shape[0]
    slow = np.empty((n1, n2), dtype=np.float64)
    Y_ref = X if self_sim else Y
    for i in range(n1):
        for j in range(n2):
            slow[i, j] = gak(X[i], Y_ref[j], sigma=sigma)
    np.testing.assert_allclose(fast, slow, atol=ATOL)


def test_cdist_gak_preserves_nan_normalization():
    # Main propagates NaN across every column via ``diag(l) @ M @ diag(r)``;
    # the elementwise fast form leaves cells finite where the cell and its
    # scaling factors all are.
    a = [[0.0], [np.nan], [1.0]]
    b = [[0.0], [1.0], [2.0]]
    c = [[2.0], [3.0], [4.0]]
    assert np.isnan(cdist_gak([a, b], [b, c], sigma=2.0)).all()
    assert np.isnan(cdist_gak([a, b], sigma=2.0)).all()


def test_cdist_gak_rejects_feature_dimension_mismatch():
    X = np.zeros((2, 3, 1), dtype=np.float64)
    Y = np.zeros((2, 3, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="same number of columns"):
        cdist_gak(X, Y, sigma=1.0)

    empty = np.empty((0, 3, 1), dtype=np.float64)
    assert cdist_gak(empty, Y, sigma=1.0).shape == (0, 2)
    assert cdist_gak(Y, empty, sigma=1.0).shape == (2, 0)


# ------------------------------ cdist_frechet -------------------------------


def _slow_cdist_frechet(X, Y, **mp):
    return _cdist_generic(
        dist_fun=_njit_frechet,
        dataset1=X,
        dataset2=Y,
        n_jobs=None,
        verbose=0,
        be=NUMPY_BE,
        compute_diagonal=False,
        **mp,
    )


@pytest.mark.parametrize("d", [1, 4])
@pytest.mark.parametrize("constraint", [UNCONSTRAINED, SAKOE_R2])
@pytest.mark.parametrize("self_sim", [True, False])
def test_cdist_frechet_parity(d, constraint, self_sim):
    rng = np.random.RandomState(10)
    X = rng.randn(5, 18, d).astype(np.float64)
    Y = X if self_sim else rng.randn(4, 18, d).astype(np.float64)

    fast = cdist_frechet(X, None if self_sim else Y, **constraint)
    slow = _slow_cdist_frechet(
        X, X if self_sim else Y, **_constraint_kwargs(constraint)
    )
    np.testing.assert_allclose(fast, slow, atol=ATOL)


# ------------------------------- STOMP MP -----------------------------------


def _naive_matrix_profile(x, m, scale, band):
    # ``band`` must mirror MatrixProfile's exclusion zone (currently
    # ``ceil(m / 4)``). If the transformer's band changes, callers must
    # update theirs to match or the parity assertions will fail.
    sz, d = x.shape
    n = sz - m + 1
    segs = np.empty((n, m, d), dtype=np.float64)
    for i in range(n):
        segs[i] = x[i:i + m]
    if scale:
        mu = segs.mean(axis=1, keepdims=True)
        sd = segs.std(axis=1, keepdims=True)
        sd[sd == 0] = 1.0
        segs = (segs - mu) / sd
    out = np.full(n, np.inf)
    for i in range(n):
        for j in range(n):
            if abs(i - j) <= band:
                continue
            diff = segs[i] - segs[j]
            d_ = np.sqrt((diff * diff).sum())
            if d_ < out[i]:
                out[i] = d_
    return out


# --------------------- constraint-gating regressions -----------------------


def test_is_sakoe_chiba_only_predicate():
    from tslearn.metrics.utils import _is_sakoe_chiba_only

    # Explicit Sakoe wins, even when an Itakura slope is also passed.
    assert _is_sakoe_chiba_only("sakoe_chiba", 3, None)
    assert _is_sakoe_chiba_only("sakoe_chiba", 3, 2.0)
    assert not _is_sakoe_chiba_only("sakoe_chiba", None, None)

    # Explicit Itakura always wins; a stale radius must not fool us.
    assert not _is_sakoe_chiba_only("itakura", None, 2.0)
    assert not _is_sakoe_chiba_only("itakura", 5, 2.0)
    assert not _is_sakoe_chiba_only("itakura", 5, None)

    # Unset global_constraint: Sakoe radius alone is OK.
    assert _is_sakoe_chiba_only(None, 3, None)
    assert _is_sakoe_chiba_only("", 3, None)
    # Both set with no explicit constraint = ambiguous → fall through.
    assert not _is_sakoe_chiba_only(None, 3, 2.0)
    # Itakura slope alone is not Sakoe.
    assert not _is_sakoe_chiba_only(None, None, 2.0)
    # Nothing set is not Sakoe.
    assert not _is_sakoe_chiba_only(None, None, None)

    # Int-coded global_constraint is also accepted (as cdist_dtw_fast
    # routes through with codes).
    assert _is_sakoe_chiba_only(2, 3, None)            # SAKOE_CHIBA
    assert not _is_sakoe_chiba_only(1, 3, None)        # ITAKURA
    assert _is_sakoe_chiba_only(0, 3, None)            # NO_CONSTRAINT


def test_dtw_kneighbors_falls_through_for_ambiguous_constraint():
    # Regression: the LB-prune fast path used to silently return Sakoe
    # neighbors whenever a sakoe_chiba_radius was present, even when
    # itakura_max_slope was also passed (which the slow cdist raises on).
    from tslearn.neighbors import KNeighborsTimeSeries

    rng = np.random.RandomState(0)
    X = rng.randn(8, 12, 1).astype(np.float64)

    knn = KNeighborsTimeSeries(
        n_neighbors=2, metric="dtw",
        metric_params={
            "sakoe_chiba_radius": 2,
            "itakura_max_slope": 2.0,
        },
    ).fit(X)
    # The fall-through must surface the same RuntimeWarning the
    # surface-level cdist_dtw raises on this input.
    with pytest.raises(RuntimeWarning, match="global_constraint is not set"):
        knn.kneighbors(X)


def test_dtw_kneighbors_honors_explicit_itakura():
    # Regression: explicit global_constraint="itakura" must not be
    # silently overridden by a stale sakoe_chiba_radius in metric_params.
    from tslearn.neighbors import KNeighborsTimeSeries

    rng = np.random.RandomState(1)
    X = rng.randn(6, 16, 1).astype(np.float64)

    # Reference: full cdist with explicit Itakura.
    D_itakura = cdist_dtw(
        X, X, global_constraint="itakura", itakura_max_slope=2.0,
    )
    ref_idx = np.argsort(D_itakura, axis=1)[:, :2]

    knn = KNeighborsTimeSeries(
        n_neighbors=2, metric="dtw",
        metric_params={
            "global_constraint": "itakura",
            "itakura_max_slope": 2.0,
            "sakoe_chiba_radius": 5,  # stale — must be ignored.
        },
    ).fit(X)
    fast_idx = knn.kneighbors(X, return_distance=False)

    # The neighbors must match the Itakura reference. (Sakoe with r=5 on
    # length-16 series produces a noticeably different ordering on this
    # seed, so a silent fallback would fail this assertion.)
    np.testing.assert_array_equal(fast_idx, ref_idx)


def test_kmeans_dtw_assign_falls_through_for_ambiguous_constraint():
    # Regression: KMeans._assign used to call the LB-prune top-1 with
    # only sakoe_chiba_radius, ignoring Itakura — flagged in review for
    # producing different labels vs _transform(...).argmin(axis=1) on
    # ambiguous params.
    from tslearn.clustering import TimeSeriesKMeans

    rng = np.random.RandomState(2)
    X = rng.randn(12, 16, 1).astype(np.float64)
    km = TimeSeriesKMeans(
        n_clusters=3, metric="dtw", n_init=1, max_iter=1,
        random_state=0,
        metric_params={
            "sakoe_chiba_radius": 2,
            "itakura_max_slope": 2.0,
        },
    )
    with pytest.raises(RuntimeWarning, match="global_constraint is not set"):
        km.fit(X)


def test_dba_path_dispatch_honors_explicit_itakura():
    # Two related regressions in `_dtw_path_dispatch`:
    #   (a) the Sakoe band-iteration kernel was chosen solely from
    #       sakoe_chiba_radius, so explicit Itakura + a stale radius
    #       silently produced a Sakoe alignment;
    #   (b) on the Itakura fall-through, `metric_params['global_constraint']`
    #       was passed as the public string form straight into the
    #       numba-jitted `_njit_dtw_path`, where the type comparisons
    #       silently degraded to "no constraint" (different distance and
    #       different path length than the public `dtw_path`).
    # The chosen seed + length make (b) observable — picking a length
    # where Itakura actually changes the optimal path.
    from tslearn.barycenters.dba import _dtw_path_dispatch
    from tslearn.metrics import dtw_path

    rng = np.random.RandomState(0)
    s1 = rng.randn(30, 1).astype(np.float64)
    s2 = rng.randn(30, 1).astype(np.float64)

    ref_path, ref_dist = dtw_path(
        s1, s2,
        global_constraint="itakura", itakura_max_slope=2.0,
    )

    # Stale Sakoe radius (regression a): must be ignored under explicit
    # Itakura. Public-string global_constraint (regression b): must be
    # normalized to the int code before the numba helper.
    disp_dist, disp_path = _dtw_path_dispatch(
        s1, s2,
        global_constraint="itakura",
        itakura_max_slope=2.0,
        sakoe_chiba_radius=4,
    )

    np.testing.assert_allclose(disp_dist, ref_dist, atol=ATOL)
    assert disp_path == ref_path


def test_dtw_path_sakoe_falls_back_on_nan_prefix():
    # The band-iteration Sakoe shortcut has different NaN tie behavior
    # than the legacy mask DP. Public dtw_path accepts these inputs, so
    # it must fall back when a valid prefix contains NaN.
    s1 = np.array([[0.0], [np.nan], [1.0]], dtype=np.float64)
    s2 = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)

    path, dist = dtw_path(s1, s2, sakoe_chiba_radius=1)
    ref_dist, ref_path = _njit_dtw_path(
        s1, s2,
        global_constraint=GLOBAL_CONSTRAINT_CODE[None],
        sakoe_chiba_radius=1,
        itakura_max_slope=None,
    )
    assert path == list(ref_path)
    assert np.isnan(dist) and np.isnan(ref_dist)


# Divergence from main: main forwards the public-string ``global_constraint``
# to a numba-jitted ``_njit_frechet`` whose signature defaults to int 0, so
# string-vs-int comparisons silently fell through to no-constraint and cdist
# computed unconstrained Frechet. We pass the int code now, so cdist matches
# the per-pair ``frechet()`` entrypoint across every constraint variant.
@pytest.mark.parametrize("kwargs", [
    pytest.param(
        {"global_constraint": "itakura", "itakura_max_slope": 2.0},
        id="explicit_itakura",
    ),
    pytest.param({"itakura_max_slope": 2.0}, id="itakura_slope_only"),
    pytest.param({"global_constraint": "itakura"}, id="itakura_default"),
    pytest.param({"sakoe_chiba_radius": 1}, id="sakoe_radius_only"),
    pytest.param({"global_constraint": "sakoe_chiba"}, id="sakoe_default"),
])
def test_cdist_frechet_applies_constraint(kwargs):
    rng = np.random.RandomState(0)
    X = rng.randn(4, 30, 1).astype(np.float64)
    out = cdist_frechet(X, X, **kwargs)
    ref = np.array([
        [frechet(X[i], X[j], **kwargs) for j in range(X.shape[0])]
        for i in range(X.shape[0])
    ])
    np.testing.assert_allclose(out, ref, atol=ATOL)


def test_cdist_frechet_ambiguous_raises():
    # Divergence from main: main silently fell through to no-constraint
    # for ambiguous params (sakoe_chiba_radius + itakura_max_slope and no
    # explicit ``global_constraint``) because the string-vs-int comparison
    # bug bypassed the ambiguity check inside ``_compute_mask``. We pass
    # the int code now, so cdist surfaces the same per-pair raise as the
    # single-pair ``frechet()`` entrypoint. (``_compute_mask`` actually
    # ``raise``s RuntimeWarning, not ``warnings.warn`` — preserving main's
    # quirk of treating the ambiguity as a hard error in the cdist path.)
    rng = np.random.RandomState(1)
    X = rng.randn(4, 6, 1).astype(np.float64)
    with pytest.raises(RuntimeWarning, match="global_constraint is not set"):
        cdist_frechet(X, X, sakoe_chiba_radius=3, itakura_max_slope=2.0)


def test_cdist_soft_dtw_rejects_non_finite_inputs():
    # Regression: the fused kernel doesn't go through SquaredEuclidean,
    # so cdist_soft_dtw silently produced a NaN-only matrix on inputs
    # that the legacy per-pair path raises on. Validate over the valid
    # prefix at the dispatcher.
    rng = np.random.RandomState(0)
    Y = rng.randn(2, 3, 1).astype(np.float64)
    with pytest.raises(ValueError, match="NaN"):
        cdist_soft_dtw(
            [[[0.0], [np.nan], [1.0]]],
            [[[1.0], [2.0], [3.0]]],
        )
    with pytest.raises(ValueError, match="infinity"):
        cdist_soft_dtw([[[np.inf], [1.0], [2.0]]], Y)
    # Trailing NaN (the variable-length sentinel) must still be allowed.
    out = cdist_soft_dtw(
        [[[1.0], [2.0], [np.nan]]], [[[1.0], [2.0], [3.0]]],
    )
    assert out.shape == (1, 1)


def test_cdist_soft_dtw_empty_cross_skips_validation():
    # Regression: ``cdist_soft_dtw([], invalid)`` returns shape (0, n)
    # without inspecting any pair on main; the fast path used to
    # validate the non-empty side first and raise even though no
    # pair would be computed.
    invalid = np.array([[[0.0], [np.nan], [1.0]]], dtype=np.float64)
    empty_left = np.empty((0, 3, 1), dtype=np.float64)
    out = cdist_soft_dtw(empty_left, invalid)
    assert out.shape == (0, 1)
    out = cdist_soft_dtw(invalid, empty_left)
    assert out.shape == (1, 0)


def test_softdtw_barycenter_rejects_non_finite_inputs():
    # Regression: the fused per-i objective contributed 0 value/0
    # gradient for non-finite inputs (the legacy `_softdtw_func` raises
    # via SquaredEuclidean.compute), so softdtw_barycenter silently
    # returned the initial barycenter. Validate trimmed prefixes and
    # the init at the dispatcher.
    from tslearn.barycenters import softdtw_barycenter
    init = np.array([[0.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="NaN"):
        softdtw_barycenter(
            [[[0.0], [np.nan], [1.0]], [[1.0], [2.0], [3.0]]],
            init=init,
        )
    # Non-finite init must also raise.
    bad_init = np.array([[np.nan]], dtype=np.float64)
    with pytest.raises(ValueError):
        softdtw_barycenter([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]], init=bad_init)


def test_softdtw_barycenter_gamma_zero_uses_legacy_error():
    # Zero gamma is accepted by the API surface until the legacy Soft-DTW
    # recursion divides by gamma and raises ZeroDivisionError. The fused
    # objective must not convert that into a numba SystemError.
    from tslearn.barycenters import softdtw_barycenter
    X = np.array([[[0.0], [1.0]], [[1.0], [2.0]]], dtype=np.float64)
    with pytest.raises(ZeroDivisionError):
        softdtw_barycenter(X, gamma=0.0, max_iter=1)


def test_cdist_frechet_preserves_main_nan_ordering():
    # Main's ``max(d^2, min(acc_up, acc_left, acc_diag))`` recurrence is
    # asymmetric under NaN because ``min(NaN, x, y) = NaN`` but
    # ``min(x, NaN, y) = x``. Fast kernels must preserve that argument
    # order across cross + self × unconstrained + Sakoe-Chiba.
    a = [[0.0], [np.nan], [1.0]]
    b = [[0.0], [1.0], [2.0]]
    cross_expected = np.array([[0.0, 1.0], [np.inf, 0.0]])
    self_expected = np.array([[0.0, 1.0], [1.0, 0.0]])
    for kwargs in ({}, {"sakoe_chiba_radius": 1}):
        out = cdist_frechet([a, b], [a, b], **kwargs)
        np.testing.assert_array_equal(out, cross_expected)
        out = cdist_frechet([a, b], **kwargs)
        np.testing.assert_array_equal(out, self_expected)
    # Empty cross-distance with NaN on the other side returns (0, n)
    # without raising, matching main.
    empty = np.empty((0, 3, 1), dtype=np.float64)
    a_arr = np.array([a, b], dtype=np.float64)
    out = cdist_frechet(empty, a_arr)
    assert out.shape == (0, 2)
    out = cdist_frechet(a_arr, empty)
    assert out.shape == (2, 0)


def test_cdist_dtw_empty_skips_ambiguity_raise():
    # Regression: ``cdist_dtw([], Y, sakoe_chiba_radius=2,
    # itakura_max_slope=2.0)`` returns the empty matrix on main rather
    # than raising, because main raises per-pair via ``_compute_mask``
    # — the empty iteration never triggers the raise. The fast path
    # used to raise eagerly before checking shape.
    Y = np.array([[[1.0], [2.0], [3.0]]], dtype=np.float64)
    empty = np.empty((0, 3, 1), dtype=np.float64)
    out = cdist_dtw(empty, Y, sakoe_chiba_radius=2, itakura_max_slope=2.0)
    assert out.shape == (0, 1)
    out = cdist_dtw(Y, empty, sakoe_chiba_radius=2, itakura_max_slope=2.0)
    assert out.shape == (1, 0)
    out = cdist_dtw(empty, sakoe_chiba_radius=2, itakura_max_slope=2.0)
    assert out.shape == (0, 0)


def test_cdist_frechet_empty_skips_ambiguity_raise():
    # Same regression for cdist_frechet: empty datasets must not surface
    # the ambiguous-constraint raise on main, since no pair iterates.
    Y = np.array([[[1.0], [2.0], [3.0]]], dtype=np.float64)
    empty = np.empty((0, 3, 1), dtype=np.float64)
    out = cdist_frechet(empty, Y, sakoe_chiba_radius=2, itakura_max_slope=2.0)
    assert out.shape == (0, 1)
    out = cdist_frechet(Y, empty, sakoe_chiba_radius=2, itakura_max_slope=2.0)
    assert out.shape == (1, 0)


def test_cdist_dtw_topk_falls_back_for_non_finite_inputs():
    # Regression: LB_Keogh + early-abandon turns NaN cell costs into
    # +inf-pruned candidates because ``NaN < ub_sq`` is False, so a
    # query with mid-series NaN would yield -1 indices on the fast
    # path while the full cdist_dtw path returns NaN distances with
    # real neighbor indices. Decline to cover the divergent case.
    X = np.array(
        [[[0.0], [np.nan], [1.0]]], dtype=np.float64,
    )
    Y = np.array(
        [[[0.0], [1.0], [2.0]], [[3.0], [4.0], [5.0]]], dtype=np.float64,
    )
    assert cdist_dtw_topk_fast(X, Y, k=1, radius=1) is None
    assert cdist_dtw_topk_fast(Y, X, k=1, radius=1) is None
    # All-finite inputs still take the fast path.
    Y_clean = np.array(
        [[[0.0], [1.0], [2.0]]], dtype=np.float64,
    )
    res = cdist_dtw_topk_fast(Y_clean, Y_clean, k=1, radius=1)
    assert res is not None


def test_matrix_profile_scaled_propagates_nan():
    # Regression: STOMP's ``s > EPS`` check silently maps NaN-derived
    # std to "constant per-dim" and contributes 0 to the distance,
    # so the scaled numpy path returned finite ``sqrt(m)`` values
    # while main propagates NaN through pdist.
    series = np.array(
        [[0.0], [1.0], [2.0], [np.nan], [4.0], [5.0], [6.0]],
        dtype=np.float64,
    )
    mp = MatrixProfile(subsequence_length=3, scale=True)
    out = mp.fit_transform([series])[0, :, 0]
    # Each window touching the NaN sample must propagate NaN; clean
    # windows on either side may legitimately find a non-NaN nearest
    # neighbor outside the exclusion band.
    assert np.isnan(out[1])
    assert np.isnan(out[2])
    assert np.isnan(out[3])


def test_matrix_profile_unscaled_propagates_nan():
    # Same regression for the unscaled formula: ``norm_sq[i] +
    # norm_sq[j] - 2*qt`` becomes NaN on a NaN window, but
    # ``NaN < local_mins`` is False so the window's min stays at
    # +inf, while main returns NaN.
    series = np.array(
        [[0.0], [1.0], [2.0], [np.nan], [4.0], [5.0], [6.0]],
        dtype=np.float64,
    )
    mp = MatrixProfile(subsequence_length=3, scale=False)
    out = mp.fit_transform([series])[0, :, 0]
    assert np.isnan(out[1])
    assert np.isnan(out[2])
    assert np.isnan(out[3])


def test_cdist_soft_dtw_rejects_zero_length_rows():
    # Regression: an all-NaN-padded series (e.g. ``[[NaN]]`` in a
    # length-1 dataset) trims to shape (0, d) on main, where
    # ``SquaredEuclidean.compute`` raises via sklearn's ``check_array``.
    # The fused kernel emits 0/DBL_MAX silently. Reject upfront.
    from tslearn.metrics import cdist_soft_dtw_normalized
    bad = np.array([[[np.nan]], [[1.0]]], dtype=np.float64)
    good = np.array([[[1.0], [2.0], [3.0]]], dtype=np.float64)
    with pytest.raises(ValueError, match="zero-length"):
        cdist_soft_dtw(bad, bad)
    with pytest.raises(ValueError, match="zero-length"):
        cdist_soft_dtw(bad, good)
    with pytest.raises(ValueError, match="zero-length"):
        cdist_soft_dtw(good, bad)
    # Self-similarity also rejects.
    with pytest.raises(ValueError, match="zero-length"):
        cdist_soft_dtw(bad)
    # Normalized variant must reject too — the self-diagonal kernel
    # propagates a 0 normalizer for zero-length rows otherwise.
    with pytest.raises(ValueError, match="zero-length"):
        cdist_soft_dtw_normalized(bad, good)
    # Truly empty datasets still return their (0, n) / (n, 0) shape.
    empty = np.empty((0, 3, 1), dtype=np.float64)
    out = cdist_soft_dtw(empty, good)
    assert out.shape == (0, 1)


def test_cdist_dtw_clamps_oversized_n_jobs():
    # Regression: numba's ``set_num_threads`` rejects values above
    # ``NUMBA_NUM_THREADS`` (the configured pool size), but main's
    # joblib-backed API accepts oversubscribed positive ``n_jobs``.
    rng = np.random.RandomState(0)
    X = rng.randn(3, 6, 1).astype(np.float64)
    # 99 is almost certainly above the pool size on any test machine;
    # the call must not raise.
    out = cdist_dtw(X, X, n_jobs=99)
    assert out.shape == (3, 3)
    out = cdist_frechet(X, X, n_jobs=99)
    assert out.shape == (3, 3)


def test_cdist_dtw_accepts_float_sakoe_radius():
    # Regression: a public-facing float ``sakoe_chiba_radius`` (e.g.
    # ``1.0``) was forwarded straight to the numba kernel which uses
    # ``radius`` in ``range(...)`` bounds — Numba TypingError. Main
    # accepts the float form because its mask helpers slice into the
    # mask buffer, not generate ranges.
    rng = np.random.RandomState(0)
    X = rng.randn(3, 6, 1).astype(np.float64)
    out_int = cdist_dtw(X, X, sakoe_chiba_radius=1)
    out_float = cdist_dtw(X, X, sakoe_chiba_radius=1.0)
    np.testing.assert_allclose(out_int, out_float, atol=ATOL)


def test_cdist_frechet_accepts_float_sakoe_radius():
    # Same regression for cdist_frechet: float ``sakoe_chiba_radius``
    # used to crash the numba kernel via ``range(jj_lo, jj_hi)`` typing.
    rng = np.random.RandomState(1)
    X = rng.randn(3, 6, 1).astype(np.float64)
    out_int = cdist_frechet(X, X, sakoe_chiba_radius=2)
    out_float = cdist_frechet(X, X, sakoe_chiba_radius=2.0)
    np.testing.assert_allclose(out_int, out_float, atol=ATOL)


def test_cdist_frechet_invalid_sakoe_radius_skips_empty_inputs():
    # Empty cross-distance calls do not evaluate any pair on main, so
    # they must not inspect even non-coercible radius values.
    empty = np.empty((0, 3, 1), dtype=np.float64)
    X = np.array([[[1.0], [2.0], [3.0]]], dtype=np.float64)

    assert cdist_frechet(empty, sakoe_chiba_radius="bad").shape == (0, 0)
    assert cdist_frechet(empty, X, sakoe_chiba_radius="bad").shape == (0, 1)
    assert cdist_frechet(X, empty, sakoe_chiba_radius="bad").shape == (1, 0)

    with pytest.raises(ValueError, match="could not convert"):
        cdist_frechet(X, X, sakoe_chiba_radius="bad")


def test_cdist_dtw_validates_n_jobs_before_empty_return():
    # Regression: moving the empty-shape short-circuit ahead of the
    # numba-threads context bypassed ``effective_n_jobs`` validation.
    # main raises on ``n_jobs == 0`` regardless of dataset shape.
    Y = np.array([[[1.0], [2.0], [3.0]]], dtype=np.float64)
    empty = np.empty((0, 3, 1), dtype=np.float64)
    with pytest.raises(ValueError):
        cdist_dtw(empty, Y, n_jobs=0)
    with pytest.raises(ValueError):
        cdist_dtw(Y, empty, n_jobs=0)
    with pytest.raises(ValueError):
        cdist_frechet(empty, Y, n_jobs=0)


def test_cdist_soft_dtw_normalized_validates_non_empty_side():
    # Regression: ``cdist_soft_dtw_normalized(empty, invalid)`` skipped
    # validation entirely — the cross-distance returned shape (0, n)
    # without inspecting the invalid side, and the self-diagonal
    # kernel didn't validate either. Main computes self-soft-DTW for
    # the non-empty side and raises via ``SquaredEuclidean.compute``.
    from tslearn.metrics import cdist_soft_dtw_normalized
    invalid = np.array([[[0.0], [np.nan], [1.0]]], dtype=np.float64)
    empty_left = np.empty((0, 3, 1), dtype=np.float64)
    with pytest.raises(ValueError, match="NaN"):
        cdist_soft_dtw_normalized(empty_left, invalid)
    with pytest.raises(ValueError, match="NaN"):
        cdist_soft_dtw_normalized(invalid, empty_left)


def test_resampler_preserves_interp_nan_behavior():
    # Regression: the equal-length vectorized resampler computed both
    # bracketing samples with weights w0/w1; when frac is 0 (exact-grid
    # position) the unused endpoint still paid 0*NaN = NaN and
    # contaminated the output. np.interp (the per-series fallback)
    # preserves NaN at exact-grid positions.
    from tslearn.preprocessing import TimeSeriesResampler

    out = TimeSeriesResampler(sz=3).fit_transform([[[1.0], [np.nan], [3.0]]])
    expected = np.array([1.0, np.nan, 3.0])
    np.testing.assert_array_equal(out[0, :, 0], expected)


def test_softdtw_rejects_non_finite_inputs():
    # Regression: the SquaredEuclidean numpy fast formula bypassed
    # sklearn's pairwise-distance validation, so soft_dtw with NaN or
    # Inf at a non-trailing position used to return nan instead of
    # raising. Match sklearn's failure mode so callers can rely on it.
    from tslearn.metrics import soft_dtw
    from tslearn.metrics.softdtw_variants import SquaredEuclidean

    # Mid-series NaN must raise (trailing NaN is the time-series-length
    # sentinel and is handled separately upstream).
    with pytest.raises(ValueError, match="NaN"):
        soft_dtw([[0.0], [np.nan], [1.0]], [[1.0], [2.0], [3.0]])

    # +/- Inf in either input must raise.
    with pytest.raises(ValueError, match="infinity"):
        soft_dtw([[np.inf]], [[1.0]])
    with pytest.raises(ValueError, match="infinity"):
        soft_dtw([[1.0]], [[-np.inf]])

    # SquaredEuclidean directly: same contract.
    with pytest.raises(ValueError):
        SquaredEuclidean(
            np.array([[np.nan]], dtype=np.float64),
            np.array([[1.0]], dtype=np.float64),
        ).compute()


def test_dtw_kneighbors_falls_through_for_negative_k():
    # Regression: the LB-prune fast path allocates output of shape
    # (n_query, k); passing k=-1 raised "negative dimensions are not
    # allowed" and left self.metric stuck as "precomputed". sklearn's
    # legacy fallback accepts negative n_neighbors (e.g. -1 means
    # "n_fit - 1"), so the fast path must fall through for k < 1.
    from tslearn.neighbors import KNeighborsTimeSeries

    rng = np.random.RandomState(0)
    X = rng.randn(5, 12, 1).astype(np.float64)
    knn = KNeighborsTimeSeries(
        n_neighbors=3, metric="dtw",
        metric_params={"sakoe_chiba_radius": 2},
    ).fit(X)

    # n_neighbors=-1 -> sklearn shape (n_query, n_fit - 1)
    dists, idxs = knn.kneighbors(X, n_neighbors=-1)
    assert dists.shape == (5, 4)
    assert idxs.shape == (5, 4)
    assert knn.metric == "dtw"

    # n_neighbors=0 -> sklearn shape (n_query, 0); metric still restored.
    dists, idxs = knn.kneighbors(X, n_neighbors=0)
    assert dists.shape == (5, 0)
    assert idxs.shape == (5, 0)
    assert knn.metric == "dtw"


def test_invalid_n_jobs_raises():
    # Regression: the new numba thread mapper used to silently treat
    # unsupported values like n_jobs=0 as "leave numba alone", hiding
    # caller misconfiguration. Match joblib (and main): raise on 0.
    rng = np.random.RandomState(0)
    X = rng.randn(4, 12, 1).astype(np.float64)
    with pytest.raises(ValueError):
        cdist_dtw(X, X, n_jobs=0)
    with pytest.raises(ValueError):
        cdist_frechet(X, X, n_jobs=0)
    with pytest.raises(ValueError):
        cdist_gak(X, X, sigma=sigma_gak(X), n_jobs=0)
    # Negative -N (N>=2) must work like joblib (cpu_count() - (N-1));
    # don't silently fall through to "all cores".
    out = cdist_dtw(X, X, n_jobs=-2)
    assert out.shape == (4, 4)


def test_softdtw_barycenter_rejects_empty_members():
    # Regression: zero-length / all-NaN members contributed 0 to value
    # and 0 gradient in the fused objective, silently dropping them
    # from the optimization. Reject them at the entry of
    # softdtw_barycenter — barycenter against empty members is
    # mathematically undefined.
    from tslearn.barycenters import softdtw_barycenter
    init = np.array([[0.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="zero-length"):
        softdtw_barycenter([[[np.nan]], [[1.0]]], init=init)


def test_dba_path_dispatch_normalizes_explicit_none_constraint():
    # Regression: an earlier cleanup gated the int-code normalization on
    # `gc is not None` and skipped the rewrite when the caller passed
    # `global_constraint=None` explicitly. None is a public value that
    # maps to the int code 0; leaving it in the dict made numba treat
    # `None == 0` as false, falling through to "unconstrained" and
    # silently swallowing the inferred-Itakura and ambiguous-constraint
    # cases.
    from tslearn.barycenters.dba import _dtw_path_dispatch
    from tslearn.metrics import dtw_path

    rng = np.random.RandomState(0)
    s1 = rng.randn(30, 1).astype(np.float64)
    s2 = rng.randn(30, 1).astype(np.float64)

    # Inferred Itakura (no global_constraint named, slope set) — must
    # match dtw_path even when the caller passes the key as explicit None.
    ref_path, ref_dist = dtw_path(s1, s2, itakura_max_slope=2.0)
    disp_dist, disp_path = _dtw_path_dispatch(
        s1, s2, global_constraint=None, itakura_max_slope=2.0,
    )
    np.testing.assert_allclose(disp_dist, ref_dist, atol=ATOL)
    assert disp_path == ref_path

    # Ambiguous (no global_constraint named, both radius and slope set)
    # — must raise the same RuntimeWarning as the public path.
    with pytest.raises(RuntimeWarning, match="global_constraint is not set"):
        _dtw_path_dispatch(
            s1, s2,
            global_constraint=None,
            sakoe_chiba_radius=2,
            itakura_max_slope=2.0,
        )


@pytest.mark.parametrize("scale", [False, True])
def test_stomp_matrix_profile_parity(scale):
    rng = np.random.RandomState(11)
    x = rng.randn(60, 1).astype(np.float64)
    m = 10
    band = int(np.ceil(m / 4))

    fast = MatrixProfile(subsequence_length=m, scale=scale).fit_transform(
        x[None, :, :]
    )[0, :, 0]
    slow = _naive_matrix_profile(x, m, scale, band)
    np.testing.assert_allclose(fast, slow, atol=ATOL)


def test_stomp_matrix_profile_rejects_multivariate_for_main_parity():
    rng = np.random.RandomState(11)
    x = rng.randn(60, 3).astype(np.float64)
    with pytest.raises(NotImplementedError):
        MatrixProfile(subsequence_length=10).fit_transform(x[None, :, :])


def test_stomp_constant_segment_edge_case():
    # A subsequence with zero std maps to an all-zero z-vector via the
    # std=1 substitution. The STOMP scaled formula must agree with the
    # explicit-segment naive path on this edge case.
    rng = np.random.RandomState(12)
    x = np.r_[np.zeros(15), rng.randn(45)].astype(np.float64)
    m = 8
    band = int(np.ceil(m / 4))
    fast = MatrixProfile(subsequence_length=m, scale=True).fit_transform(
        x[None, :, None]
    )[0, :, 0]
    slow = _naive_matrix_profile(x[:, None], m, True, band)
    np.testing.assert_allclose(fast, slow, atol=ATOL)


@pytest.mark.parametrize("ratio,expect_stomp", [(40.0, True), (60.0, False)])
def test_stomp_safe_threshold_parity_near_boundary(ratio, expect_stomp):
    # The ``_stomp_safe`` gate falls back to the pdist path when the
    # max ``|mu|/sig`` per subsequence exceeds 50. Pin the gate decision
    # directly (not just numeric parity) so a future threshold change
    # doesn't silently degrade to testing the fallback path for both cases.
    m = 8
    n = 40
    # Perfect alternating square wave +/-A around ``offset``. Every even-
    # length window has mu = offset exactly and sig = A exactly, giving
    # |mu|/sig = offset/A = ratio with no sample-variance noise.
    offset = 2.5
    A = offset / ratio
    x = (offset + A * ((-1.0) ** np.arange(n))).astype(np.float64)

    mp = MatrixProfile(subsequence_length=m, scale=True)
    assert mp._stomp_safe(x[:, None]) is expect_stomp, (
        f"_stomp_safe returned unexpected value for ratio={ratio}"
    )

    band = int(np.ceil(m / 4))
    fast = mp.fit_transform(x[None, :, None])[0, :, 0]
    slow = _naive_matrix_profile(x[:, None], m, True, band)
    np.testing.assert_allclose(fast, slow, atol=ATOL)


def test_cdist_dtw_oversize_n_jobs_matches_clamped():
    # ``numba_threads_for`` clamps ``n_jobs`` to ``NUMBA_NUM_THREADS``
    # (the numba pool size); above that ceiling results must still
    # match the un-oversubscribed reference, since the clamping is a
    # thread-count knob and should not change the numeric output.
    from numba import config as _numba_config
    rng = np.random.RandomState(7)
    X = rng.randn(8, 16, 1).astype(np.float64)

    pool_size = _numba_config.NUMBA_NUM_THREADS
    oversize = pool_size * 4 + 7

    out_default = cdist_dtw(X, X)
    out_oversize = cdist_dtw(X, X, n_jobs=oversize)
    np.testing.assert_allclose(out_oversize, out_default, atol=ATOL)

    # Same parity on cdist_frechet / cdist_gak / cdist_soft_dtw.
    out_default_f = cdist_frechet(X, X)
    out_oversize_f = cdist_frechet(X, X, n_jobs=oversize)
    np.testing.assert_allclose(out_oversize_f, out_default_f, atol=ATOL)

    sigma = float(sigma_gak(X))
    out_default_g = cdist_gak(X, X, sigma=sigma)
    out_oversize_g = cdist_gak(X, X, sigma=sigma, n_jobs=oversize)
    np.testing.assert_allclose(out_oversize_g, out_default_g, atol=ATOL)

    # ``cdist_soft_dtw`` takes no ``n_jobs`` (its fast path threads
    # internally via numba's pool); skip soft-DTW here.
