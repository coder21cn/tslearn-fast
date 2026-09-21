import numpy as np

import pytest

from tslearn.neighbors import KNeighborsTimeSeries, KNeighborsTimeSeriesClassifier

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def test_k_neighbors_timeseries():
    n, sz, d = 15, 10, 3
    rng = np.random.RandomState(0)
    X = rng.randn(n, sz, d)

    model = KNeighborsTimeSeries()
    np.testing.assert_equal(
        model.fit(X).kneighbors(X, return_distance=False)[0],
        [0, 13, 7, 12, 3]
    )

    model = KNeighborsTimeSeries(metric='ctw')
    np.testing.assert_equal(
        model.fit(X).kneighbors(X, return_distance=False)[0],
        [0, 13, 7, 12, 3]
    )

    model = KNeighborsTimeSeries(metric='softdtw')
    np.testing.assert_equal(
        model.fit(X).kneighbors(X, return_distance=False)[0],
        [0, 13, 12, 7, 3]
    )

    model = KNeighborsTimeSeries(metric='frechet')
    np.testing.assert_equal(
        model.fit(X).kneighbors(X, return_distance=False)[0],
        [0, 3, 13, 5, 1]
    )

    with pytest.raises(ValueError):
        KNeighborsTimeSeries(metric='invalid').fit(X)


def test_k_neighbors_classifier():
    n, sz, d = 15, 10, 3
    rng = np.random.RandomState(0)
    X = rng.randn(n, sz, d)
    y = rng.randint(low=0, high=3, size=n)

    model_euc = KNeighborsTimeSeriesClassifier(n_neighbors=3,
                                               metric="euclidean")
    y_pred_euc = model_euc.fit(X, y).predict(X)
    model_dtw_sakoe = KNeighborsTimeSeriesClassifier(
        n_neighbors=3,
        metric="dtw",
        metric_params={
            "global_constraint": "sakoe_chiba",
            "sakoe_chiba_radius": 0
        }
    )
    y_pred_sakoe = model_dtw_sakoe.fit(X, y).predict(X)
    np.testing.assert_equal(y_pred_euc, y_pred_sakoe)

    model_softdtw = KNeighborsTimeSeriesClassifier(
        n_neighbors=3,
        metric="softdtw",
        metric_params={
            "gamma": 1e-6
        }
    )
    y_pred_softdtw = model_softdtw.fit(X, y).predict(X)

    model_dtw = KNeighborsTimeSeriesClassifier(
        n_neighbors=3,
        metric="dtw"
    )
    y_pred_dtw = model_dtw.fit(X, y).predict(X)

    np.testing.assert_equal(y_pred_dtw, y_pred_softdtw)

    model_ctw = KNeighborsTimeSeriesClassifier(
        n_neighbors=3,
        metric="ctw"
    )
    # Just testing that things run, nothing smart here :(
    model_ctw.fit(X, y).predict(X)

    model_sax = KNeighborsTimeSeriesClassifier(
        n_neighbors=3,
        metric="sax",
        metric_params={
            "alphabet_size_avg": 6,
            "n_segments": 10
        }
    )
    model_sax.fit(X, y)

    model_frechet = KNeighborsTimeSeriesClassifier(
        n_neighbors=1,
        metric="frechet"
    )
    np.testing.assert_equal(model_frechet.fit(X, y).predict(X), y)

    model_frechet = KNeighborsTimeSeriesClassifier(
        n_neighbors=3,
        metric="frechet"
    )
    np.testing.assert_equal(
        model_frechet.fit(X, y).kneighbors(X, return_distance=False)[0],
        [0,  3, 13]
    )

    with pytest.raises(ValueError):
        KNeighborsTimeSeriesClassifier(metric='invalid').fit(X, y)

    # The MINDIST of SAX is a lower bound of the euclidean distance
    euc_dist, _ = model_euc.kneighbors(X, n_neighbors=5)
    sax_dist, _ = model_sax.kneighbors(X, n_neighbors=5)

    # First column will contain zeroes
    np.testing.assert_array_less(sax_dist[:, 1:], euc_dist[:, 1:])


@pytest.mark.parametrize("n_neighbors", [1, 3, 5])
def test_dtw_kneighbors_returns_valid_indices_on_overflow(n_neighbors):
    from tslearn.metrics import cdist_dtw

    X_fit = np.array([[0., 0.], [1e154, 1e154], [-1e154, -1e154]])[..., None]
    X_query = np.array([[0., 0.], [3e154, 3e154]])[..., None]
    model = KNeighborsTimeSeries(
        n_neighbors=n_neighbors, metric="dtw",
        metric_params={"sakoe_chiba_radius": 1},
    ).fit(X_fit)
    dists, idxs = model.kneighbors(X_query)
    k = min(n_neighbors, len(X_fit))
    assert idxs.shape == (len(X_query), k)
    assert ((idxs >= 0) & (idxs < len(X_fit))).all()
    assert all(len(np.unique(row)) == k for row in idxs)
    full = cdist_dtw(X_query, X_fit, sakoe_chiba_radius=1)
    np.testing.assert_array_equal(dists, np.sort(full, axis=1)[:, :k])
    np.testing.assert_array_equal(dists, np.take_along_axis(full, idxs, axis=1))
    np.testing.assert_array_equal(
        model.kneighbors(X_query, return_distance=False), idxs,
    )


def test_dtw_kneighbors_clamps_k_to_fitted_population():
    # Regression: when n_neighbors > n_fit, the DTW LB-prune fast path
    # used to return padded -1 indices and +inf distances. It must
    # behave like the fallback and clip to the fitted population size.
    rng = np.random.RandomState(0)
    n_fit = 5
    X_fit = rng.randn(n_fit, 20, 1)
    X_query = rng.randn(2, 20, 1)

    model = KNeighborsTimeSeries(
        n_neighbors=3, metric="dtw",
        metric_params={"sakoe_chiba_radius": 2},
    ).fit(X_fit)

    dists, idxs = model.kneighbors(X_query, n_neighbors=10,
                                   return_distance=True)
    assert dists.shape == (2, n_fit)
    assert idxs.shape == (2, n_fit)
    assert not (idxs == -1).any()
    assert not np.isinf(dists).any()
    # All indices must point to the fitted corpus.
    assert idxs.min() >= 0
    assert idxs.max() < n_fit
