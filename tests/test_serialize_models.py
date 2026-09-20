import tempfile
from pathlib import Path

import numpy

import pytest

from sklearn.exceptions import NotFittedError

from tslearn import hdftools
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.neighbors import KNeighborsTimeSeries, \
    KNeighborsTimeSeriesClassifier
from tslearn.clustering import KShape, TimeSeriesKMeans, \
    KernelKMeans
from tslearn.generators import random_walks
from tslearn.piecewise import PiecewiseAggregateApproximation, \
    SymbolicAggregateApproximation, OneD_SymbolicAggregateApproximation


all_formats = ['json', 'hdf5', 'pickle']


def test_hdftools(tmp_path):
    dtypes = [int, numpy.int8, numpy.int16, numpy.int32, numpy.int64,
              float, numpy.float32, numpy.float64]

    d = {}

    for dtype in dtypes:
        name = numpy.dtype(dtype).name
        d[name] = (numpy.random.rand(100, 100) * 10).astype(dtype)

    fname = str(tmp_path / 'hdf_test.hdf5')

    hdftools.save_dict(d, filename=fname, group='data')

    d2 = hdftools.load_dict(fname, 'data')

    for k in d2.keys():
        numpy.testing.assert_equal(d[k], d2[k])


def _check_not_fitted(model, tmp_path):
    # not serializable if not fitted
    for fmt in all_formats:
        with pytest.raises(NotFittedError):
            getattr(model, "to_{}".format(fmt))(
                str(tmp_path / "{}.{}".format(model.__class__.__name__, fmt))
            )


def _check_params_predict(model, X, test_methods, tmp_path,
                          check_params_fun=None,
                          formats=None, exclude_hyper_params=None):
    if formats is None:
        formats = all_formats
    # Each call gets its own sub-dir: tests sometimes serialize the same
    # class twice (different configs) within one test, and hdftools
    # refuses to overwrite an existing target.
    sub = Path(tempfile.mkdtemp(dir=str(tmp_path)))
    for fmt in formats:
        getattr(model, "to_{}".format(fmt))(
            str(sub / "{}.{}".format(model.__class__.__name__, fmt))
        )

    # loaded models should have same model params
    # and provide the same predictions
    for fmt in formats:
        sm = getattr(model, "from_{}".format(fmt))(
            str(sub / "{}.{}".format(model.__class__.__name__, fmt))
        )

        # make sure it's restored to the same class
        assert isinstance(sm, model.__class__)

        # test that predictions/transforms etc. are the same
        for method in test_methods:
            m1 = getattr(model, method)
            m2 = getattr(sm, method)
            numpy.testing.assert_equal(m1(X), m2(X))

        model_params = model._get_model_params()
        if check_params_fun is None:
            # check that the model-params are the same
            for p in model_params.keys():
                numpy.testing.assert_equal(getattr(model, p), getattr(sm, p))
        else:
            numpy.testing.assert_equal(check_params_fun(model),
                                       check_params_fun(sm))

        # check that hyper-params are the same
        exclude_hyper_params = exclude_hyper_params or []
        hyper_params = model.get_params()
        for p in hyper_params.keys():
            if p in exclude_hyper_params:
                continue
            numpy.testing.assert_equal(getattr(model, p), getattr(sm, p))


def test_serialize_global_alignment_kernel_kmeans(tmp_path):
    n, sz, d = 15, 10, 3
    rng = numpy.random.RandomState(0)
    X = rng.randn(n, sz, d)

    gak_km = KernelKMeans(n_clusters=3, verbose=False,
                          max_iter=5)

    _check_not_fitted(gak_km, tmp_path)

    gak_km.fit(X)

    _check_params_predict(gak_km, X, ['predict'], tmp_path)


def test_serialize_timeserieskmeans(tmp_path):
    n, sz, d = 15, 10, 3
    rng = numpy.random.RandomState(0)
    X = rng.randn(n, sz, d)

    dba_km = TimeSeriesKMeans(n_clusters=3,
                              n_init=2,
                              metric="dtw",
                              verbose=True,
                              max_iter_barycenter=10)

    _check_not_fitted(dba_km, tmp_path)

    dba_km.fit(X)

    _check_params_predict(dba_km, X, ['predict'], tmp_path)

    sdtw_km = TimeSeriesKMeans(n_clusters=3,
                               metric="softdtw",
                               metric_params={"gamma": .01},
                               verbose=True)

    _check_not_fitted(sdtw_km, tmp_path)

    sdtw_km.fit(X)

    _check_params_predict(sdtw_km, X, ['predict'], tmp_path)


def test_serialize_kshape(tmp_path):
    n, sz, d = 15, 10, 3
    rng = numpy.random.RandomState(0)
    time_series = rng.randn(n, sz, d)
    X = TimeSeriesScalerMeanVariance().fit_transform(time_series)

    ks = KShape(n_clusters=3, verbose=True)

    _check_not_fitted(ks, tmp_path)

    ks.fit(X)

    _check_params_predict(ks, X, ['predict'], tmp_path)

    seed_ixs = [numpy.random.randint(0, X.shape[0] - 1) for i in range(3)]
    seeds = numpy.array([X[i] for i in seed_ixs])

    ks_seeded = KShape(n_clusters=3, verbose=True, init=seeds)

    _check_not_fitted(ks_seeded, tmp_path)

    ks_seeded.fit(X)

    _check_params_predict(ks_seeded, X, ['predict'], tmp_path)


def test_serialize_knn(tmp_path):
    n, sz, d = 15, 10, 3
    rng = numpy.random.RandomState(0)
    X = rng.randn(n, sz, d)
    y = rng.randint(low=0, high=3, size=n)

    n_neighbors = 3

    knn = KNeighborsTimeSeries(n_neighbors=n_neighbors)

    _check_not_fitted(knn, tmp_path)

    knn.fit(X, y)

    _check_params_predict(knn, X, ['kneighbors'], tmp_path)


def test_serialize_knn_classifier(tmp_path):
    n, sz, d = 15, 10, 3
    rng = numpy.random.RandomState(0)
    X = rng.randn(n, sz, d)
    y = rng.randint(low=0, high=3, size=n)

    knc = KNeighborsTimeSeriesClassifier()

    _check_not_fitted(knc, tmp_path)

    knc.fit(X, y)

    _check_params_predict(knc, X, ['predict'], tmp_path)


def _get_random_walk():
    numpy.random.seed(0)
    # Generate a random walk time series
    n_ts, sz, d = 1, 100, 1
    dataset = random_walks(n_ts=n_ts, sz=sz, d=d)
    scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)
    return scaler.fit_transform(dataset)


def test_serialize_paa(tmp_path):
    X = _get_random_walk()
    # PAA transform (and inverse transform) of the data
    n_paa_segments = 10
    paa = PiecewiseAggregateApproximation(n_segments=n_paa_segments)

    _check_not_fitted(paa, tmp_path)

    paa.fit(X)

    _check_params_predict(paa, X, ['transform'], tmp_path)


def test_serialize_sax(tmp_path):
    n_paa_segments = 10
    n_sax_symbols = 8
    sax = SymbolicAggregateApproximation(n_segments=n_paa_segments,
                                         alphabet_size_avg=n_sax_symbols)

    _check_not_fitted(sax, tmp_path)

    X = _get_random_walk()

    sax.fit(X)

    _check_params_predict(sax, X, ['transform'], tmp_path)


def test_serialize_1dsax(tmp_path):

    n_paa_segments = 10
    n_sax_symbols_avg = 8
    n_sax_symbols_slope = 8

    one_d_sax = OneD_SymbolicAggregateApproximation(
        n_segments=n_paa_segments,
        alphabet_size_avg=n_sax_symbols_avg,
        alphabet_size_slope=n_sax_symbols_slope)

    _check_not_fitted(one_d_sax, tmp_path)

    X = _get_random_walk()
    one_d_sax.fit(X)

    _check_params_predict(one_d_sax, X, ['transform'], tmp_path)


def test_serialize_shapelets(tmp_path):
    shapelets = pytest.importorskip('tslearn.shapelets', exc_type=ImportError)
    from keras.optimizers import Adam

    def get_model_weights(model):
        return model.model_.get_weights()

    n, sz, d = 15, 10, 3
    rng = numpy.random.RandomState(0)
    X = rng.randn(n, sz, d)

    for y in [rng.randint(low=0, high=3, size=n),
              rng.choice(["one", "two", "three"], size=n)]:

        # Test with default args
        shp = shapelets.LearningShapelets(max_iter=1)
        _check_not_fitted(shp, tmp_path)
        shp.fit(X, y)
        _check_params_predict(shp, X, ['predict'], tmp_path,
                              check_params_fun=get_model_weights,
                              formats=["json", "pickle"])

        # Tests args with types differing from default
        shp = shapelets.LearningShapelets(
            n_shapelets_per_size={2: 2},
            optimizer=Adam(0.01),
            max_iter=1,
            max_size=10,
            random_state=42
        )
        _check_not_fitted(shp, tmp_path)
        shp.fit(X, y)
        _check_params_predict(
            shp,
            X,
            ['predict'],
            tmp_path,
            check_params_fun=get_model_weights,
            formats=["json", "pickle"],
            exclude_hyper_params=["optimizer"]
        )

        # HDF5 serialization not supported
        with pytest.raises(NotImplementedError):
            _check_params_predict(shp, X, ['predict'], tmp_path,
                                  check_params_fun=get_model_weights,
                                  formats=["hdf5"])
