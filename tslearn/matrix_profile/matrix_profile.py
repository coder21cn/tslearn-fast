import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.spatial.distance import pdist, squareform
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted

from tslearn.bases import BaseModelPackage, TimeSeriesMixin
from tslearn.metrics.utils import _numba_allows_concurrent_calls
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import check_array, check_dims
from ._mp_fast import _run_mp_stomp


def _series_to_segments(time_series, segment_size):
    """Sliding window view of ``time_series`` with shape
    ``(n - segment_size + 1, segment_size, d)``. Read-only view; no copy.
    """
    if time_series.ndim != 2:
        raise ValueError(
            "_series_to_segments expects a single (sz, d) time series; got "
            f"shape {time_series.shape}"
        )
    elem_size = time_series.strides[0]
    return as_strided(
        time_series,
        strides=(elem_size, elem_size, time_series.strides[1]),
        shape=(time_series.shape[0] - segment_size + 1,
               segment_size, time_series.shape[1]),
        writeable=False,
    )

stumpy_msg = ('stumpy is not installed, stumpy features will not be'
              'supported.\n Install stumpy to use stumpy features:'
              'https://stumpy.readthedocs.io/en/latest/')

try:
    import stumpy
except ImportError:
    STUMPY_INSTALLED = False
else:
    STUMPY_INSTALLED = True

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class MatrixProfile(TimeSeriesMixin,
                    TransformerMixin,
                    BaseEstimator,
                    BaseModelPackage):
    """Matrix Profile transformation.

    Matrix Profile was originally presented in [1]_.

    Parameters
    ----------
    subsequence_length : int (default: 1)
        Length of the subseries (also called window size) to be used for
        subseries distance computations.

    implementation : str (default: "numpy")
        Matrix profile implementation to use.
        Defaults to "numpy" to use the pure numpy version.
        All the available implementations are ["numpy", "stump", "gpu_stump"].

        "stump" and "gpu_stump" are both implementations from the stumpy
        python library, the latter requiring a GPU.
        Stumpy is a library for efficiently computing the matrix profile which
        is optimized for speed, performance and memory.
        See [2]_ for the documentation.
        "numpy" is the default pure numpy implementation and does not require
        stumpy to be installed.

    scale: bool (default: True)
         Whether input data should be scaled for each feature of each time
         series to have zero mean and unit variance.
         Default for this parameter is set to `True` to match the standard
         matrix profile setup.

    Notes
    -----
    Matrix profiles are restricted to univariate input (``d == 1``),
    matching the public behavior of the upstream implementation. The
    optimized ``"numpy"`` implementation keeps the same restriction for
    drop-in compatibility.

    Inputs containing non-finite values (``NaN`` / ``Inf``) bypass the
    fast STOMP kernel and fall back to a pdist-based path that
    propagates ``NaN`` cells through the matrix profile, matching the
    legacy behavior.

    Examples
    --------
    >>> time_series = [0., 1., 3., 2., 9., 1., 14., 15., 1., 2., 2., 10., 7.]
    >>> ds = [time_series]
    >>> mp = MatrixProfile(subsequence_length=4, scale=False)
    >>> mp.fit_transform(ds)[0, :, 0]  # doctest: +ELLIPSIS
    array([ 6.85...,  1.41...,  6.16...,  7.93..., 11.40...,
           13.56..., 18.  ..., 13.96...,  1.41...,  6.16...])

    References
    ----------
    .. [1] C. M. Yeh, Y. Zhu, L. Ulanova, N.Begum et al.
       Matrix Profile I: All Pairs Similarity Joins for Time Series: A
       Unifying View that Includes Motifs, Discords and Shapelets.
       ICDM 2016.

    .. [2] STUMPY documentation https://stumpy.readthedocs.io/en/latest/

    """

    def __init__(
        self, subsequence_length=1, implementation="numpy", scale=True
    ):
        self.subsequence_length = subsequence_length
        self.scale = scale
        self.implementation = implementation

    def _is_fitted(self):
        check_is_fitted(self, '_X_fit_dims')
        return True

    def _fit(self, X, y=None):
        self._X_fit_dims = X.shape
        self.n_features_in_ = X.shape[-1]
        return self

    def fit(self, X, y=None):
        """Fit a Matrix Profile representation.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset

        Returns
        -------
        MatrixProfile
            self
        """
        X = check_array(X, allow_nd=True, force_all_finite=False)
        X = check_dims(X)
        return self._fit(X)

    def _transform(self, X, y=None):
        n_ts, sz, d = X.shape

        if d > 1:
            # Main rejects any multivariate input here. The numpy STOMP
            # kernel can handle ``d > 1``, but accepting it would produce
            # results with no main-parity baseline. Restore main's
            # blanket rejection to keep this branch behavior-preserving;
            # multivariate support can land as a separate feature commit.
            raise NotImplementedError("We currently don't support using "
                                      "multi-dimensional matrix profiles "
                                      "from the stumpy library.")

        output_size = sz - self.subsequence_length + 1
        X_transformed = np.empty((n_ts, output_size, 1))

        if self.implementation == "stump":
            if not STUMPY_INSTALLED:
                raise ImportError(stumpy_msg)

            for i_ts in range(n_ts):
                result = stumpy.stump(
                    T_A=X[i_ts, :, 0].ravel(),
                    m=self.subsequence_length)
                X_transformed[i_ts, :, 0] = result[:, 0].astype(float)

        elif self.implementation == "gpu_stump":
            if not STUMPY_INSTALLED:
                raise ImportError(stumpy_msg)

            for i_ts in range(n_ts):
                result = stumpy.gpu_stump(
                    T_A=X[i_ts, :, 0].ravel(),
                    m=self.subsequence_length)
                X_transformed[i_ts, :, 0] = result[:, 0].astype(float)

        elif self.implementation == "numpy":
            band_width = int(np.ceil(self.subsequence_length / 4))
            for i_ts in range(n_ts):
                ts = np.ascontiguousarray(X[i_ts], dtype=np.float64)
                # STOMP's std>EPS guard and the unscaled formula both
                # silently swallow NaN cells, so non-finite inputs
                # diverge from main's pdist path. Fall back per series.
                if (np.isfinite(ts).all() and self._stomp_safe(ts)
                        and _numba_allows_concurrent_calls()):
                    mins_sq = np.empty(output_size, dtype=np.float64)
                    _run_mp_stomp(
                        ts, self.subsequence_length, self.scale,
                        band_width, mins_sq,
                    )
                    X_transformed[i_ts] = np.sqrt(mins_sq)[:, None]
                else:
                    X_transformed[i_ts] = self._numpy_pdist_matrix_profile(
                        ts, band_width
                    )

        else:
            available_implementations = ["numpy", "stump", "gpu_stump"]
            raise ValueError(
                'This "{}" matrix profile implementation is not'
                ' recognized. Available implementations are {}.'
                .format(self.implementation,
                        available_implementations)
            )

        return X_transformed

    def _stomp_safe(self, ts):
        """Gate the STOMP recurrence on per-subsequence mean-to-std ratio.

        For ``scale=True``, STOMP computes the centered cross-product as
        ``qt - m*mu_i*mu_j``, which suffers catastrophic cancellation
        when ``|mu|`` is large relative to ``sig`` (e.g. drift signals,
        non-zero-mean signals). Main's pdist path z-normalizes each
        segment first and avoids the cancellation. When the ratio is
        too high, fall back to pdist to preserve numerical parity.

        The unscaled squared-norm formula also suffers cancellation on
        large-offset signals, so apply the same gate in both modes.
        Scaled inputs with tiny standard deviations use the fallback
        because the kernel treats standard deviations <= 1e-12 as zero.
        Large changes in window energy also require the fallback: running
        sums retain roundoff from earlier high-energy windows.
        """
        m = self.subsequence_length
        sz = ts.shape[0]
        n = sz - m + 1
        if n < 1:
            # Preserve the fallback's validation when no windows exist.
            return False
        cs = np.concatenate(
            [np.zeros((1, ts.shape[1])),
             np.cumsum(ts, axis=0, dtype=np.float64)]
        )
        cs2 = np.concatenate(
            [np.zeros((1, ts.shape[1])),
             np.cumsum(ts * ts, axis=0, dtype=np.float64)]
        )
        sums = cs[m:m + n] - cs[:n]
        sums2 = cs2[m:m + n] - cs2[:n]
        # Keep each window's energy well above the accumulated roundoff
        # scale. Otherwise neither running norms nor dot products remain
        # accurate after a sharp drop in signal amplitude.
        if (sums2 <= 1e-6 * cs2[-1]).any():
            return False
        mu = sums / m
        var = np.maximum(sums2 / m - mu * mu, 0.0)
        sig = np.sqrt(var)
        if self.scale and (sig <= 1e-12).any():
            return False
        # Reject strongly ill-conditioned window statistics. The kernel
        # separately recomputes near-match distances to account for the
        # cancellation that can still occur below this mean/std limit.
        return bool((np.abs(mu) <= 50.0 * (sig + 1e-12)).all())

    def _numpy_pdist_matrix_profile(self, ts, band_width):
        """Pdist-based matrix profile that propagates NaN. Mirrors
        main's segment / scaler / pdist pipeline, used as a fallback
        when the STOMP recurrence would silently swallow NaN cells."""
        m = self.subsequence_length
        segments = _series_to_segments(ts, m)
        if self.scale:
            segments = TimeSeriesScalerMeanVariance().fit_transform(segments)
        n = segments.shape[0]
        # Match upstream's reshape, including its error for zero-size windows.
        flat = segments.reshape(-1, m * ts.shape[1])
        dists = squareform(pdist(flat, metric="euclidean"))
        band = (np.tri(n, n, band_width, dtype=bool)
                & ~np.tri(n, n, -(band_width + 1), dtype=bool))
        dists[band] = np.inf
        return dists.min(axis=1, keepdims=True)

    def transform(self, X, y=None):
        """Transform a dataset of time series into its Matrix Profile
         representation.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset

        Returns
        -------
        numpy.ndarray of shape (n_ts, output_size, 1)
            Matrix-Profile-Transformed dataset. `ouput_size` is equal to
            `sz - subsequence_length + 1`
        """
        self._is_fitted()
        X = check_array(X, allow_nd=True, force_all_finite=False)
        X = check_dims(X, X_fit_dims=self._X_fit_dims,
                       check_n_features_only=True)
        return self._transform(X, y)

    def fit_transform(self, X, y=None, **fit_params):
        """Transform a dataset of time series into its Matrix Profile
         representation.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset

        Returns
        -------
        numpy.ndarray of shape (n_ts, output_size, 1)
            Matrix-Profile-Transformed dataset. `ouput_size` is equal to
            `sz - subsequence_length + 1`
        """
        X = check_array(X, allow_nd=True, force_all_finite=False)
        X = check_dims(X)
        return self._fit(X)._transform(X)

    def _more_tags(self):
        tags = super()._more_tags()
        tags.update({'allow_nan': True, 'allow_variable_length': True})
        return tags

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.allow_variable_length = True
        return tags
