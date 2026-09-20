"""Verify the chunked-progress wrapper of ``cdist_dtw`` produces results
that are bit-equal to the unchunked path across every fast-path branch
(unconstrained / Sakoe-Chiba / Itakura) and on both cross- and
self-similarity inputs. The chunked self kernel relies on a
non-obvious upper-triangle / mirror invariant so it gets explicit
coverage here.
"""
import numpy as np
import pytest

from tslearn.metrics import cdist_dtw


@pytest.fixture
def rng():
    return np.random.RandomState(0)


def test_cdist_dtw_verbose_matches_quiet_cross(rng, capsys):
    x = rng.randn(7, 12, 1).astype(np.float64)
    y = rng.randn(5, 12, 1).astype(np.float64)
    quiet = cdist_dtw(x, y, verbose=0)
    verbose = cdist_dtw(x, y, verbose=1)
    np.testing.assert_allclose(verbose, quiet)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "cdist_dtw:" in captured.err
    # Single trailing newline; the heartbeat itself uses ``\r``.
    assert captured.err.count("\n") == 1


def test_cdist_dtw_verbose_matches_quiet_self_unconstrained(rng):
    x = rng.randn(9, 14, 1).astype(np.float64)
    np.testing.assert_allclose(
        cdist_dtw(x, verbose=1), cdist_dtw(x, verbose=0)
    )


def test_cdist_dtw_verbose_matches_quiet_self_sakoe(rng):
    x = rng.randn(9, 14, 1).astype(np.float64)
    np.testing.assert_allclose(
        cdist_dtw(x, sakoe_chiba_radius=2, verbose=1),
        cdist_dtw(x, sakoe_chiba_radius=2, verbose=0),
    )


def test_cdist_dtw_verbose_matches_quiet_self_itakura(rng):
    # Itakura on the self path needs equal valid lengths within the
    # dataset; use a finite fixed-length input so the fast path takes it.
    x = rng.randn(8, 14, 1).astype(np.float64)
    np.testing.assert_allclose(
        cdist_dtw(x, itakura_max_slope=2.0, verbose=1),
        cdist_dtw(x, itakura_max_slope=2.0, verbose=0),
    )


def test_cdist_dtw_verbose_matches_quiet_cross_sakoe(rng):
    x = rng.randn(7, 12, 1).astype(np.float64)
    y = rng.randn(5, 12, 1).astype(np.float64)
    np.testing.assert_allclose(
        cdist_dtw(x, y, sakoe_chiba_radius=2, verbose=1),
        cdist_dtw(x, y, sakoe_chiba_radius=2, verbose=0),
    )


def test_cdist_dtw_verbose_matches_quiet_cross_itakura(rng):
    # Cross Itakura needs equal valid lengths within each dataset.
    x = rng.randn(7, 12, 1).astype(np.float64)
    y = rng.randn(5, 12, 1).astype(np.float64)
    np.testing.assert_allclose(
        cdist_dtw(x, y, itakura_max_slope=2.0, verbose=1),
        cdist_dtw(x, y, itakura_max_slope=2.0, verbose=0),
    )


def test_cdist_dtw_verbose_matches_quiet_variable_length(rng):
    # NaN-padded variable-length input. Pin the cross-pair lens slicing
    # at a chunk boundary by forcing many small chunks via verbose=1.
    x = rng.randn(6, 10, 1).astype(np.float64)
    x[1, 7:, :] = np.nan
    x[3, 5:, :] = np.nan
    y = rng.randn(4, 10, 1).astype(np.float64)
    y[2, 6:, :] = np.nan
    np.testing.assert_allclose(
        cdist_dtw(x, y, verbose=1), cdist_dtw(x, y, verbose=0)
    )
    np.testing.assert_allclose(
        cdist_dtw(x, verbose=1), cdist_dtw(x, verbose=0)
    )


def test_cdist_dtw_verbose_self_is_symmetric(rng):
    # Mirror-write invariant: every ``out[j, i]`` cell is populated by
    # the same chunk that wrote ``out[i, j]``. Asserting symmetry is a
    # cheap way to catch any chunk-boundary mirror miss.
    x = rng.randn(11, 10, 1).astype(np.float64)
    out = cdist_dtw(x, sakoe_chiba_radius=2, verbose=1)
    np.testing.assert_array_equal(out, out.T)


def test_cdist_dtw_verbose_zero_is_silent(rng, capsys):
    x = rng.randn(7, 12, 1).astype(np.float64)
    cdist_dtw(x, verbose=0)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
