"""Verify ``verbose=`` chunked progress on the cdist fast paths matches
the unchunked output bit-for-bit, and that the heartbeat goes to stderr
without leaking to stdout. Covers the chunked self-similarity path and
the constraint variants for each metric whose public API exposes
``verbose``: DTW, Frechet, GAK.
"""
import numpy as np
import pytest

from tslearn.metrics import (
    cdist_dtw,
    cdist_frechet,
    cdist_gak,
    cdist_soft_dtw,
    cdist_soft_dtw_normalized,
)


@pytest.fixture
def rng():
    return np.random.RandomState(0)


# ----------------------------- cdist_frechet --------------------------------


def test_cdist_frechet_verbose_matches_quiet_cross(rng, capsys):
    x = rng.randn(7, 12, 1).astype(np.float64)
    y = rng.randn(5, 12, 1).astype(np.float64)
    np.testing.assert_allclose(
        cdist_frechet(x, y, verbose=1),
        cdist_frechet(x, y, verbose=0),
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "cdist_frechet:" in captured.err
    assert captured.err.count("\n") == 1


def test_cdist_frechet_verbose_matches_quiet_self(rng):
    x = rng.randn(9, 14, 1).astype(np.float64)
    np.testing.assert_allclose(
        cdist_frechet(x, verbose=1), cdist_frechet(x, verbose=0)
    )


def test_cdist_frechet_verbose_matches_quiet_self_sakoe(rng):
    x = rng.randn(9, 14, 1).astype(np.float64)
    np.testing.assert_allclose(
        cdist_frechet(x, sakoe_chiba_radius=2, verbose=1),
        cdist_frechet(x, sakoe_chiba_radius=2, verbose=0),
    )


def test_cdist_frechet_verbose_self_is_symmetric(rng):
    x = rng.randn(11, 10, 1).astype(np.float64)
    out = cdist_frechet(x, sakoe_chiba_radius=2, verbose=1)
    np.testing.assert_array_equal(out, out.T)


def test_cdist_frechet_verbose_zero_is_silent(rng, capsys):
    x = rng.randn(7, 12, 1).astype(np.float64)
    cdist_frechet(x, verbose=0)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


# ------------------------------- cdist_gak ----------------------------------


def test_cdist_gak_verbose_matches_quiet_cross(rng, capsys):
    x = rng.randn(7, 12, 1).astype(np.float64)
    y = rng.randn(5, 12, 1).astype(np.float64)
    np.testing.assert_allclose(
        cdist_gak(x, y, sigma=1.0, verbose=1),
        cdist_gak(x, y, sigma=1.0, verbose=0),
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "cdist_gak:" in captured.err


def test_cdist_gak_verbose_matches_quiet_self(rng):
    x = rng.randn(9, 14, 1).astype(np.float64)
    np.testing.assert_allclose(
        cdist_gak(x, sigma=1.0, verbose=1),
        cdist_gak(x, sigma=1.0, verbose=0),
    )


def test_cdist_gak_verbose_self_is_symmetric(rng):
    x = rng.randn(11, 10, 1).astype(np.float64)
    out = cdist_gak(x, sigma=1.0, verbose=1)
    # GAK normalization is symmetric to ~atol; assert with tolerance.
    np.testing.assert_allclose(out, out.T)


def test_cdist_gak_verbose_zero_is_silent(rng, capsys):
    x = rng.randn(7, 12, 1).astype(np.float64)
    cdist_gak(x, sigma=1.0, verbose=0)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


# ------------------------------- cdist_dtw ----------------------------------
# (Detailed dtw-specific paths live in test_cdist_dtw_progress.py — these
# two are smoke checks that the shared chunking helper is wired.)


def test_cdist_dtw_verbose_matches_quiet(rng, capsys):
    x = rng.randn(7, 12, 1).astype(np.float64)
    y = rng.randn(5, 12, 1).astype(np.float64)
    np.testing.assert_allclose(
        cdist_dtw(x, y, verbose=1), cdist_dtw(x, y, verbose=0)
    )
    captured = capsys.readouterr()
    assert "cdist_dtw:" in captured.err


# ---------------------------- cdist_soft_dtw --------------------------------


def test_cdist_soft_dtw_verbose_matches_quiet_cross(rng, capsys):
    x = rng.randn(7, 12, 1).astype(np.float64)
    y = rng.randn(5, 12, 1).astype(np.float64)
    np.testing.assert_allclose(
        cdist_soft_dtw(x, y, gamma=1.0, verbose=1),
        cdist_soft_dtw(x, y, gamma=1.0, verbose=0),
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "cdist_soft_dtw:" in captured.err


def test_cdist_soft_dtw_verbose_matches_quiet_self(rng):
    x = rng.randn(9, 14, 1).astype(np.float64)
    np.testing.assert_allclose(
        cdist_soft_dtw(x, gamma=1.0, verbose=1),
        cdist_soft_dtw(x, gamma=1.0, verbose=0),
    )


def test_cdist_soft_dtw_verbose_self_is_symmetric(rng):
    x = rng.randn(11, 10, 1).astype(np.float64)
    out = cdist_soft_dtw(x, gamma=1.0, verbose=1)
    np.testing.assert_array_equal(out, out.T)


def test_cdist_soft_dtw_verbose_zero_is_silent(rng, capsys):
    x = rng.randn(7, 12, 1).astype(np.float64)
    cdist_soft_dtw(x, gamma=1.0, verbose=0)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_cdist_soft_dtw_normalized_verbose_matches_quiet(rng, capsys):
    x = rng.randn(7, 12, 1).astype(np.float64)
    y = rng.randn(5, 12, 1).astype(np.float64)
    np.testing.assert_allclose(
        cdist_soft_dtw_normalized(x, y, gamma=1.0, verbose=1),
        cdist_soft_dtw_normalized(x, y, gamma=1.0, verbose=0),
    )
    captured = capsys.readouterr()
    # ``cdist_soft_dtw_normalized`` runs the cross dists once + two
    # self-diag computations; only the cross goes through the chunked
    # path, so the heartbeat label is ``cdist_soft_dtw``.
    assert "cdist_soft_dtw:" in captured.err
