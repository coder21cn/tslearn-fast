"""Head-to-head benchmark vs the ``main`` branch.

Times a small set of representative perf-sensitive calls under both
``main`` (via the same ``.main_parity_wt`` worktree the parity harness
maintains) and HEAD, and prints a side-by-side speedup table.

Why not asv: asv builds a virtualenv per rev, which is overkill when
we already have a clean main worktree managed by ``check_main_parity``
and want a quick sanity number. asv stays available for cross-commit
tracking; this script is the "did the perf branch deliver" smoke test.

Usage::

    python tests/bench_vs_main.py [--repeats N] [--main-rev REV]
"""
from __future__ import annotations

import argparse
import os
import pickle
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


# Force the repo root onto ``sys.path`` so ``tests._main_worktree``
# resolves to ours even when an editable install puts another
# ``tests`` package on ``sys.path``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from tests._main_worktree import REPO_ROOT, WORKTREE_DIR, ensure_worktree

RUNNER = REPO_ROOT / "tests" / "_bench_runner.py"


# --- benchmark scenarios ---------------------------------------------------

_RNG = np.random.RandomState(0)


def _scenarios():
    """Yield ``(case_id, dotted_func, args, kwargs)`` tuples.

    Each call returns quickly enough (<1s typical) that we can repeat
    it a handful of times for a reasonable median.
    """
    # cdist_dtw — short, medium, sakoe-banded.
    yield ("cdist_dtw[N=100, L=32]",
           "tslearn.metrics.cdist_dtw",
           (_RNG.randn(100, 32, 1).astype(np.float64),), {})
    yield ("cdist_dtw[N=200, L=64]",
           "tslearn.metrics.cdist_dtw",
           (_RNG.randn(200, 64, 1).astype(np.float64),), {})
    yield ("cdist_dtw[N=80, L=200, sakoe=10]",
           "tslearn.metrics.cdist_dtw",
           (_RNG.randn(80, 200, 1).astype(np.float64),),
           {"global_constraint": "sakoe_chiba", "sakoe_chiba_radius": 10})
    yield ("cdist_dtw[N=40, L=400, multivariate d=5]",
           "tslearn.metrics.cdist_dtw",
           (_RNG.randn(40, 400, 5).astype(np.float64),), {})

    # cdist_soft_dtw
    yield ("cdist_soft_dtw[N=100, L=32]",
           "tslearn.metrics.cdist_soft_dtw",
           (_RNG.randn(100, 32, 1).astype(np.float64),),
           {"gamma": 1.0})

    # cdist_gak
    yield ("cdist_gak[N=100, L=32]",
           "tslearn.metrics.cdist_gak",
           (_RNG.randn(100, 32, 1).astype(np.float64),),
           {"sigma": 1.0})

    # cdist_frechet
    yield ("cdist_frechet[N=100, L=32]",
           "tslearn.metrics.cdist_frechet",
           (_RNG.randn(100, 32, 1).astype(np.float64),), {})

    # kNN-DTW end-to-end (LB-prune fast path on Sakoe-banded inputs).
    knn_train = _RNG.randn(200, 64, 1).astype(np.float64)
    knn_test = _RNG.randn(50, 64, 1).astype(np.float64)
    yield ("kneighbors_dtw[N_train=200, N_test=50, L=64, sakoe=4, k=3]",
           "local:run_kneighbors_dtw", (knn_train, knn_test, 3, 4), {})

    # KMeans-DTW end-to-end.
    km_X = _RNG.randn(80, 32, 1).astype(np.float64)
    yield ("kmeans_dtw[N=80, L=32, k=4, sakoe=3]",
           "local:run_kmeans_dtw", (km_X, 4, 3), {})

    # Soft-DTW barycenter (L-BFGS).
    bary_X = _RNG.randn(20, 24, 1).astype(np.float64)
    yield ("softdtw_barycenter[N=20, L=24]",
           "local:run_softdtw_barycenter", (bary_X, 1.0, 20), {})

    # MatrixProfile.
    mp_X = _RNG.randn(1, 1000, 1).astype(np.float64)
    yield ("matrix_profile[N=1, L=1000, m=32]",
           "local:run_matrix_profile", (mp_X, 32, True), {})


SCENARIOS = list(_scenarios())


# --- subprocess harness ----------------------------------------------------

def _run_in_subprocess(env_extra, repeats):
    """Time each scenario in a single subprocess so cold-start cost is
    paid once, not per scenario. Returns ``{case_id: median_seconds}``."""
    payload = (SCENARIOS, repeats)
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as fin:
        pickle.dump(payload, fin)
        in_path = fin.name
    out_path = in_path + ".out"

    env = os.environ.copy()
    env.update(env_extra)

    completed = subprocess.run(
        [sys.executable, str(RUNNER), in_path, out_path],
        env=env,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"bench runner failed (rc={completed.returncode}):\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )

    with open(out_path, "rb") as f:
        results = pickle.load(f)

    os.unlink(in_path)
    os.unlink(out_path)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5,
                        help="Median over N runs per scenario (default 5).")
    parser.add_argument("--main-rev", default="main",
                        help="Git rev to compare against (default: main).")
    args = parser.parse_args(argv)

    main_dir = ensure_worktree(args.main_rev)

    # Run main first so its kernels are JIT-compiled when measured;
    # then HEAD. Both runs go through the same runner, so the
    # measurement methodology is identical.
    main_env = {"PYTHONPATH": str(main_dir) + os.pathsep
                + os.environ.get("PYTHONPATH", "")}
    main_results = _run_in_subprocess(main_env, args.repeats)

    head_env = {"PYTHONPATH": str(REPO_ROOT) + os.pathsep
                + os.environ.get("PYTHONPATH", "")}
    head_results = _run_in_subprocess(head_env, args.repeats)

    # Print table.
    name_w = max(len(case_id) for case_id, *_ in SCENARIOS)
    print(f"{'case':<{name_w}}  {'main (s)':>10}  {'HEAD (s)':>10}  "
          f"{'speedup':>8}")
    print("-" * (name_w + 36))
    for case_id, *_ in SCENARIOS:
        m = main_results.get(case_id)
        h = head_results.get(case_id)
        if m is None or h is None:
            print(f"{case_id:<{name_w}}  {'?':>10}  {'?':>10}  {'?':>8}")
            continue
        speedup = m / h if h > 0 else float("inf")
        print(f"{case_id:<{name_w}}  {m:>10.4f}  {h:>10.4f}  "
              f"{speedup:>7.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
