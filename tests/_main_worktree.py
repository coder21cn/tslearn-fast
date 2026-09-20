"""Shared worktree management for the main-parity harness and the
head-to-head benchmark.

Single source of truth so the two drivers can't drift on the
freshness check — the previous duplicate in ``bench_vs_main.py``
silently compared against a stale worktree whenever the parity
harness had left it on a different rev.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
WORKTREE_DIR = REPO_ROOT / ".main_parity_wt"


def ensure_worktree(rev: str) -> Path:
    """Idempotently create a detached worktree of ``rev`` at ``WORKTREE_DIR``.

    Refreshes the cached worktree when its HEAD doesn't match the
    requested rev, so ``--main-rev`` is honored across re-runs even
    when a different rev was checked out previously. Skips the
    ``checkout --detach`` when the worktree HEAD already resolves to
    the same commit — even a no-op checkout rewrites the index and
    stat-refreshes every tracked file (~hundreds of ms on Windows
    for a tslearn-sized tree).
    """
    if (WORKTREE_DIR / "tslearn").exists():
        cur = subprocess.run(
            ["git", "-C", str(WORKTREE_DIR), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        target = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", rev],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        if cur != target:
            subprocess.run(
                ["git", "-C", str(WORKTREE_DIR), "checkout", "--detach", rev],
                check=True, capture_output=True,
            )
        return WORKTREE_DIR
    subprocess.run(
        ["git", "-C", str(REPO_ROOT), "worktree", "add",
         "--detach", str(WORKTREE_DIR), rev],
        check=True, capture_output=True,
    )
    return WORKTREE_DIR
