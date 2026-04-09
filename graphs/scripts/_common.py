"""
Shared utilities for paper-ready plotting scripts under `graphs/scripts/`.

Design goals:
- Single-file runnable scripts from repo root.
- Matplotlib-only styling (IEEE-friendly).
- Graceful missing-artifact handling (no fabricated data).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

# Ensure Matplotlib/fontconfig caches are writable when running on shared/remote systems.
REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))

import matplotlib as mpl


GRAPHS_DIR = REPO_ROOT / "graphs"
FIG_DIR = GRAPHS_DIR / "figures"
DATA_DIR = GRAPHS_DIR / "data"


def apply_ieee_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 600,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "lines.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / ".cache").mkdir(parents=True, exist_ok=True)


def save_figure(fig, stem: str, out_dir: Optional[Path] = None) -> list[Path]:
    ensure_dirs()
    out_dir = out_dir or FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for ext in ("png", "pdf"):
        out_path = out_dir / f"{stem}.{ext}"
        fig.savefig(out_path, bbox_inches="tight")
        paths.append(out_path)
    return paths


def print_inputs_used(paths: Iterable[os.PathLike | str]) -> None:
    paths = list(paths)
    if not paths:
        return
    print("Inputs used:")
    for p in paths:
        print(f"  - {p}")


def watermark_placeholder(ax, text: str = "PLACEHOLDER / TBD") -> None:
    ax.text(
        0.5,
        0.5,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=14,
        color="0.6",
        rotation=20,
        alpha=0.6,
    )
