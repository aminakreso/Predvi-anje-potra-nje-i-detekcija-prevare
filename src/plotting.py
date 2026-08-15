"""Shared figure handling for every script in this project.

Two behaviours live here so all scripts stay consistent:

1. DISPLAY -- pass `--show` (or set SHOW_FIGURES=1) and each figure opens in a
   window as the script runs. The backend must be chosen before pyplot is
   imported, which is why every script imports `plt` from this module rather
   than importing matplotlib directly.

2. NO DUPLICATES -- every figure is kept, and each one is written to a fixed
   filename (e.g. `07_roc_curves.png`). Re-running a script therefore
   OVERWRITES its previous output in place instead of accumulating numbered
   copies, so `reports/figures/` holds exactly one current version of each
   figure no matter how many times you run.

Import as:
    from plotting import emit, plt
"""

import os
import sys
from pathlib import Path

import matplotlib

# Selecting the backend has to happen before pyplot is imported, so the flag
# is read straight from argv rather than waiting for argparse to run.
SHOW = ("--show" in sys.argv) or (os.environ.get("SHOW_FIGURES") == "1")
if not SHOW:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (import must follow backend choice)

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"

_saved: list[Path] = []


def emit(fig=None, name: str = "figure.png", dpi: int = 150) -> Path:
    """Display the figure and save it under a fixed, reusable filename.

    `fig` may be None for libraries such as SHAP that draw onto the current
    figure instead of handing one back.
    """
    if fig is None:
        fig = plt.gcf()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    # Stable filename => a re-run replaces this exact file rather than adding
    # another copy alongside it.
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    _saved.append(path)

    if SHOW:
        plt.show()  # blocks until the window is closed

    plt.close(fig)
    return path


def saved_figures() -> list[Path]:
    """Figures written during this run, in the order they were produced."""
    return list(_saved)


def report_saved() -> str:
    """One-line summary for the end of a script."""
    return f"Saved {len(_saved)} figure(s) to {FIG_DIR}"


def add_show_argument(parser) -> None:
    """Register `--show` so argparse does not reject it."""
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display each figure in a window as it is produced",
    )
