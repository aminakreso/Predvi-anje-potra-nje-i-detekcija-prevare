"""Shared figure handling for every script in this project.

Three behaviours live here so all scripts stay consistent:

1. DISPLAY -- pass `--show` (or set SHOW_FIGURES=1) and each figure opens in a
   window as the script runs. The backend must be chosen before pyplot is
   imported, which is why every script imports `plt` from this module rather
   than importing matplotlib directly.

2. NO DUPLICATES -- every figure is kept, and each one is written to a fixed
   filename (e.g. `07_roc_curves.png`). Re-running a script therefore
   OVERWRITES its previous output in place instead of accumulating numbered
   copies, so `reports/figures/` holds exactly one current version of each
   figure no matter how many times you run.

3. THESIS STYLE -- figures are saved at 300 dpi (print quality) and numeric
   tick labels use a decimal comma (0,5 rather than 0.5), matching the
   Bosnian text of the thesis. Use `num()` for numbers placed in titles,
   legends and annotations.

Import as:
    from plotting import emit, num, plt
"""

import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

# Windows consoles default to a legacy code page without č, ć and đ, so
# printing a Bosnian label would crash the script.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Selecting the backend has to happen before pyplot is imported, so the flag
# is read straight from argv rather than waiting for argparse to run.
SHOW = ("--show" in sys.argv) or (os.environ.get("SHOW_FIGURES") == "1")
if not SHOW:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (import must follow backend choice)
from matplotlib.ticker import ScalarFormatter  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
DPI = 300

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
})

_saved: list[Path] = []


def num(value: float, decimals: int = 3) -> str:
    """Format a number with a decimal comma: num(0.9517) -> '0,952'."""
    return f"{value:.{decimals}f}".replace(".", ",")


class _CommaFormatter(ScalarFormatter):
    """Matplotlib's default tick formatter, but with a decimal comma."""

    def __call__(self, x, pos=None):
        return super().__call__(x, pos).replace(".", ",")


def _use_decimal_comma(fig) -> None:
    # Only plain numeric axes are touched: categorical, log-scale and
    # hand-labelled axes use other formatter classes and are left alone.
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if type(axis.get_major_formatter()) is ScalarFormatter:
                axis.set_major_formatter(_CommaFormatter())


def emit(fig=None, name: str = "figure.png", dpi: int = DPI) -> Path:
    """Display the figure and save it under a fixed, reusable filename.

    `fig` may be None for libraries such as SHAP that draw onto the current
    figure instead of handing one back.
    """
    if fig is None:
        fig = plt.gcf()

    _use_decimal_comma(fig)

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


def confusion_matrix_figure(cm: np.ndarray, class_labels: list[str], title: str):
    """Confusion matrix with counts and the share of each actual class."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    row_totals = cm.sum(axis=1, keepdims=True)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            share = cm[i, j] / row_totals[i, 0] * 100 if row_totals[i, 0] else 0.0
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]:,}".replace(",", ".") + f"\n({num(share, 1)} %)",
                    ha="center", va="center", color=color, fontsize=10)
    ax.set_xticks([0, 1], class_labels)
    ax.set_yticks([0, 1], class_labels)
    ax.set_xlabel("Predviđena klasa")
    ax.set_ylabel("Stvarna klasa")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Broj primjera")
    return fig


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
