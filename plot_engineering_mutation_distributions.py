"""Plot the substitution-count distribution of every engineering landscape."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from matplotlib.ticker import FuncFormatter, MaxNLocator

from plot_engineering_score_distributions import (
    COLORS,
    DISPLAY,
    INK,
    load_labels,
    spaced,
)

logger = logging.getLogger(__name__)

OUTPUT = Path("plots/engineering/mutation_distributions.png")


def _format_log_counts(value: float, _pos: object) -> str:
    return spaced(value) if value >= 1 else ""


def summarize(labels: pl.DataFrame) -> pl.DataFrame:
    """One row per landscape, largest first."""
    return (
        labels.group_by("dataset")
        .agg(
            n=pl.len(),
            n_hits=pl.col("is_hit").sum(),
            positions=pl.col("mutations").max(),
            median=pl.col("mutations").median(),
        )
        .sort("n", descending=True)
    )


def _plot_counts(ax: plt.Axes, panel: pl.DataFrame) -> None:
    """Variants per substitution count, with the hits that sit at each one.

    Hits are overlaid on narrower bars rather than stacked: the counts span four orders
    of magnitude, so the axis is logarithmic and stacked segments would not read as
    their own values.
    """
    counts = (
        panel.group_by("mutations")
        .agg(n=pl.len(), n_hits=pl.col("is_hit").sum())
        .sort("mutations")
    )
    ax.set_yscale("log")
    x = counts["mutations"].to_list()
    ax.bar(x, counts["n"].to_list(), color=COLORS[0], width=0.75)
    hits = counts.filter(pl.col("n_hits") > 0)
    ax.bar(
        hits["mutations"].to_list(),
        hits["n_hits"].to_list(),
        color=COLORS[1],
        width=0.38,
    )
    ax.yaxis.set_major_formatter(FuncFormatter(_format_log_counts))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _label_panel(ax: plt.Axes, row: dict) -> None:
    ax.set_xlabel("# Substitutions from wild type")
    ax.set_ylabel("# Measured Variants")
    ax.set_title(DISPLAY[row["dataset"]], loc="center", fontsize=15, color=INK, pad=22)
    ax.annotate(
        f"n={spaced(row['n'])} · up to {row['positions']} substitutions"
        f" · median {row['median']:g}",
        xy=(0.5, 1.015),
        xycoords="axes fraction",
        ha="center",
        va="bottom",
        color=INK,
    )


def build_figure(labels: pl.DataFrame, summary: pl.DataFrame) -> plt.Figure:
    sns.set_theme(style="ticks")
    fig, axes = plt.subplots(4, 2, figsize=(13, 17))

    for ax, row in zip(axes.flat, summary.iter_rows(named=True)):
        _plot_counts(ax, labels.filter(pl.col("dataset") == row["dataset"]))
        _label_panel(ax, row)

    legend_ax = axes.flat[summary.height]
    legend_ax.set_axis_off()
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS[0]),
        plt.Rectangle((0, 0), 1, 1, color=COLORS[1]),
    ]
    legend_ax.legend(
        handles,
        ["All variants", "Hits"],
        loc="center",
        frameon=False,
        fontsize=18,
        handlelength=1.8,
        handleheight=1.4,
        labelspacing=1.0,
    )

    fig.suptitle(
        "Substitution counts across engineering landscapes", fontsize=19, color=INK
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def main() -> None:
    labels = load_labels()
    summary = summarize(labels)
    logger.info(
        "%s variants across %s landscapes", f"{labels.height:,}", summary.height
    )
    with pl.Config(tbl_rows=20, tbl_width_chars=200, float_precision=4):
        logger.info("%s", summary)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    build_figure(labels, summary).savefig(OUTPUT, dpi=300)
    logger.info("wrote %s", OUTPUT)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
