"""Plot the label distribution of every engineering landscape.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from biotrainer_core.input_files import read_FASTA
from matplotlib.ticker import FuncFormatter

from al_simulation_container import ALSimulatorDataset

logger = logging.getLogger(__name__)

OUTPUT = Path("plots/engineering/score_distributions.png")
HIT_PERCENTILE = 1.0  # server default; see ActiveLearningScreeningSimulationConfig
NUCB_HIT_CLASSES = ("2", "3")
TARGET_BINS = 60

COLORS = ["#CC0C00", "#5C88DA", "#84BD00", "#FFCD00", "#7C878E", "#00B5E2", "#00AF66"]
INK = "#000000"

DISPLAY = {
    "GB1": "GB1",
    "CREILOV": "CreiLOV",
    "CR9114": "CR9114",
    "MTAGBFP2": "mTagBFP2",
    "SACAS9": "SaCas9",
    "TRPB": "TrpB",
    "NUCB": "NucB",
}
PROPERTY = {
    "GB1": "Binding fitness",
    "CREILOV": "Fluorescence (assay units)",
    "CR9114": "H1 binding",
    "MTAGBFP2": "Combined blue/red fluorescence",
    "SACAS9": "Enzymatic activity",
    "TRPB": "Growth-based fitness",
    "NUCB": "Nuclease activity class",
}
NUCB_LABELS = {"0": "non-functional", "1": "<WT", "2": ">WT", "3": ">A73R"}


def spaced(value: float, decimals: int = 0) -> str:
    """Thousands grouped with a space."""
    return f"{value:,.{decimals}f}".replace(",", " ")


def _format_scale(value: float, _pos: object) -> str:
    return spaced(value) if abs(value) >= 1000 else f"{value:g}"


def _format_log_counts(value: float, _pos: object) -> str:
    return spaced(value) if value >= 1 else ""


def load_labels() -> pl.DataFrame:
    """One row per variant, with the hit rule applied the way the server applies it."""
    frames = []
    for dataset in ALSimulatorDataset.engineering():
        labels = [s.label for s in read_FASTA(dataset.to_path())]
        scores = np.asarray(labels, dtype=float)
        discrete = dataset.definition().optimization_mode.value == "DISCRETE"
        if discrete:
            is_hit, hit_cut = np.isin(labels, NUCB_HIT_CLASSES), math.nan
        else:
            hit_cut = float(np.percentile(scores, 100 - HIT_PERCENTILE))
            is_hit = scores >= hit_cut
        frames.append(
            pl.DataFrame(
                {
                    "dataset": dataset.name.removeprefix("COMBINGYM_").removeprefix(
                        "FLIP2_"
                    ),
                    "discrete": discrete,
                    "label": labels,
                    "score": scores,
                    "is_hit": is_hit,
                    "hit_cut": hit_cut,
                }
            )
        )
    return pl.concat(frames)


def summarize(scores: pl.DataFrame) -> pl.DataFrame:
    """One row per landscape, largest first.

    Notes:
    - `discrete` and `hit_cut` are constant within a landscape, so `first()` reads them back rather than grouping on them."""
    return (
        scores.group_by("dataset")
        .agg(
            discrete=pl.col("discrete").first(),
            hit_cut=pl.col("hit_cut").first(),
            n=pl.len(),
            n_hits=pl.col("is_hit").sum(),
            minimum=pl.col("score").min(),
            median=pl.col("score").median(),
            maximum=pl.col("score").max(),
        )
        .with_columns(hit_share=pl.col("n_hits") / pl.col("n"))
        .sort("n", descending=True)
    )


def edges_through(
    values: np.ndarray, cut: float, target_bins: int = TARGET_BINS
) -> np.ndarray:
    """Uniform-width bin edges with `cut` landing exactly on one of them."""
    low, high = float(values.min()), float(values.max())
    width = (high - low) / target_bins
    index = math.ceil((cut - low) / width)
    first = cut - width * index
    edges = first + width * np.arange(math.ceil((high - first) / width) + 1)
    edges[index] = cut  # exact, rather than reconstructed through floating point
    return edges


def _plot_classes(ax: plt.Axes, panel: pl.DataFrame) -> None:
    """NucB: four ordinal classes. Linear counts with direct labels, because class 3 is
    0.36% of the tallest bar and bars on a log axis start from an arbitrary floor."""
    counts = panel.group_by("label", "is_hit").agg(n=pl.len()).sort("label")
    bars = ax.bar(
        [NUCB_LABELS[v] for v in counts["label"]],
        counts["n"].to_list(),
        color=[COLORS[1] if h else COLORS[0] for h in counts["is_hit"]],
        width=0.7,
    )
    ax.bar_label(bars, labels=[spaced(n) for n in counts["n"]], padding=3, color=INK)
    ax.margins(y=0.20)
    ax.yaxis.set_major_formatter(FuncFormatter(_format_scale))
    ax.tick_params(axis="x", labelrotation=30)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")


def _plot_histogram(ax: plt.Axes, panel: pl.DataFrame, hit_cut: float) -> None:
    ax.set_yscale("log")
    edges = edges_through(panel["score"].to_numpy(), hit_cut)
    for is_hit, colour in ((False, COLORS[0]), (True, COLORS[1])):
        values = panel.filter(pl.col("is_hit") == is_hit)["score"].to_numpy()
        if values.size:
            sns.histplot(x=values, bins=edges, ax=ax, color=colour)
    ax.axvline(hit_cut, color=INK, linewidth=1.1, linestyle="--")
    ax.yaxis.set_major_formatter(FuncFormatter(_format_log_counts))
    ax.xaxis.set_major_formatter(FuncFormatter(_format_scale))


def _label_panel(ax: plt.Axes, row: dict) -> None:
    name = row["dataset"]
    ax.set_xlabel(PROPERTY[name])
    ax.set_ylabel("# Measured Variants")
    ax.set_title(DISPLAY[name], loc="center", fontsize=15, color=INK, pad=22)
    # Above the axes, where it can neither overlap the bars nor be clipped by the panel
    # edge. The hit share is spelled out only for NucB, where it is not the 1% a
    # percentile cut guarantees.
    stats = f"n={spaced(row['n'])} · {spaced(row['n_hits'])} hits"
    stats += (
        f" ({row['hit_share']:.1%})"
        if row["discrete"]
        else f" · {HIT_PERCENTILE:g}% cut {row['hit_cut']:.3g}"
    )
    ax.annotate(
        stats,
        xy=(0.5, 1.015),
        xycoords="axes fraction",
        ha="center",
        va="bottom",
        color=INK,
    )


def build_figure(scores: pl.DataFrame, summary: pl.DataFrame) -> plt.Figure:
    sns.set_theme(style="ticks")
    fig, axes = plt.subplots(4, 2, figsize=(13, 17))

    for ax, row in zip(axes.flat, summary.iter_rows(named=True)):
        panel = scores.filter(pl.col("dataset") == row["dataset"])
        if row["discrete"]:
            _plot_classes(ax, panel)
        else:
            _plot_histogram(ax, panel, row["hit_cut"])
        _label_panel(ax, row)

    legend_ax = axes.flat[summary.height]
    legend_ax.set_axis_off()
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS[0]),
        plt.Rectangle((0, 0), 1, 1, color=COLORS[1]),
    ]
    legend_ax.legend(
        handles,
        ["Below hit cut", "Hit"],
        loc="center",
        frameon=False,
        fontsize=18,
        handlelength=1.8,
        handleheight=1.4,
        labelspacing=1.0,
    )

    fig.suptitle(
        "Score distributions across engineering landscapes", fontsize=19, color=INK
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def main() -> None:
    scores = load_labels()
    summary = summarize(scores)
    logger.info("%s variants across %s landscapes", f"{scores.height:,}", summary.height)
    with pl.Config(tbl_rows=20, tbl_width_chars=200, float_precision=4):
        logger.info("%s", summary)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    build_figure(scores, summary).savefig(OUTPUT, dpi=300)
    logger.info("wrote %s", OUTPUT)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
