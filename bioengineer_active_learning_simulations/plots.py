from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd


def _extract_consecutive_failures_threshold(stop_reasons: Optional[List[str]]) -> Optional[int]:
    if not stop_reasons:
        return None
    for reason in stop_reasons:
        if "consecutive failures" in reason.lower():
            match = re.search(r"\((\d+)\)", reason)
            if match:
                return int(match.group(1))
    return None


def _align_iteration_lengths(*lists: Optional[List]) -> Tuple[int, List[List]]:
    lengths = [len(lst or []) for lst in lists]
    n = max(lengths) if lengths else 0
    aligned = [(lst or [])[:n] for lst in lists]
    return n, aligned


def _compute_convergence_stats(
    per_sim_n_hits: List[List[int]],
    n_hits_threshold: int,
) -> Tuple[List[int], Optional[float], Optional[float]]:
    convergence_iterations: List[int] = []
    for successes in per_sim_n_hits:
        if not successes:
            continue
        cumsum = np.cumsum(successes)
        converged_at = np.where(cumsum >= n_hits_threshold)[0]
        if len(converged_at) > 0:
            convergence_iterations.append(int(converged_at[0]) + 1)
    if not convergence_iterations:
        return convergence_iterations, None, None
    return (
        convergence_iterations,
        float(np.mean(convergence_iterations)),
        float(np.std(convergence_iterations)),
    )


def build_metric_evolution_chart(
    iteration_metrics_total: List[float],
    iteration_metrics_suggestions: List[float],
    n_iteration_hits: List[int],
    metric_name: str,
) -> alt.LayerChart:
    n, (total, suggestions, successes) = _align_iteration_lengths(
        iteration_metrics_total,
        iteration_metrics_suggestions,
        n_iteration_hits,
    )
    iterations = list(range(1, n + 1))

    rows: List[Dict] = []
    for i, v in enumerate(total):
        rows.append({"iteration": iterations[i], "series": "All Data", "value": float(v)})
    for i, v in enumerate(suggestions):
        rows.append({"iteration": iterations[i], "series": "Suggestions", "value": float(v)})
    df = pd.DataFrame(rows)

    line = (
        alt.Chart(df)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("iteration:Q", title="Iteration"),
            y=alt.Y("value:Q", title=f"Metric ({metric_name})"),
            color=alt.Color("series:N", title="Series"),
            tooltip=["iteration:Q", "series:N", alt.Tooltip("value:Q", format=".4f")],
        )
    )

    success_iters = [iterations[i] for i, s in enumerate(successes) if s > 0]
    success_values = [
        total[i] for i, s in enumerate(successes) if s > 0 and i < len(total)
    ]
    layers: List[alt.Chart] = [line]
    if success_iters and success_values:
        stars_df = pd.DataFrame(
            {"iteration": success_iters[: len(success_values)], "value": success_values}
        )
        stars = (
            alt.Chart(stars_df)
            .mark_point(shape="diamond", size=220, color="gold", filled=True, stroke="black", strokeWidth=1.2)
            .encode(
                x="iteration:Q",
                y="value:Q",
                tooltip=[alt.Tooltip("iteration:Q", title="Iteration"), alt.Tooltip("value:Q", title="Metric", format=".4f")],
            )
        )
        layers.append(stars)

    return alt.layer(*layers).properties(
        title="Metric Comparison with Success Markers", height=320
    )


def build_n_hits_chart(n_iteration_hits: List[int]) -> alt.LayerChart:
    successes = n_iteration_hits or []
    n_iterations = len(successes)
    iterations = list(range(1, n_iterations + 1))
    cumulative = np.cumsum(successes).tolist() if successes else []

    df = pd.DataFrame(
        {
            "iteration": iterations,
            "per_iteration": successes,
            "cumulative": cumulative,
        }
    )

    bars = (
        alt.Chart(df)
        .mark_bar(opacity=0.35, color="#7fc97f")
        .encode(
            x=alt.X("iteration:Q", title="Iteration"),
            y=alt.Y("per_iteration:Q", title="Target Successes"),
            tooltip=[
                alt.Tooltip("iteration:Q", title="Iteration"),
                alt.Tooltip("per_iteration:Q", title="Per Iteration"),
                alt.Tooltip("cumulative:Q", title="Cumulative"),
            ],
        )
    )
    line = (
        alt.Chart(df)
        .mark_line(point=True, color="#1b7837", strokeWidth=2)
        .encode(
            x="iteration:Q",
            y=alt.Y("cumulative:Q", title="Target Successes"),
            tooltip=[
                alt.Tooltip("iteration:Q", title="Iteration"),
                alt.Tooltip("cumulative:Q", title="Cumulative"),
            ],
        )
    )
    return alt.layer(bars, line).properties(
        title="Target Successes Over Iterations", height=320
    )


def build_consecutive_failures_chart(
    iteration_consecutive_failures: List[int],
    stop_reasons: Optional[List[str]],
) -> alt.LayerChart:
    failures = iteration_consecutive_failures or []
    iterations = list(range(0, len(failures) + 1))
    values = [0] + list(failures)
    df = pd.DataFrame({"iteration": iterations, "failures": values})

    step = (
        alt.Chart(df)
        .mark_line(interpolate="step-after", color="#d7301f", strokeWidth=2, point=True)
        .encode(
            x=alt.X("iteration:Q", title="Iteration"),
            y=alt.Y("failures:Q", title="Consecutive Failures"),
            tooltip=["iteration:Q", "failures:Q"],
        )
    )
    layers: List[alt.Chart] = [step]

    threshold = _extract_consecutive_failures_threshold(stop_reasons)
    if threshold is not None:
        rule_df = pd.DataFrame({"y": [threshold], "label": [f"Threshold ({threshold})"]})
        rule = (
            alt.Chart(rule_df)
            .mark_rule(color="#990000", strokeDash=[6, 4])
            .encode(y="y:Q", tooltip=["label:N"])
        )
        layers.append(rule)

    return alt.layer(*layers).properties(
        title="Consecutive Failures Tracking", height=320
    )


def build_suggested_labels_chart(
    iteration_suggestions: List[List[str]],
    id2label: Dict[str, str],
    unique_labels: List[str],
    optimization_targets: Optional[List[str]],
) -> alt.Chart:
    rows: List[Dict] = []
    targets = set(optimization_targets or [])
    for iter_idx, sugg_ids in enumerate(iteration_suggestions, start=1):
        iter_labels = [id2label.get(sid, "<unknown>") for sid in sugg_ids]
        counts = {label: iter_labels.count(label) for label in unique_labels}
        for label, count in counts.items():
            display = f"★ {label}" if label in targets else label
            rows.append(
                {
                    "iteration": iter_idx,
                    "label": label,
                    "label_display": display,
                    "count": count,
                    "is_target": label in targets,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return alt.Chart(pd.DataFrame({"iteration": [], "count": []})).mark_bar().properties(
            title="Distribution of Suggested Labels", height=320
        )

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("iteration:O", title="Iteration"),
            y=alt.Y("count:Q", title="Suggested Labels (per iteration)", stack="zero"),
            color=alt.Color("label_display:N", title="Labels"),
            tooltip=[
                alt.Tooltip("iteration:O", title="Iteration"),
                alt.Tooltip("label_display:N", title="Label"),
                alt.Tooltip("count:Q", title="Count"),
            ],
        )
        .properties(title="Distribution of Suggested Labels", height=320)
    )
    return chart


def build_performance_summary_chart(
    success_count: int,
    total_runs: int,
    convergence_iterations: List[int],
    max_iterations: int,
) -> alt.Chart:
    success_rate = (success_count / total_runs * 100.0) if total_runs > 0 else 0.0
    failure_rate = 100.0 - success_rate

    if convergence_iterations:
        avg_conv = float(np.mean(convergence_iterations))
        std_conv = float(np.std(convergence_iterations))
        score = max(0.0, 100.0 * (1.0 - avg_conv / max(max_iterations, 1)))
        conv_label = f"Convergence Speed ({len(convergence_iterations)} of {total_runs} converged)"
        conv_annotation = f"Avg: {avg_conv:.1f} ± {std_conv:.1f} iter"
    else:
        score = 0.0
        conv_label = "Convergence Speed (none converged)"
        conv_annotation = "n/a"

    rows = [
        {"row": f"Success Rate ({success_count}/{total_runs} runs)", "segment": "Success/Fast", "value": success_rate, "annotation": f"{success_rate:.1f}%"},
        {"row": f"Success Rate ({success_count}/{total_runs} runs)", "segment": "Failure/Slow", "value": failure_rate, "annotation": f"{failure_rate:.1f}%"},
        {"row": conv_label, "segment": "Success/Fast", "value": score, "annotation": conv_annotation},
        {"row": conv_label, "segment": "Failure/Slow", "value": 100.0 - score, "annotation": ""},
    ]
    df = pd.DataFrame(rows)

    color_scale = alt.Scale(
        domain=["Success/Fast", "Failure/Slow"],
        range=["#2ecc71", "#e74c3c"],
    )

    bars = (
        alt.Chart(df)
        .mark_bar(stroke="black", strokeWidth=0.5)
        .encode(
            x=alt.X("value:Q", stack="zero", title="Percentage / Score", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("row:N", title=None, sort=None),
            color=alt.Color("segment:N", scale=color_scale, title="Segment"),
            tooltip=[
                alt.Tooltip("row:N", title="Metric"),
                alt.Tooltip("segment:N", title="Segment"),
                alt.Tooltip("value:Q", title="Value", format=".1f"),
                alt.Tooltip("annotation:N", title="Detail"),
            ],
        )
    )

    text_df = df[df["segment"] == "Success/Fast"].copy()
    text_df["x_mid"] = text_df["value"] / 2.0
    text = (
        alt.Chart(text_df)
        .mark_text(color="black", fontWeight="bold")
        .encode(
            x=alt.X("x_mid:Q"),
            y=alt.Y("row:N", sort=None),
            text="annotation:N",
        )
    )

    return alt.layer(bars, text).properties(
        title="Overall Performance Summary", height=180
    )


def build_mean_cumulative_successes_chart(
    per_sim_n_hits: List[List[int]],
    per_sim_is_success: List[bool],
    target_threshold: int,
) -> alt.LayerChart:
    if not per_sim_n_hits:
        return alt.layer(alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_line()).properties(
            title="Target Discovery Progress", height=320
        )

    max_iter = max(len(s) for s in per_sim_n_hits)
    iterations = np.arange(1, max_iter + 1)

    padded = np.array(
        [list(s) + [0] * (max_iter - len(s)) for s in per_sim_n_hits],
        dtype=float,
    )
    cumulative = np.cumsum(padded, axis=1)
    mean_cum = cumulative.mean(axis=0)
    std_cum = cumulative.std(axis=0)

    indiv_rows = []
    for sim_idx, row in enumerate(cumulative):
        for it_idx, value in enumerate(row):
            indiv_rows.append(
                {
                    "iteration": int(iterations[it_idx]),
                    "value": float(value),
                    "sim": sim_idx,
                    "is_success": bool(per_sim_is_success[sim_idx]) if sim_idx < len(per_sim_is_success) else False,
                }
            )
    indiv_df = pd.DataFrame(indiv_rows)

    summary_df = pd.DataFrame(
        {
            "iteration": iterations,
            "mean": mean_cum,
            "low": np.maximum(0, mean_cum - std_cum),
            "high": mean_cum + std_cum,
        }
    )

    individual = (
        alt.Chart(indiv_df)
        .mark_line(opacity=0.25, strokeWidth=1)
        .encode(
            x=alt.X("iteration:Q", title="Iteration"),
            y=alt.Y("value:Q", title="Cumulative Target Successes"),
            detail="sim:N",
            color=alt.Color(
                "is_success:N",
                scale=alt.Scale(domain=[True, False], range=["#2ecc71", "#e74c3c"]),
                legend=alt.Legend(title="Run success"),
            ),
            tooltip=[
                alt.Tooltip("iteration:Q", title="Iteration"),
                alt.Tooltip("value:Q", title="Cumulative"),
                alt.Tooltip("sim:N", title="Sim #"),
                alt.Tooltip("is_success:N", title="Success"),
            ],
        )
    )

    band = (
        alt.Chart(summary_df)
        .mark_area(opacity=0.3, color="#16a085")
        .encode(
            x="iteration:Q",
            y="low:Q",
            y2="high:Q",
        )
    )
    mean_line = (
        alt.Chart(summary_df)
        .mark_line(point=True, strokeWidth=3, color="#16a085")
        .encode(
            x="iteration:Q",
            y=alt.Y("mean:Q", title="Cumulative Target Successes"),
            tooltip=[
                alt.Tooltip("iteration:Q", title="Iteration"),
                alt.Tooltip("mean:Q", title="Mean", format=".2f"),
                alt.Tooltip("low:Q", title="-1 SD", format=".2f"),
                alt.Tooltip("high:Q", title="+1 SD", format=".2f"),
            ],
        )
    )
    rule_df = pd.DataFrame({"y": [target_threshold], "label": [f"Target ({target_threshold})"]})
    target_rule = (
        alt.Chart(rule_df)
        .mark_rule(color="red", strokeDash=[6, 4], strokeWidth=2)
        .encode(y="y:Q", tooltip=["label:N"])
    )

    return alt.layer(individual, band, mean_line, target_rule).properties(
        title="Target Discovery Progress", height=360
    )


def build_mean_metric_evolution_chart(
    per_sim_metrics_total: List[List[float]],
    best_run_metrics: List[float],
    worst_run_metrics: List[float],
    metric_name: str,
) -> alt.LayerChart:
    if not per_sim_metrics_total:
        return alt.layer(alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_line()).properties(
            title=f"Model Performance Evolution ({metric_name})", height=320
        )

    max_iter = max(len(m) for m in per_sim_metrics_total)
    iterations = np.arange(1, max_iter + 1)

    padded = np.array(
        [list(m) + [np.nan] * (max_iter - len(m)) for m in per_sim_metrics_total],
        dtype=float,
    )
    mean = np.nanmean(padded, axis=0)
    std = np.nanstd(padded, axis=0)

    summary_df = pd.DataFrame(
        {
            "iteration": iterations,
            "mean": mean,
            "low": mean - std,
            "high": mean + std,
        }
    )

    band = (
        alt.Chart(summary_df)
        .mark_area(opacity=0.3, color="#3498db")
        .encode(x="iteration:Q", y="low:Q", y2="high:Q")
    )
    mean_line = (
        alt.Chart(summary_df)
        .mark_line(point=True, strokeWidth=3, color="#3498db")
        .encode(
            x=alt.X("iteration:Q", title="Iteration"),
            y=alt.Y("mean:Q", title=metric_name),
            tooltip=[
                alt.Tooltip("iteration:Q", title="Iteration"),
                alt.Tooltip("mean:Q", title="Mean", format=".4f"),
                alt.Tooltip("low:Q", title="-1 SD", format=".4f"),
                alt.Tooltip("high:Q", title="+1 SD", format=".4f"),
            ],
        )
    )

    overlay_rows: List[Dict] = []
    for it_idx, value in enumerate(best_run_metrics or [], start=1):
        overlay_rows.append({"iteration": it_idx, "value": float(value), "run": "Best Run"})
    for it_idx, value in enumerate(worst_run_metrics or [], start=1):
        overlay_rows.append({"iteration": it_idx, "value": float(value), "run": "Worst Run"})
    overlay_df = pd.DataFrame(overlay_rows)

    overlay = (
        alt.Chart(overlay_df)
        .mark_line(strokeDash=[6, 4], point=True, opacity=0.7, strokeWidth=2)
        .encode(
            x="iteration:Q",
            y=alt.Y("value:Q", title=metric_name),
            color=alt.Color(
                "run:N",
                scale=alt.Scale(domain=["Best Run", "Worst Run"], range=["#2ecc71", "#e74c3c"]),
                title="Run",
            ),
            tooltip=[
                alt.Tooltip("iteration:Q", title="Iteration"),
                alt.Tooltip("run:N", title="Run"),
                alt.Tooltip("value:Q", title=metric_name, format=".4f"),
            ],
        )
    )

    return alt.layer(band, mean_line, overlay).properties(
        title=f"Model Performance Evolution ({metric_name})", height=360
    )


def build_cross_experiment_cumulative_chart(
    experiments: Dict[str, List[List[int]]],
    targets: Dict[str, int],
) -> alt.LayerChart:
    rows: List[Dict] = []
    rule_rows: List[Dict] = []
    for name, per_sim in experiments.items():
        if not per_sim:
            continue
        max_iter = max(len(s) for s in per_sim)
        padded = np.array(
            [list(s) + [0] * (max_iter - len(s)) for s in per_sim], dtype=float
        )
        cumulative = np.cumsum(padded, axis=1)
        mean = cumulative.mean(axis=0)
        std = cumulative.std(axis=0)
        for it_idx in range(max_iter):
            rows.append(
                {
                    "experiment": name,
                    "iteration": it_idx + 1,
                    "mean": float(mean[it_idx]),
                    "low": float(max(0.0, mean[it_idx] - std[it_idx])),
                    "high": float(mean[it_idx] + std[it_idx]),
                }
            )
        if name in targets:
            rule_rows.append({"experiment": name, "target": targets[name]})

    df = pd.DataFrame(rows)
    if df.empty:
        return alt.layer(alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_line()).properties(
            title="Cross-Experiment: Cumulative Target Successes (mean)", height=360
        )

    band = (
        alt.Chart(df)
        .mark_area(opacity=0.15)
        .encode(
            x=alt.X("iteration:Q", title="Iteration"),
            y="low:Q",
            y2="high:Q",
            color=alt.Color("experiment:N", title="Experiment"),
        )
    )
    line = (
        alt.Chart(df)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x="iteration:Q",
            y=alt.Y("mean:Q", title="Cumulative Target Successes (mean)"),
            color=alt.Color("experiment:N", title="Experiment"),
            tooltip=[
                alt.Tooltip("experiment:N", title="Experiment"),
                alt.Tooltip("iteration:Q", title="Iteration"),
                alt.Tooltip("mean:Q", title="Mean", format=".2f"),
                alt.Tooltip("low:Q", title="-1 SD", format=".2f"),
                alt.Tooltip("high:Q", title="+1 SD", format=".2f"),
            ],
        )
    )

    layers: List[alt.Chart] = [band, line]
    if rule_rows:
        rule_df = pd.DataFrame(rule_rows)
        rules = (
            alt.Chart(rule_df)
            .mark_rule(strokeDash=[6, 4], opacity=0.6)
            .encode(
                y="target:Q",
                color=alt.Color("experiment:N", title="Experiment"),
                tooltip=[
                    alt.Tooltip("experiment:N", title="Experiment"),
                    alt.Tooltip("target:Q", title="Target"),
                ],
            )
        )
        layers.append(rules)

    return alt.layer(*layers).properties(
        title="Cross-Experiment: Cumulative Target Successes (mean)", height=380
    )


def build_cross_experiment_metric_chart(
    experiments: Dict[str, List[List[float]]],
    metric_name: str,
) -> alt.LayerChart:
    rows: List[Dict] = []
    for name, per_sim in experiments.items():
        if not per_sim:
            continue
        max_iter = max(len(m) for m in per_sim)
        padded = np.array(
            [list(m) + [np.nan] * (max_iter - len(m)) for m in per_sim], dtype=float
        )
        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        for it_idx in range(max_iter):
            rows.append(
                {
                    "experiment": name,
                    "iteration": it_idx + 1,
                    "mean": float(mean[it_idx]) if not np.isnan(mean[it_idx]) else None,
                    "low": float(mean[it_idx] - std[it_idx]) if not np.isnan(mean[it_idx]) else None,
                    "high": float(mean[it_idx] + std[it_idx]) if not np.isnan(mean[it_idx]) else None,
                }
            )

    df = pd.DataFrame(rows).dropna()
    if df.empty:
        return alt.layer(alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_line()).properties(
            title=f"Cross-Experiment: {metric_name} (mean)", height=360
        )

    band = (
        alt.Chart(df)
        .mark_area(opacity=0.15)
        .encode(
            x=alt.X("iteration:Q", title="Iteration"),
            y="low:Q",
            y2="high:Q",
            color=alt.Color("experiment:N", title="Experiment"),
        )
    )
    line = (
        alt.Chart(df)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x="iteration:Q",
            y=alt.Y("mean:Q", title=f"{metric_name} (mean)"),
            color=alt.Color("experiment:N", title="Experiment"),
            tooltip=[
                alt.Tooltip("experiment:N", title="Experiment"),
                alt.Tooltip("iteration:Q", title="Iteration"),
                alt.Tooltip("mean:Q", title="Mean", format=".4f"),
                alt.Tooltip("low:Q", title="-1 SD", format=".4f"),
                alt.Tooltip("high:Q", title="+1 SD", format=".4f"),
            ],
        )
    )

    return alt.layer(band, line).properties(
        title=f"Cross-Experiment: {metric_name} (mean)", height=380
    )
