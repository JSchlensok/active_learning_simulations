"""Streamlit dashboard for active learning simulation results.

Run with:
    uv run streamlit run al_sim_dashboard.py
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st
import pandas as pd
import numpy as np

import al_plots
from al_simulation_container import ALSimulatorDataset, get_simulator
from al_simulator import (
    ActiveLearningSimulator,
    DashboardCompressedData,
    DashboardExperimentData,
)

RESULTS_DIR = Path("simulation_v1_results")
COMPRESSED_DATA_PATH = RESULTS_DIR / "compressed_dashboard_data.json"

st.set_page_config(page_title="AL Simulation Dashboard", layout="wide")


@st.cache_data(show_spinner=False)
def _load_compressed() -> Optional[DashboardCompressedData]:
    if not COMPRESSED_DATA_PATH.exists():
        return None
    try:
        return DashboardCompressedData.model_validate_json(COMPRESSED_DATA_PATH.read_text())
    except Exception as exc:
        st.sidebar.error(f"Failed to load compressed data: {exc}")
        return None


@st.cache_data(show_spinner=True)
def _load_simulator(selected_dataset: str) -> ActiveLearningSimulator:
    al_sim_dataset = ALSimulatorDataset[selected_dataset]
    return get_simulator(al_sim_dataset)


def _render_aggregate_from_compressed(exp: DashboardExperimentData) -> None:
    st.subheader("Overall performance summary")

    cols = st.columns(4)
    cols[0].metric("Total number of hits", exp.aggregated_hits)
    cols[1].metric("Total number of suggestions", exp.aggregated_suggestions)

    total_sims = len(exp.per_sim_is_success)
    avg_hits = exp.aggregated_hits / total_sims if total_sims > 0 else 0
    success_rate = sum(1 for s in exp.per_sim_is_success if s) / total_sims if total_sims > 0 else 0

    cols[2].metric("Avg. hits per campaign", f"{avg_hits:.2f}")
    cols[3].metric("Success rate", f"{success_rate:.1%}")

    summary = exp.summary
    is_discrete = summary["is_discrete"]
    metric_name = "Accuracy" if is_discrete else "RMSE"

    # We need to compute stats for the charts
    max_iterations = max((len(m) for m in exp.per_sim_metrics_total), default=0)
    success_count = sum(1 for s in exp.per_sim_is_success if s)

    convergence_iterations, _, _ = al_plots._compute_convergence_stats(
        per_sim_n_hits=exp.per_sim_n_hits,
        n_hits_threshold=summary["n_hits_threshold"],
    )

    hits_per_sim = [sum(s) for s in exp.per_sim_n_hits]
    best_idx = np.argmax(hits_per_sim)
    worst_idx = np.argmin(hits_per_sim)

    perf_summary_chart = al_plots.build_performance_summary_chart(
        success_count=success_count,
        total_runs=len(exp.per_sim_is_success),
        convergence_iterations=convergence_iterations,
        max_iterations=max_iterations,
    )
    st.altair_chart(perf_summary_chart.interactive(), use_container_width=True)

    st.subheader("Target discovery progress")
    cum_success_chart = al_plots.build_mean_cumulative_successes_chart(
        per_sim_n_hits=exp.per_sim_n_hits,
        per_sim_is_success=exp.per_sim_is_success,
        target_threshold=summary["n_hits_threshold"],
    )
    st.altair_chart(cum_success_chart.interactive(), use_container_width=True)

    st.subheader("Model performance evolution")
    metric_evo_chart = al_plots.build_mean_metric_evolution_chart(
        per_sim_metrics_total=exp.per_sim_metrics_total,
        best_run_metrics=exp.per_sim_metrics_total[best_idx],
        worst_run_metrics=exp.per_sim_metrics_total[worst_idx],
        metric_name=metric_name,
    )
    st.altair_chart(metric_evo_chart.interactive(), use_container_width=True)

    # Added: chart with suggestions metric overlay if available
    if exp.per_sim_metrics_suggestions:
        st.subheader(f"Model performance evolution (Suggestions) — mean per simulation")
        # Build a custom overlay chart? 
        # Actually al_plots.build_mean_metric_evolution_chart doesn't seem to support suggestions
        # But we can at least show the mean suggestions metric
        suggestions_chart = al_plots.build_mean_metric_evolution_chart(
            per_sim_metrics_total=exp.per_sim_metrics_suggestions,
            best_run_metrics=exp.per_sim_metrics_suggestions[best_idx],
            worst_run_metrics=exp.per_sim_metrics_suggestions[worst_idx],
            metric_name=f"{metric_name} (Suggestions)",
        )
        st.altair_chart(suggestions_chart.interactive(), use_container_width=True)


def _render_single_from_compressed(sim_data, summary: dict) -> None:
    stop_reasons = sim_data.stop_reasons
    cols_row1 = st.columns(2)
    cols_row1[0].metric("Seed", sim_data.seed)
    cols_row1[1].metric("Success", sim_data.is_success)
    st.metric("Primary Stop Reason", stop_reasons[0])

    metric_name = "Accuracy" if summary["is_discrete"] else "RMSE"

    n_iteration_hits = list(map(len, sim_data.iteration_hits))

    chart_metric = al_plots.build_metric_evolution_chart(
        iteration_metrics_total=sim_data.iteration_metrics_total,
        iteration_metrics_suggestions=sim_data.iteration_metrics_suggestions,
        n_iteration_hits=n_iteration_hits,
        metric_name=metric_name
    )

    chart_failures = al_plots.build_consecutive_failures_chart(
        iteration_consecutive_failures=sim_data.iteration_consecutive_failures,
        stop_reasons=stop_reasons
    )

    chart_successes = al_plots.build_n_hits_chart(
        n_iteration_hits=n_iteration_hits
    )

    col_left, col_right = st.columns(2)
    with col_left:
        st.altair_chart(chart_metric.interactive(), use_container_width=True)
        st.altair_chart(chart_failures.interactive(), use_container_width=True)
    with col_right:
        st.altair_chart(chart_successes.interactive(), use_container_width=True)

    with st.expander("Raw stats"):
        st.markdown(f"- **Stop reasons:** {stop_reasons}")
        st.markdown(f"- **Total iterations:** {sim_data.iteration_results_count}")
        st.markdown(f"- **iteration_metrics_total:** `{sim_data.iteration_metrics_total}`")
        st.markdown(f"- **iteration_metrics_suggestions:** `{sim_data.iteration_metrics_suggestions}`")
        st.markdown(f"- **n_iteration_hits:** `{n_iteration_hits}`")
        st.markdown(f"- **iteration_consecutive_failures:** `{sim_data.iteration_consecutive_failures}`")


def _extract_consecutive_failures_threshold(stop_reasons: List[str]) -> Optional[int]:
    for reason in stop_reasons:
        if "consecutive failures" in reason.lower():
            match = re.search(r"\((\d+)\)", reason)
            if match:
                return int(match.group(1))
    return None


def _render_comparison_from_compressed(subset: List[DashboardExperimentData]) -> None:
    st.header("Comparison")
    mode = st.radio("Compare by", ["Surrogate Model", "Embedder"], horizontal=True)

    group_by = "model" if mode == "Surrogate Model" else "embedder"

    groups: Dict[str, List[DashboardExperimentData]] = {}
    for exp in subset:
        key = getattr(exp, group_by)
        if key not in groups:
            groups[key] = []
        if group_by == "embedder" and exp.model == "RANDOM":
            continue  # Exclude random surrogate models from embedder comparison
        groups[key].append(exp)

    comparison_data = []
    cross_cum_inputs = {}
    cross_metric_inputs = {}
    cross_targets = {}
    metric_names = set()

    for key, exps in groups.items():
        all_hits = sum(e.aggregated_hits for e in exps)
        all_suggestions = sum(e.aggregated_suggestions for e in exps)
        all_is_success = [s for e in exps for s in e.per_sim_is_success]
        total_simulations = len(all_is_success)

        conv_speeds = []
        for e in exps:
            target_threshold = e.summary["n_hits_threshold"]
            for i, is_succ in enumerate(e.per_sim_is_success):
                if is_succ:
                    cumsum = np.cumsum(e.per_sim_n_hits[i])
                    converged_at = np.where(cumsum >= target_threshold)[0]
                    if len(converged_at) > 0:
                        conv_speeds.append(int(converged_at[0]) + 1)

        avg_conv = np.mean(conv_speeds) if conv_speeds else None

        comparison_data.append({
            mode: key,
            "Total Hits": all_hits,
            "Total Suggestions": all_suggestions,
            "Hit Rate": f"{all_hits / all_suggestions:.2%}" if all_suggestions > 0 else "0%",
            "Avg. Hits per Campaign": f"{all_hits / total_simulations:.2f}" if total_simulations > 0 else "0.00",
            "Percentage of successful campaigns": f"{sum(all_is_success) / total_simulations:.1%}" if total_simulations > 0 else "0.0%",
            "Avg. Convergence Iteration": f"{avg_conv:.2f}" if avg_conv is not None else "N/A"
        })

    if group_by == "embedder":
        st.markdown("*Note: Random surrogate models are excluded from the comparison.*")
    st.table(pd.DataFrame(comparison_data))

    st.subheader("Visual Comparison")
    for exp in subset:
        cross_cum_inputs[exp.name] = exp.per_sim_n_hits
        cross_metric_inputs[exp.name] = exp.per_sim_metrics_total
        cross_targets[exp.name] = exp.summary["n_hits_threshold"]
        metric_names.add("Accuracy" if exp.summary["is_discrete"] else "RMSE")

    metric_name = ", ".join(sorted(metric_names)) if metric_names else "Metric"

    st.subheader("Cumulative target successes — mean per experiment")
    st.altair_chart(
        al_plots.build_cross_experiment_cumulative_chart(cross_cum_inputs, cross_targets).interactive(),
        use_container_width=True,
    )
    st.subheader(f"Model performance ({metric_name}) — mean per experiment")
    st.altair_chart(
        al_plots.build_cross_experiment_metric_chart(cross_metric_inputs, metric_name).interactive(),
        use_container_width=True,
    )


def _render_header_from_compressed(exp: DashboardExperimentData) -> None:
    summary = exp.summary
    st.markdown(f"### `{exp.name}`")
    cols_row1 = st.columns(3)
    cols_row1[0].metric("Embedder", exp.embedder)
    cols_row1[1].metric("Model", exp.model)
    cols_row1[2].metric("Mode", str(summary["optimization_mode"]))

    cols_row2 = st.columns(3)
    cols_row2[0].metric("Sims", summary["n_simulations"])
    cols_row2[1].metric("Successful", f"{summary['n_successful']}/{summary['n_simulations']}")
    cols_row2[2].metric("Target threshold", summary["n_hits_threshold"])
    if summary.get("discrete_targets"):
        st.caption(f"Discrete targets: {', '.join(summary['discrete_targets'])}")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Simulation results")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("Active Learning Simulation Dashboard")

compressed_data = _load_compressed()

if compressed_data:
    st.sidebar.success("Loaded pre-calculated results.")
    # Group by dataset
    ds_groups: Dict[str, List[DashboardExperimentData]] = {}
    for exp in compressed_data.experiments:
        if exp.dataset not in ds_groups:
            ds_groups[exp.dataset] = []
        ds_groups[exp.dataset].append(exp)

    selected_dataset = st.sidebar.selectbox("Select Dataset", sorted(ds_groups.keys()))
    al_simulator = _load_simulator(selected_dataset)
    base_config = al_simulator.base_config
    simulation_config = al_simulator.get_simulation_config()

    # Display configuration summary in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Simulation Setup")

    with st.sidebar.expander("Base Configuration", expanded=True):
        st.markdown(
            f"**Optimization Mode:** {base_config.optimization_mode.value}",
        help=f"The goal of this simulation: {base_config.explain_optimization_mode()}")
        st.markdown(f"**Total Sequences:** {len(base_config.simulation_data)}",
                    help="Total number of sequences in the simulation dataset.")

        if base_config.target_lb is not None:
            st.markdown(f"**Target Lower Bound:** {base_config.target_lb}")
        if base_config.target_ub is not None:
            st.markdown(f"**Target Upper Bound:** {base_config.target_ub}")
        if base_config.target_value is not None:
            st.markdown(f"**Target Value:** {base_config.target_value}")
        if base_config.discrete_targets:
            st.markdown(f"**Discrete Targets:** {', '.join(base_config.discrete_targets)}")

    with st.sidebar.expander("Simulation Configuration", expanded=True):
        if simulation_config.n_start is not None:
            st.markdown(f"**Initial Training Sequences:** {simulation_config.n_start}",
                        help="These sequences are chosen at random from the total sequences (using the simulation seed).")
        if simulation_config.start_ids is not None:
            st.markdown(f"**Start IDs Count:** {len(simulation_config.start_ids)}",
                        help="These sequences and their labels are used to start the simulation.")

        st.markdown(f"**Suggestions per Iteration:** {simulation_config.n_suggestions_per_iteration}",
                    help="Number of suggestions selected per iteration of each campaign.")

        conv_cfg = simulation_config.convergence_config
        st.markdown("**Convergence Criteria:**", help="Determines when the simulated campaign should stop.")
        if hasattr(conv_cfg, 'n_hits') and conv_cfg.n_hits is not None:
            st.markdown(f"  - Number of Hits: {conv_cfg.n_hits}",
                        help="Stops the campaign when this number of targets (hits) is found.")
        if hasattr(conv_cfg, 'max_iterations') and conv_cfg.max_iterations is not None:
            st.markdown(f"  - Max Iterations: {conv_cfg.max_iterations}",
                        help="Always stops the campaign after this number of iterations.")
        if hasattr(conv_cfg, 'max_consecutive_failures') and conv_cfg.max_consecutive_failures is not None:
            st.markdown(f"  - Max Consecutive Failures: {conv_cfg.max_consecutive_failures}",
                        help="Stops the campaign after this number of consecutive failures (no hits found in an iteration).")

    subset = ds_groups[selected_dataset]

    main_tabs = st.tabs(["Individual Simulation Result", "Comparison"])

    with main_tabs[0]:
        exp_names = {e.name: e for e in subset}
        selected_name = st.selectbox("Select Experiment", sorted(exp_names.keys()))
        exp = exp_names[selected_name]

        _render_header_from_compressed(exp)

        sub_tabs = st.tabs(["Aggregate", "Per-Simulation Drill-down"])
        with sub_tabs[0]:
            _render_aggregate_from_compressed(exp)
        with sub_tabs[1]:
            sim_labels = {s.label: s for s in exp.single_sims}
            selected_label = st.selectbox("Simulation run", sorted(sim_labels.keys()))
            sim_data = sim_labels[selected_label]
            _render_single_from_compressed(sim_data, exp.summary)

    with main_tabs[1]:
        _render_comparison_from_compressed(subset)

else:
    st.error(f"Compressed data file not found at `{COMPRESSED_DATA_PATH}`. Please run `compress_reports.py` first.")
    st.stop()
