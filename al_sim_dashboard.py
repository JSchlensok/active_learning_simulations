"""Streamlit dashboard for active learning simulation results.

Run with:
    uv run streamlit run al_sim_dashboard.py
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st

import al_plots
from al_simulator import (
    ActiveLearningMultipleSimulationResult,
    ActiveLearningSingleSimulationResult,
)


RESULTS_DIR = Path("simulation_v1_results")


st.set_page_config(page_title="AL Simulation Dashboard", layout="wide")


@st.cache_data(show_spinner=False)
def _load_from_path(path_str: str, mtime: float) -> ActiveLearningMultipleSimulationResult:
    return ActiveLearningMultipleSimulationResult.from_json(Path(path_str))


@st.cache_data(show_spinner=False)
def _load_from_bytes(data: bytes, _digest: str) -> ActiveLearningMultipleSimulationResult:
    json_payload = json.loads(data.decode("utf-8"))
    results = [
        ActiveLearningSingleSimulationResult.model_validate_json(json.dumps(res))
        for res in json_payload
    ]
    return ActiveLearningMultipleSimulationResult(results)


def _bytes_digest(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def _discover_disk_files() -> List[Path]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(RESULTS_DIR.glob("*.json"))


def _load_selected(disk_choices: List[str], uploads) -> Dict[str, ActiveLearningMultipleSimulationResult]:
    loaded: Dict[str, ActiveLearningMultipleSimulationResult] = {}
    for fname in disk_choices:
        path = RESULTS_DIR / fname
        try:
            loaded[fname] = _load_from_path(str(path), path.stat().st_mtime)
        except Exception as exc:
            st.sidebar.error(f"Failed to load {fname}: {exc}")
    for upload in uploads or []:
        try:
            raw = upload.read()
            loaded[upload.name] = _load_from_bytes(raw, _bytes_digest(raw))
        except Exception as exc:
            st.sidebar.error(f"Failed to parse {upload.name}: {exc}")
    return loaded


def _render_aggregate(multi: ActiveLearningMultipleSimulationResult) -> None:
    charts = multi.build_altair_charts()
    st.subheader("Overall performance summary")
    st.altair_chart(charts["performance_summary"].interactive(), use_container_width=True)
    st.subheader("Target discovery progress")
    st.altair_chart(charts["mean_cumulative_successes"].interactive(), use_container_width=True)
    st.subheader("Model performance evolution")
    st.altair_chart(charts["mean_metric_evolution"].interactive(), use_container_width=True)


def _render_single(ssr: ActiveLearningSingleSimulationResult) -> None:
    charts = ssr.build_altair_charts()
    col_left, col_right = st.columns(2)
    with col_left:
        st.altair_chart(charts["metric_evolution"].interactive(), use_container_width=True)
        st.altair_chart(charts["consecutive_failures"].interactive(), use_container_width=True)
    with col_right:
        st.altair_chart(charts["target_successes"].interactive(), use_container_width=True)
        if "suggested_labels" in charts:
            st.altair_chart(charts["suggested_labels"].interactive(), use_container_width=True)

    with st.expander("Raw stats"):
        sr = ssr.simulation_result
        st.markdown(f"- **Stop reasons:** {sr.stop_reasons}")
        st.markdown(f"- **Total iterations:** {len(sr.iteration_results)}")
        st.markdown(f"- **iteration_metrics_total:** `{sr.iteration_metrics_total}`")
        st.markdown(f"- **iteration_metrics_suggestions:** `{sr.iteration_metrics_suggestions}`")
        st.markdown(f"- **iteration_target_successes:** `{sr.iteration_target_successes}`")
        st.markdown(f"- **iteration_consecutive_failures:** `{sr.iteration_consecutive_failures}`")


def _render_cross_experiment(loaded: Dict[str, ActiveLearningMultipleSimulationResult]) -> None:
    cum_inputs: Dict[str, List[List[int]]] = {}
    metric_inputs: Dict[str, List[List[float]]] = {}
    targets: Dict[str, int] = {}
    metric_names: set = set()

    for name, multi in loaded.items():
        cum_inputs[name] = [
            list(ssr.simulation_result.iteration_target_successes or [])
            for ssr in multi.simulation_results
        ]
        metric_inputs[name] = [
            [m.mean for m in (ssr.simulation_result.iteration_metrics_total or [])]
            for ssr in multi.simulation_results
        ]
        targets[name] = multi.simulation_results[0].al_simulation_config.convergence_config.target_successes
        summary = multi.summary()
        metric_names.add("Accuracy" if summary["is_discrete"] else "RMSE")

    metric_name = ", ".join(sorted(metric_names)) if metric_names else "Metric"
    if len(metric_names) > 1:
        st.warning(
            f"Selected experiments mix optimization modes ({metric_name}). "
            "Metric overlay still rendered but the y-axis is not directly comparable."
        )

    st.subheader("Cumulative target successes — mean per experiment")
    st.altair_chart(
        al_plots.build_cross_experiment_cumulative_chart(cum_inputs, targets).interactive(),
        use_container_width=True,
    )
    st.subheader(f"Model performance ({metric_name}) — mean per experiment")
    st.altair_chart(
        al_plots.build_cross_experiment_metric_chart(metric_inputs, metric_name).interactive(),
        use_container_width=True,
    )


def _render_header(name: str, multi: ActiveLearningMultipleSimulationResult) -> None:
    summary = multi.summary()
    st.markdown(f"### `{name}`")
    cols = st.columns(6)
    cols[0].metric("Embedder", summary["embedder_name"])
    cols[1].metric("Model", str(summary["model_type"]))
    cols[2].metric("Mode", str(summary["optimization_mode"]))
    cols[3].metric("Sims", summary["n_simulations"])
    cols[4].metric("Successful", f"{summary['n_successful']}/{summary['n_simulations']}")
    cols[5].metric("Target threshold", summary["target_successes_threshold"])
    if summary.get("discrete_targets"):
        st.caption(f"Discrete targets: {', '.join(summary['discrete_targets'])}")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Load simulation results")

disk_files = _discover_disk_files()
disk_options = [p.name for p in disk_files]
if disk_options:
    disk_choices = st.sidebar.multiselect(
        f"From `{RESULTS_DIR}/`",
        options=disk_options,
        default=disk_options[:1],
    )
else:
    disk_choices = []
    st.sidebar.caption(f"No files found in `{RESULTS_DIR}/`.")

uploads = st.sidebar.file_uploader(
    "Or upload JSON files",
    type="json",
    accept_multiple_files=True,
)

loaded = _load_selected(disk_choices, uploads)

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("Active Learning Simulation Dashboard")

if not loaded:
    st.info("Select or upload a result file from the sidebar to begin.")
    st.stop()

experiment_name = st.selectbox("Experiment", list(loaded.keys()))
active_multi = loaded[experiment_name]

_render_header(experiment_name, active_multi)

tab_labels = ["Aggregate", "Per-Simulation Drill-down"]
if len(loaded) > 1:
    tab_labels.append("Cross-Experiment")

tabs = st.tabs(tab_labels)

with tabs[0]:
    _render_aggregate(active_multi)

with tabs[1]:
    options: List[Tuple[str, ActiveLearningSingleSimulationResult]] = [
        (ssr.label(), ssr) for ssr in active_multi.simulation_results
    ]
    chosen_label = st.selectbox(
        "Simulation run",
        options=[label for label, _ in options],
    )
    chosen_ssr = next(ssr for label, ssr in options if label == chosen_label)
    _render_single(chosen_ssr)

if len(loaded) > 1:
    with tabs[2]:
        _render_cross_experiment(loaded)
