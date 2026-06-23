import re

from pathlib import Path
from al_simulator import (
    ActiveLearningMultipleSimulationResult,
    DashboardCompressedData,
    DashboardExperimentData, DashboardSingleSimulationData,
)

RESULTS_DIR = Path("simulation_v1_results")


def parse_dataset_from_filename(fname: str) -> str:
    # al_sim_{DATASET}_{EMBEDDER_NAME}_{MODEL_NAME}.json
    match = re.match(r"al_sim_(.+?)_(.+?)_(.+)\.json", fname)
    if match:
        return match.group(1)
    return "Unknown"


def compress_reports(run_name: str):
    if not RESULTS_DIR.exists():
        print(f"Directory {RESULTS_DIR} not found.")
        return

    json_files = sorted(RESULTS_DIR.glob("*.json"))
    experiments = []

    for path in json_files:
        if "compressed_dashboard" in path.name:
            continue

        print(f"Processing {path.name}...")

        multi = ActiveLearningMultipleSimulationResult.from_json(path)
        dataset_name = multi.dataset_id()
        embedder_name = multi.embedder_name()
        model_name = multi.model_type().value
        potential_hits = multi.potential_hits()

        # Prepare data for DashboardExperimentData
        summary = multi.summary()

        per_sim_n_hits = [
            list(map(len, ssr.simulation_result.iteration_hits or []))
            for ssr in multi.simulation_results
        ]
        per_sim_metrics_total = [
            [m.mean for m in ssr.simulation_result.iteration_metrics_total or []]
            for ssr in multi.simulation_results
        ]
        per_sim_metrics_suggestions = [
            [m.mean for m in ssr.simulation_result.iteration_metrics_suggestions or []]
            for ssr in multi.simulation_results
        ]
        per_sim_is_success = [ssr.is_success() for ssr in multi.simulation_results]

        # Prepare single_sims for drill-down
        single_sims = []
        for ssr in multi.simulation_results:
            # We only need enough to render the single view
            # Most of it is in ssr.simulation_result and ssr.al_campaign_config
            # We can store a stripped down version
            sim_data = DashboardSingleSimulationData(
                label=ssr.label(),
                is_success=ssr.is_success(),
                seed=ssr.al_campaign_config.seed,
                stop_reasons=ssr.simulation_result.stop_reasons or ["None"],
                iteration_metrics_total=[m.mean for m in ssr.simulation_result.iteration_metrics_total or []],
                iteration_metrics_suggestions=[m.mean for m in
                                               ssr.simulation_result.iteration_metrics_suggestions or []],
                iteration_hits=list(ssr.simulation_result.iteration_hits or []),
                iteration_consecutive_failures=list(ssr.simulation_result.iteration_consecutive_failures or []),
                iteration_results_count=len(ssr.simulation_result.iteration_results or []),
                n_hits_threshold=ssr.al_simulation_config.convergence_config.n_hits,
            )
            single_sims.append(sim_data)

        exp_data = DashboardExperimentData(
            name=path.name,
            dataset_id=dataset_name,
            embedder=embedder_name,
            model=model_name,
            summary=summary,
            aggregated_hits=multi.get_aggregated_hits(),
            aggregated_suggestions=multi.get_aggregated_number_of_suggestions(),
            potential_hits=potential_hits,
            per_sim_n_hits=per_sim_n_hits,
            per_sim_metrics_total=per_sim_metrics_total,
            per_sim_metrics_suggestions=per_sim_metrics_suggestions,
            per_sim_is_success=per_sim_is_success,
            single_sims=single_sims
        )
        experiments.append(exp_data)

    compressed = DashboardCompressedData(run_name=run_name, experiments=experiments)
    OUTPUT_FILE = RESULTS_DIR / f"compressed_dashboard_data_{run_name}.json"

    with open(OUTPUT_FILE, "w") as f:
        f.write(compressed.model_dump_json(indent=4))

    print(f"Successfully created {OUTPUT_FILE} with {len(experiments)} experiments.")
