from __future__ import annotations

import json

import altair as alt

from pathlib import Path
from typing import List, Optional
from biocentral_api import SequenceData, ActiveLearningCampaignConfig, ActiveLearningSimulationConfig, \
    ActiveLearningOptimizationMode, ActiveLearningModelType, BiocentralAPI, ActiveLearningIterationResult, \
    ActiveLearningSimulationResult, ActiveLearningConvergenceConfig

from pydantic import BaseModel


class ActiveLearningFixedBaseConfig(BaseModel):
    class Config:
        frozen = False

    """ Fixed base config for active learning simulations that does not change throughout simulation"""
    # Simulation config
    simulation_data: List[SequenceData]

    # Campaign config
    optimization_mode: ActiveLearningOptimizationMode
    target_lb: Optional[float] = None
    target_ub: Optional[float] = None
    target_value: Optional[float] = None
    discrete_targets: Optional[List[str]] = None


class ActiveLearningSimulator:
    def __init__(self, al_base_config: ActiveLearningFixedBaseConfig):
        self.base_config = al_base_config
        self.biocentral_api = BiocentralAPI(local_only=True)

    def _run_simulation(self, model_type: ActiveLearningModelType, embedder_name: str,
                        seed: int) -> ActiveLearningSingleSimulationResult:
        al_campaign_config = ActiveLearningCampaignConfig(name="Test",  # TODO
                                                          embedder_name=embedder_name,
                                                          model_type=model_type,
                                                          optimization_mode=self.base_config.optimization_mode,
                                                          seed=seed,
                                                          target_lb=self.base_config.target_lb,
                                                          target_ub=self.base_config.target_ub,
                                                          target_value=self.base_config.target_value,
                                                          discrete_targets=self.base_config.discrete_targets)
        al_simulation_config = ActiveLearningSimulationConfig(simulation_data=self.base_config.simulation_data,
                                                              n_start=10,  # TODO
                                                              n_suggestions_per_iteration=5,  # TODO
                                                              convergence_config=ActiveLearningConvergenceConfig(
                                                                  max_labels_budget=50,
                                                                  target_successes=10,
                                                                  max_consecutive_failures=5
                                                              ),  # TODO
                                                              )
        result = self.biocentral_api.al_simulation(campaign_config=al_campaign_config,
                                                   simulation_config=al_simulation_config).run_with_progress()
        return ActiveLearningSingleSimulationResult(al_campaign_config=al_campaign_config,
                                                    al_simulation_config=al_simulation_config,
                                                    simulation_result=result)

    def simulate(self, embedder_name: str, model_type: ActiveLearningModelType,
                 n_rounds: int) -> ActiveLearningMultipleSimulationResult:
        simulation_results = []
        for iteration_idx in range(n_rounds):
            print(f"Running simulation round {iteration_idx + 1}/{n_rounds}...")
            seed = 42 + iteration_idx
            single_simulation_result = self._run_simulation(model_type=model_type,
                                                            embedder_name=embedder_name,
                                                            seed=seed)
            simulation_results.append(single_simulation_result)
        return ActiveLearningMultipleSimulationResult(simulation_results)


class ActiveLearningSingleSimulationResult:
    def __init__(self, al_campaign_config: ActiveLearningCampaignConfig,
                 al_simulation_config: ActiveLearningSimulationConfig,
                 simulation_result: ActiveLearningSimulationResult):
        self.al_campaign_config = al_campaign_config
        self.al_simulation_config = al_simulation_config
        self.simulation_result = simulation_result

    def model_dump_json(self):
        """Serialize to JSON string"""
        return json.dumps({
            'al_campaign_config': json.loads(self.al_campaign_config.model_dump_json()),
            'al_simulation_config': json.loads(self.al_simulation_config.model_dump_json()),
            'simulation_result': json.loads(self.simulation_result.model_dump_json())
        })

    @classmethod
    def model_validate_json(cls, json_str: str):
        """Create instance from JSON string"""
        data = json.loads(json_str)
        return cls(
            al_campaign_config=ActiveLearningCampaignConfig.model_validate_json(json.dumps(data['al_campaign_config'])),
            al_simulation_config=ActiveLearningSimulationConfig.model_validate_json(
                json.dumps(data['al_simulation_config'])),
            simulation_result=ActiveLearningSimulationResult.model_validate_json(json.dumps(data['simulation_result']))
        )

    def is_success(self):
        required_target_successes = 10  # TODO CONFIG
        return sum(self.simulation_result.iteration_target_successes) >= required_target_successes

    @staticmethod
    def _print_stats(result: ActiveLearningSimulationResult):
        print(f"Simulation campaign stats:")
        print(f"Simulation stop reasons: {result.stop_reasons}")
        print(f"Total number of iterations: {len(result.iteration_results or [])}")
        print(f"Metrics for all masked data points per iteration: {result.iteration_metrics_total}")
        print(f"Metrics for suggested data points per iteration: {result.iteration_metrics_suggestions}")
        print(f"Target successes over iterations: {result.iteration_target_successes}")
        filtered_results_suggestions = [[sugg for sugg in res.results if sugg.entity_id in res.suggestions][0]
                                        for res in result.iteration_results or []]
        print(f"Iteration result for top suggestion: {filtered_results_suggestions}")

    def _compose_layout(self, charts: dict) -> alt.VConcatChart:
        top = alt.hconcat(charts["metric_evolution"], charts["target_successes"])
        bottom_charts = [charts["consecutive_failures"]]
        if "suggested_labels" in charts:
            bottom_charts.append(charts["suggested_labels"])
        bottom = alt.hconcat(*bottom_charts)
        return alt.vconcat(top, bottom).resolve_scale(color="independent")

    def visualize(self, save_path: Optional[Path] = None) -> Path:
        self._print_stats(self.simulation_result)
        charts = self.build_altair_charts()
        layout = self._compose_layout(charts)

        if save_path is None:
            save_path = Path(f"{self.simulation_result.campaign_name}_single.png")
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        layout.save(str(save_path), ppi=144)
        print(f"Saved single-simulation plot to {save_path}")
        return save_path

    def label(self) -> str:
        seed = self.al_campaign_config.seed
        model_type = getattr(self.al_campaign_config.model_type, "value",
                             self.al_campaign_config.model_type)
        return f"seed={seed} | {model_type} | success={self.is_success()}"

    def build_altair_charts(self):
        """Build interactive Altair charts mirroring `_visualize_results()`.

        Returns a dict keyed by chart name. The 'suggested_labels' key is
        only present for DISCRETE optimization mode.
        """
        import al_plots

        result = self.simulation_result
        is_discrete = self.al_campaign_config.optimization_mode == ActiveLearningOptimizationMode.DISCRETE
        metric_name = "Accuracy" if is_discrete else "RMSE"

        n_rounds_candidates = [
            len(getattr(result, "iteration_results", []) or []),
            len(getattr(result, "iteration_metrics_total", []) or []),
            len(getattr(result, "iteration_metrics_suggestions", []) or []),
            len(getattr(result, "iteration_target_successes", []) or []),
            len(getattr(result, "iteration_consecutive_failures", []) or []),
        ]
        n_rounds = max(n_rounds_candidates) if any(n_rounds_candidates) else 0

        iteration_metrics_total = [m.mean for m in (result.iteration_metrics_total or [])[:n_rounds]]
        iteration_metrics_suggestions = [m.mean for m in (result.iteration_metrics_suggestions or [])[:n_rounds]]
        iteration_target_successes = (result.iteration_target_successes or [])[:n_rounds]
        iteration_consecutive_failures = (result.iteration_consecutive_failures or [])[:n_rounds]
        iteration_results = (result.iteration_results or [])[:n_rounds]

        charts = {
            "metric_evolution": al_plots.build_metric_evolution_chart(
                iteration_metrics_total=list(iteration_metrics_total),
                iteration_metrics_suggestions=list(iteration_metrics_suggestions),
                iteration_target_successes=list(iteration_target_successes),
                metric_name=metric_name,
            ),
            "target_successes": al_plots.build_target_successes_chart(
                iteration_target_successes=list(iteration_target_successes),
            ),
            "consecutive_failures": al_plots.build_consecutive_failures_chart(
                iteration_consecutive_failures=list(iteration_consecutive_failures),
                stop_reasons=list(result.stop_reasons or []),
            ),
        }

        if is_discrete:
            id2label = {dp.seq_id: dp.label for dp in self.al_simulation_config.simulation_data}
            iteration_suggestions = [list(ir.suggestions) for ir in iteration_results]
            unique_labels = sorted({dp.label for dp in self.al_simulation_config.simulation_data})
            charts["suggested_labels"] = al_plots.build_suggested_labels_chart(
                iteration_suggestions=iteration_suggestions,
                id2label=id2label,
                unique_labels=unique_labels,
                optimization_targets=self.al_campaign_config.discrete_targets,
            )

        return charts


class ActiveLearningMultipleSimulationResult:
    def __init__(self, simulation_results: List[ActiveLearningSingleSimulationResult]):
        self.simulation_results = simulation_results
        self._validate()

    def _validate(self):
        results = self.simulation_results
        first_result = results[0]
        for result in results:
            assert first_result.al_campaign_config.embedder_name == result.al_campaign_config.embedder_name, "Embedder config must be the same"
            assert first_result.al_campaign_config.optimization_mode == result.al_campaign_config.optimization_mode, "Optimization mode must be the same"
            assert first_result.al_simulation_config.convergence_config.target_successes == result.al_simulation_config.convergence_config.target_successes, "Simulation configs must be the same"

    @classmethod
    def from_json(cls, path: Path) -> ActiveLearningMultipleSimulationResult:
        """Load simulation results from JSON file"""
        with open(path, 'r') as f:
            json_results = json.load(f)
            # Convert dict to JSON string first, then validate
            results = [
                ActiveLearningSingleSimulationResult.model_validate_json(json.dumps(res))
                for res in json_results
            ]
            return cls(results)

    def get_best_simulation(self):
        return max(self.simulation_results, key=lambda ssr: ssr.simulation_result.iteration_target_successes)

    def get_worst_simulation(self):
        return min(self.simulation_results, key=lambda ssr: ssr.simulation_result.iteration_target_successes)

    def _percent_successful(self):
        return sum([1 for ssr in self.simulation_results if ssr.is_success()]) / len(self.simulation_results) * 100

    def _print_stats(self):
        print(f"Summary over {len(self.simulation_results)} simulations:")
        print(f"Percent successful: {self._percent_successful()}%")

    def _compose_layout(self, charts: dict) -> alt.HConcatChart:
        return alt.hconcat(
            charts["performance_summary"],
            charts["mean_cumulative_successes"],
            charts["mean_metric_evolution"],
        ).resolve_scale(color="independent")

    def visualize(self, save_path: Optional[Path] = None) -> Path:
        self._print_stats()
        print("Visualizing aggregated results across multiple simulations...")
        charts = self.build_altair_charts()
        layout = self._compose_layout(charts)

        if save_path is None:
            save_path = Path("multi_sim.png")
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        layout.save(str(save_path), ppi=144)
        print(f"Saved multi-simulation plot to {save_path}")

        stem = save_path.stem
        worst_path = save_path.with_name(f"{stem}_worst.png")
        best_path = save_path.with_name(f"{stem}_best.png")

        print("Visualizing worst simulation...")
        self.get_worst_simulation().visualize(save_path=worst_path)
        print("Visualizing best simulation...")
        self.get_best_simulation().visualize(save_path=best_path)
        return save_path

    def save(self, path: Path):
        """Save simulation results to JSON file"""
        with open(path, 'w') as f:
            json_results = []
            for result in self.simulation_results:
                # Parse the result into dict structure
                result_dict = json.loads(result.model_dump_json())

                # Remove simulation_data from al_simulation_config to reduce file size
                if 'al_simulation_config' in result_dict and 'simulation_data' in result_dict['al_simulation_config']:
                    result_dict['al_simulation_config']['simulation_data'] = [
                        SequenceData(seq_id="Dummy1", seq="MDUMMY").model_dump(),
                        SequenceData(seq_id="Dummy2", seq="ADUMMY").model_dump(),
                        SequenceData(seq_id="Dummy3", seq="GDUMMY").model_dump()
                    ]

                json_results.append(result_dict)

            # Write the JSON with proper formatting
            json.dump(json_results, f, indent=4)

    def build_altair_charts(self):
        """Build interactive Altair charts mirroring `_visualize()`.

        Returns a dict with keys: 'performance_summary',
        'mean_cumulative_successes', 'mean_metric_evolution'.
        """
        import al_plots

        is_discrete = self.simulation_results[
                          0].al_campaign_config.optimization_mode == ActiveLearningOptimizationMode.DISCRETE
        metric_name = "Accuracy" if is_discrete else "RMSE"

        per_sim_target_successes = [
            list(ssr.simulation_result.iteration_target_successes or [])
            for ssr in self.simulation_results
        ]
        per_sim_metrics_total = [
            [m.mean for m in ssr.simulation_result.iteration_metrics_total or []]
            for ssr in self.simulation_results
        ]
        per_sim_is_success = [ssr.is_success() for ssr in self.simulation_results]

        target_threshold = self.simulation_results[0].al_simulation_config.convergence_config.target_successes
        max_iterations = max((len(m) for m in per_sim_metrics_total), default=0)
        success_count = sum(1 for s in per_sim_is_success if s)

        convergence_iterations, _, _ = al_plots._compute_convergence_stats(
            per_sim_target_successes=per_sim_target_successes,
            target_successes_threshold=target_threshold,
        )

        best = self.get_best_simulation()
        worst = self.get_worst_simulation()

        return {
            "performance_summary": al_plots.build_performance_summary_chart(
                success_count=success_count,
                total_runs=len(self.simulation_results),
                convergence_iterations=convergence_iterations,
                max_iterations=max_iterations,
            ),
            "mean_cumulative_successes": al_plots.build_mean_cumulative_successes_chart(
                per_sim_target_successes=per_sim_target_successes,
                per_sim_is_success=per_sim_is_success,
                target_threshold=target_threshold,
            ),
            "mean_metric_evolution": al_plots.build_mean_metric_evolution_chart(
                per_sim_metrics_total=per_sim_metrics_total,
                best_run_metrics=[m.mean for m in best.simulation_result.iteration_metrics_total or []],
                worst_run_metrics=[m.mean for m in worst.simulation_result.iteration_metrics_total or []],
                metric_name=metric_name,
            ),
        }

    def summary(self) -> dict:
        """Return a small dict of summary values for the dashboard header."""
        first = self.simulation_results[0]
        return {
            "embedder_name": first.al_campaign_config.embedder_name,
            "model_type": getattr(first.al_campaign_config.model_type, "value",
                                  first.al_campaign_config.model_type),
            "optimization_mode": getattr(first.al_campaign_config.optimization_mode, "value",
                                         first.al_campaign_config.optimization_mode),
            "n_simulations": len(self.simulation_results),
            "n_successful": sum(1 for ssr in self.simulation_results if ssr.is_success()),
            "target_successes_threshold": first.al_simulation_config.convergence_config.target_successes,
            "is_discrete": first.al_campaign_config.optimization_mode == ActiveLearningOptimizationMode.DISCRETE,
            "discrete_targets": first.al_campaign_config.discrete_targets,
        }


class ActiveLearningSimulationComparer:
    pass  # TODO
