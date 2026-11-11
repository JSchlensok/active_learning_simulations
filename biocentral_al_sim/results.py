from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar, List, Optional

import altair as alt
import numpy as np
from biocentral_api import (
    ActiveLearningEngineeringCampaignConfig,
    ActiveLearningEngineeringSimulationConfig,
    ActiveLearningModelType,
    ActiveLearningOptimizationMode,
    ActiveLearningScreeningCampaignConfig,
    ActiveLearningScreeningSimulationConfig,
    ActiveLearningScreeningSimulationResult,
    SequenceData,
)
from pydantic import BaseModel

from .datasets import ALSimulatorDataset
from .splits import ALSimulatorSplit


class ActiveLearningSingleSimulationResult:
    """One simulated campaign; subclassed per campaign type."""

    # Saved with the result, so loading can pick the subclass and its config types back
    path: ClassVar[str]
    campaign_config_cls: ClassVar[type[BaseModel]]
    simulation_config_cls: ClassVar[type[BaseModel]]

    def __init__(
        self,
        dataset_id: ALSimulatorDataset,
        al_campaign_config: ActiveLearningScreeningCampaignConfig
        | ActiveLearningEngineeringCampaignConfig,
        al_simulation_config: ActiveLearningScreeningSimulationConfig
        | ActiveLearningEngineeringSimulationConfig,
        simulation_result: ActiveLearningScreeningSimulationResult,
        split_id: ALSimulatorSplit = ALSimulatorSplit.FULL_POOL,
    ):
        self.dataset_id = dataset_id
        self.al_campaign_config = al_campaign_config
        self.al_simulation_config = al_simulation_config
        self.simulation_result = simulation_result
        self.split_id = split_id

    def get_total_number_of_suggestions(self):
        return sum([len(it_res.suggestions) for it_res in self.simulation_result.iteration_results])

    def model_dump_json(self):
        """Serialize to JSON string"""
        return json.dumps(
            {
                "dataset_id": self.dataset_id.value,
                "split_id": self.split_id.value,
                "path": self.path,
                "al_campaign_config": json.loads(
                    self.al_campaign_config.model_dump_json()
                ),
                "al_simulation_config": json.loads(
                    self.al_simulation_config.model_dump_json()
                ),
                "simulation_result": json.loads(
                    self.simulation_result.model_dump_json()
                ),
            }
        )

    @classmethod
    def model_validate_json(cls, json_str: str) -> ActiveLearningSingleSimulationResult:
        """Create an instance of the subclass the JSON string was saved from"""
        data = json.loads(json_str)
        # Results written before engineering simulations existed are all screening runs.
        result_cls = SINGLE_SIMULATION_RESULTS[
            data.get("path", ActiveLearningScreeningSingleSimulationResult.path)
        ]
        return result_cls(
            dataset_id=ALSimulatorDataset(data["dataset_id"]),
            # Results written before splits existed have no split_id; they are all full-pool runs.
            split_id=ALSimulatorSplit(
                data.get("split_id", ALSimulatorSplit.FULL_POOL.value)
            ),
            al_campaign_config=result_cls.campaign_config_cls.model_validate_json(
                json.dumps(data["al_campaign_config"])
            ),
            al_simulation_config=result_cls.simulation_config_cls.model_validate_json(
                json.dumps(data["al_simulation_config"])
            ),
            simulation_result=ActiveLearningScreeningSimulationResult.model_validate_json(
                json.dumps(data["simulation_result"])
            ),
        )

    def is_success(self):
        required_n_hits = self.al_simulation_config.stopping_config.n_hits
        if required_n_hits is None:
            return False  # No hit threshold to measure success against
        return sum(map(len, self.simulation_result.iteration_hits or [])) >= required_n_hits

    def n_hits_found(self) -> int:
        return sum(map(len, self.simulation_result.iteration_hits or []))

    def n_measured(self) -> int:
        """Variants the campaign actually had labelled, excluding the starting set.

        Raises:
            ValueError: If the result has rounds but no iteration results to count.
        """
        per_round = self.simulation_result.iteration_results or []
        if self.simulation_result.iteration_hits and not per_round:
            raise ValueError("Result has rounds but no iteration results to count")
        return sum(len(result.suggestions or []) for result in per_round)

    def hit_rate(self) -> Optional[float]:
        """Fraction of measured variants that turned out to be hits.

        Threshold-free, so it stays meaningful when no hit target is configured and the
        campaign spends its whole budget. None when nothing was measured.
        """
        measured = self.n_measured()
        return self.n_hits_found() / measured if measured else None

    @staticmethod
    def _print_stats(result: ActiveLearningScreeningSimulationResult):
        print(f"Simulation campaign stats:")
        print(f"Simulation stop reasons: {result.stop_reasons}")
        print(f"Total number of iterations: {len(result.iteration_results or [])}")
        print(f"Metrics for all masked data points per iteration: {result.iteration_metrics_total}")
        print(f"Metrics for suggested data points per iteration: {result.iteration_metrics_suggestions}")
        print(f"Number of hits over iterations: {list(map(len, result.iteration_hits or []))}")
        filtered_results_suggestions = [next((sugg for sugg in res.results if sugg.entity_id in res.suggestions), None)
                                        for res in result.iteration_results or []]
        if any(sugg is not None for sugg in filtered_results_suggestions):
            print(f"Iteration result for top suggestion: {filtered_results_suggestions}")

    def _compose_layout(self, charts: dict) -> alt.VConcatChart:
        top = alt.hconcat(charts["metric_evolution"], charts["n_hits"])
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
        from . import plots as al_plots

        result = self.simulation_result
        is_discrete = self.al_campaign_config.optimization_mode == ActiveLearningOptimizationMode.DISCRETE
        metric_name = "Accuracy" if is_discrete else "MAE"

        n_rounds_candidates = [
            len(getattr(result, "iteration_results", []) or []),
            len(getattr(result, "iteration_metrics_total", []) or []),
            len(getattr(result, "iteration_metrics_suggestions", []) or []),
            len(getattr(result, "iteration_hits", []) or []),
            len(getattr(result, "iteration_consecutive_failures", []) or []),
        ]
        n_rounds = max(n_rounds_candidates) if any(n_rounds_candidates) else 0

        iteration_metrics_total = [m.mean for m in (result.iteration_metrics_total or [])[:n_rounds]]
        iteration_metrics_suggestions = [m.mean for m in (result.iteration_metrics_suggestions or [])[:n_rounds]]
        iteration_hits = (result.iteration_hits or [])[:n_rounds]
        n_iteration_hits = list(map(len, iteration_hits))
        iteration_consecutive_failures = (result.iteration_consecutive_failures or [])[:n_rounds]
        iteration_results = (result.iteration_results or [])[:n_rounds]

        charts = {
            "metric_evolution": al_plots.build_metric_evolution_chart(
                iteration_metrics_total=list(iteration_metrics_total),
                iteration_metrics_suggestions=list(iteration_metrics_suggestions),
                n_iteration_hits=n_iteration_hits,
                metric_name=metric_name,
            ),
            "n_hits": al_plots.build_n_hits_chart(
                n_iteration_hits=n_iteration_hits,
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


class ActiveLearningScreeningSingleSimulationResult(
    ActiveLearningSingleSimulationResult
):
    path = "screening"
    campaign_config_cls = ActiveLearningScreeningCampaignConfig
    simulation_config_cls = ActiveLearningScreeningSimulationConfig


class ActiveLearningEngineeringSingleSimulationResult(
    ActiveLearningSingleSimulationResult
):
    path = "engineering"
    campaign_config_cls = ActiveLearningEngineeringCampaignConfig
    simulation_config_cls = ActiveLearningEngineeringSimulationConfig


SINGLE_SIMULATION_RESULTS = {
    result_cls.path: result_cls
    for result_cls in (
        ActiveLearningScreeningSingleSimulationResult,
        ActiveLearningEngineeringSingleSimulationResult,
    )
}


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
            assert first_result.al_simulation_config.stopping_config.n_hits == result.al_simulation_config.stopping_config.n_hits, "Simulation configs must be the same"
            assert first_result.dataset_id == result.dataset_id, "Dataset ID must be the same"
            assert first_result.split_id == result.split_id, "Split ID must be the same"
            assert len(first_result.simulation_result.potential_hits) == len(
                result.simulation_result.potential_hits), "Potential hits must be the same"

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

    def dataset_id(self) -> ALSimulatorDataset:
        return self.simulation_results[0].dataset_id

    def split_id(self) -> ALSimulatorSplit:
        return self.simulation_results[0].split_id

    def embedder_name(self) -> str:
        return self.simulation_results[0].al_campaign_config.embedder_name

    def model_type(self) -> ActiveLearningModelType:
        return self.simulation_results[0].al_campaign_config.model_type

    def potential_hits(self) -> List[str]:
        return self.simulation_results[0].simulation_result.potential_hits

    def get_best_simulation(self):
        return max(self.simulation_results,
                   key=lambda ssr: list(map(len, ssr.simulation_result.iteration_hits)))

    def get_worst_simulation(self):
        return min(self.simulation_results,
                   key=lambda ssr: list(map(len, ssr.simulation_result.iteration_hits)))

    def get_aggregated_hits(self):
        return sum([sum(list(map(len, ssr.simulation_result.iteration_hits or []))) for ssr in self.simulation_results])

    def __get_aggreged_unique_hits(self):
        raise NotImplementedError  # TODO Needs to use the simulation dataset

    def get_aggregated_number_of_suggestions(self):
        return sum([ssr.get_total_number_of_suggestions() for ssr in self.simulation_results])

    def _percent_successful(self):
        return sum([1 for ssr in self.simulation_results if ssr.is_success()]) / len(self.simulation_results) * 100

    def mean_hit_rate(self) -> Optional[float]:
        rates = [rate for rate in (ssr.hit_rate() for ssr in self.simulation_results)
                 if rate is not None]
        return sum(rates) / len(rates) if rates else None

    def print_stats(self):
        print(f"Summary over {len(self.simulation_results)} simulations:")
        print(f"Percent successful: {self._percent_successful()}%")
        measured = [ssr.n_measured() for ssr in self.simulation_results]
        hits = [ssr.n_hits_found() for ssr in self.simulation_results]
        rate = self.mean_hit_rate()
        print(f"Mean hits: {sum(hits) / len(hits):.1f} of {sum(measured) / len(measured):.0f} "
              f"measured | mean hit rate: {'n/a' if rate is None else f'{rate:.2%}'}")

    def _compose_layout(self, charts: dict) -> alt.HConcatChart:
        return alt.hconcat(
            charts["performance_summary"],
            charts["mean_cumulative_successes"],
            charts["mean_metric_evolution"],
        ).resolve_scale(color="independent")

    def visualize(self, save_path: Optional[Path] = None) -> Path:
        self.print_stats()
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

    def build_altair_charts(self):
        """Build interactive Altair charts mirroring `_visualize()`.

        Returns a dict with keys: 'performance_summary',
        'mean_cumulative_successes', 'mean_metric_evolution'.
        """
        from . import plots as al_plots

        is_discrete = self.simulation_results[
                          0].al_campaign_config.optimization_mode == ActiveLearningOptimizationMode.DISCRETE
        metric_name = "Accuracy" if is_discrete else "MAE"

        per_sim_n_hits = [
            list(map(len, ssr.simulation_result.iteration_hits or []))
            for ssr in self.simulation_results
        ]
        per_sim_metrics_total = [
            [m.mean for m in ssr.simulation_result.iteration_metrics_total or []]
            for ssr in self.simulation_results
        ]
        per_sim_is_success = [ssr.is_success() for ssr in self.simulation_results]

        n_hits_threshold = self.simulation_results[0].al_simulation_config.stopping_config.n_hits or int(
            np.inf)  # No threshold if it was None
        max_iterations = max((len(m) for m in per_sim_metrics_total), default=0)
        success_count = sum(1 for s in per_sim_is_success if s)

        convergence_iterations, _, _ = al_plots._compute_convergence_stats(
            per_sim_n_hits=per_sim_n_hits,
            n_hits_threshold=n_hits_threshold,
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
                per_sim_n_hits=per_sim_n_hits,
                per_sim_is_success=per_sim_is_success,
                target_threshold=n_hits_threshold,
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
        is_discrete = (
            first.al_campaign_config.optimization_mode
            == ActiveLearningOptimizationMode.DISCRETE
        )
        return {
            "embedder_name": first.al_campaign_config.embedder_name,
            "model_type": getattr(
                first.al_campaign_config.model_type,
                "value",
                first.al_campaign_config.model_type,
            ),
            "optimization_mode": getattr(
                first.al_campaign_config.optimization_mode,
                "value",
                first.al_campaign_config.optimization_mode,
            ),
            "n_simulations": len(self.simulation_results),
            "n_successful": sum(
                1 for ssr in self.simulation_results if ssr.is_success()
            ),
            "n_hits_threshold": first.al_simulation_config.stopping_config.n_hits,
            "is_discrete": is_discrete,
            # Engineering configs have no discrete targets, as they never classify
            "discrete_targets": (
                first.al_campaign_config.discrete_targets if is_discrete else None
            ),
        }

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


class ActiveLearningSimulationComparer:
    pass  # TODO
