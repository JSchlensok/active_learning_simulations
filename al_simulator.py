from __future__ import annotations

import hashlib
import json
import numpy as np
import altair as alt

from functools import cache
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field
from biotrainer_core.input_files import read_FASTA

from al_simulation_container import ALSimulatorDataset
from al_splits import ALSimulatorSplit

from biocentral_api import SequenceData, ActiveLearningScreeningCampaignConfig, ActiveLearningScreeningSimulationConfig, \
    ActiveLearningOptimizationMode, ActiveLearningModelType, BiocentralAPI, \
    ActiveLearningScreeningSimulationResult, ActiveLearningStoppingConfig


class DashboardSingleSimulationData(BaseModel):
    label: str = Field(..., description="Label identifying the simulation with seed, model type, and success status")
    is_success: bool = Field(...,
                             description="Whether the simulation met the success criteria (e.g. reached target hits)")
    seed: int = Field(..., description="Random seed used for this simulation run")
    stop_reasons: List[str] = Field(...,
                                    description="Reasons why the simulation stopped (e.g., budget exhausted, convergence reached)")
    iteration_metrics_total: List[float] = Field(...,
                                                 description="Mean metric values for all masked data points per iteration")
    iteration_metrics_suggestions: List[float] = Field(...,
                                                       description="Mean metric values for suggested data points per iteration")
    iteration_hits: List[List[str]] = Field(..., description="Hits found in each iteration")
    iteration_consecutive_failures: List[int] = Field(...,
                                                      description="Count of consecutive failures in each iteration")
    iteration_results_count: int = Field(..., description="Total number of iteration results in the simulation")
    n_hits_threshold: int = Field(..., description="Target number of successful hits required for convergence")


class DashboardExperimentData(BaseModel):
    name: str
    dataset_id: ALSimulatorDataset
    embedder: str
    model: str
    # Optional so dashboard files written before splits existed still validate
    split_id: Optional[ALSimulatorSplit] = None
    summary: dict
    aggregated_hits: int
    aggregated_suggestions: int
    potential_hits: List[str]
    per_sim_n_hits: List[List[int]]
    per_sim_metrics_total: List[List[float]]
    per_sim_metrics_suggestions: List[List[float]]
    per_sim_is_success: List[bool]
    # For drill-down
    single_sims: List[DashboardSingleSimulationData]


class DashboardCompressedData(BaseModel):
    run_name: str
    experiments: List[DashboardExperimentData]


class CampaignSettings(BaseModel):
    """Defines how a simulated campaign acquires labelled variants.

    Models a lab budget: `n_iterations` experimental rounds of
    `n_suggestions_per_iteration` variants each.
    """

    model_config = ConfigDict(frozen=True)

    short_name: str = Field(description="Tag identifying the regime, used in file names")
    n_start: int = Field(description="Starting set size, including wildtype / parent")
    n_suggestions_per_iteration: int = Field(description="Variants measured per round")
    n_iterations: int = Field(default=8, description="Rounds a campaign is allowed")
    n_replicates: int = Field(default=5, description="Seeded repeats per experiment")
    first_seed: int = Field(default=42, description="Seed of the first replicate")
    redraw_start_set: bool = Field(
        default=True,
        description="Draw a fresh starting set per replicate. False shares one across all "
                    "replicates, so they vary only in model stochasticity.")
    # Stopping criteria. `n_iterations` always applies; these two are opt-in, so by default
    # a campaign spends its whole budget and the hit rate covers the full 8 rounds. The
    # server requires at least one of budget / hits / failed-rounds, which the derived
    # `max_labels_budget` always satisfies.
    stop_at_n_hits: Optional[int] = Field(
        default=None,
        description="Stop once this many hits are found. Spends full budget if set to None.")
    stop_after_failed_rounds: Optional[int] = Field(
        default=None,
        description="Stop after this many consecutive rounds without a hit.")

    @property
    def max_labels_budget(self) -> int:
        """Total variants a campaign may measure. Derived, so it cannot contradict the rounds."""
        return self.n_iterations * self.n_suggestions_per_iteration

    def fingerprint(self) -> str:
        digest = hashlib.sha256(self.model_dump_json().encode("utf-8"))
        return digest.hexdigest()[:8]


LOW_THROUGHPUT = CampaignSettings(short_name="low", n_start=10, n_suggestions_per_iteration=5)
HIGH_THROUGHPUT = CampaignSettings(short_name="high", n_start=96, n_suggestions_per_iteration=48)
THROUGHPUTS = {settings.short_name: settings for settings in (LOW_THROUGHPUT, HIGH_THROUGHPUT)}

# Default for callers that do not choose a regime
CAMPAIGN_SETTINGS = LOW_THROUGHPUT


class ActiveLearningFixedBaseConfig(BaseModel):
    class Config:
        frozen = False

    """ Fixed base config for active learning simulations that does not change throughout simulation"""
    # dataset_id
    dataset_id: ALSimulatorDataset

    # Simulation config
    simulation_data: List[SequenceData]

    # Split config (see al_splits). FULL_POOL is the identity split and changes nothing.
    split_id: ALSimulatorSplit = ALSimulatorSplit.FULL_POOL
    start_ids: Optional[List[str]] = None
    settings: CampaignSettings = CAMPAIGN_SETTINGS

    # Campaign config
    optimization_mode: ActiveLearningOptimizationMode
    target_lb: Optional[float] = None
    target_ub: Optional[float] = None
    target_value: Optional[float] = None
    discrete_targets: Optional[List[str]] = None

    def explain_optimization_mode(self) -> str:
        match self.optimization_mode:
            case ActiveLearningOptimizationMode.MAXIMIZE:
                return "Maximize the target value."
            case ActiveLearningOptimizationMode.MINIMIZE:
                return "Minimize the target value."
            case ActiveLearningOptimizationMode.DISCRETE:
                return "Classify the sequences into discrete classes."
            case ActiveLearningOptimizationMode.INTERVAL:
                return "Find target sequences with labels in the specified interval."
            case ActiveLearningOptimizationMode.VALUE:
                return "Find targets with the specified value."
            case _:
                return "Unknown optimization mode."


def get_simulator(dataset_id: ALSimulatorDataset,
                  split_id: ALSimulatorSplit = ALSimulatorSplit.FULL_POOL) -> ActiveLearningSimulator:
    simulation_data = read_FASTA(dataset_id.to_path())
    assert len(simulation_data) > 0, f"Simulation data for {dataset_id} is empty."

    simulation_data, start_ids = _apply_split(dataset_id=dataset_id, split_id=split_id,
                                              simulation_data=simulation_data)

    definition = dataset_id.definition()
    base_config = ActiveLearningFixedBaseConfig(
        dataset_id=dataset_id,
        simulation_data=simulation_data,
        split_id=split_id,
        start_ids=start_ids,
        optimization_mode=definition.optimization_mode,
        target_lb=definition.target_lb,
        target_ub=definition.target_ub,
        target_value=definition.target_value,
        discrete_targets=definition.discrete_targets)
    return ActiveLearningSimulator(al_base_config=base_config)


def _apply_split(dataset_id: ALSimulatorDataset, split_id: ALSimulatorSplit,
                 simulation_data: List[SequenceData]) -> tuple[List[SequenceData], Optional[List[str]]]:
    """Resolve a split into (pool, start_ids).

    The split's *train* half becomes the campaign's starting set (``start_ids``); its *test* half
    stays in the pool  unlabelled  for the campaign to discover
    """
    if split_id.is_identity():
        return simulation_data, None

    pool, assignment = split_id.resolve(simulation_data,
                                        explicit_reference=dataset_id.reference_sequence())
    if assignment is None:
        print(f"Split {split_id.name}: pool {len(simulation_data)} -> {len(pool)} sequences.")
        return pool, None

    print(f"Split {split_id.name} [{split_id.definition().axis.value}] on {len(pool)} sequences: "
          f"{assignment.summary()} ({assignment.description}).")
    return pool, assignment.train_ids



STORE_PREDICTIONS = False

TASK_TIMINGS: List[tuple] = []


def profile_summary(slowest: int = 5) -> str:
    """Where client-side wall time went across every campaign run so far."""
    if not TASK_TIMINGS:
        return "No campaigns timed."
    totals = {bucket: sum(getattr(timing, bucket) for _, timing in TASK_TIMINGS)
              for bucket in ("total", "http", "sleep", "handler")}
    polls = sum(timing.polls for _, timing in TASK_TIMINGS)
    dtos = sum(timing.dtos for _, timing in TASK_TIMINGS)
    # Whatever is left is server compute the client polled through.
    waited = totals["total"] - totals["http"] - totals["sleep"] - totals["handler"]

    lines = [f"{len(TASK_TIMINGS)} campaigns, {totals['total']:.0f}s of client wall time "
             f"({polls} polls, {dtos} dtos)"]
    for label, value in (("http (+deserialise)", totals["http"]),
                         ("sleep (poll interval)", totals["sleep"]),
                         ("handler", totals["handler"]),
                         ("waited on server", waited)):
        share = value / totals["total"] * 100 if totals["total"] else 0
        lines.append(f"  {label:<22} {value:8.0f}s  {share:5.1f}%")
    lines.append(f"  slowest {slowest} campaigns:")
    for label, timing in sorted(TASK_TIMINGS, key=lambda kv: -kv[1].total)[:slowest]:
        lines.append(f"    {timing.total:7.1f}s  {label}")
    return "\n".join(lines)


@cache
def biocentral_api() -> BiocentralAPI:
    """The local biocentral server, health-checked once per process."""
    return BiocentralAPI(local_only=True).wait_until_healthy()


class ActiveLearningSimulator:
    def __init__(self, al_base_config: ActiveLearningFixedBaseConfig):
        self.base_config = al_base_config

    def get_simulation_config(self):
        # start_ids and n_start are mutually exclusive: a pinned starting set or a random draw.
        start_ids = self.base_config.start_ids
        settings = self.base_config.settings
        return ActiveLearningScreeningSimulationConfig(
            simulation_data=self.base_config.simulation_data,
            n_start=None if start_ids else settings.n_start,
            start_ids=start_ids,
            n_suggestions_per_iteration=settings.n_suggestions_per_iteration,
            stopping_config=ActiveLearningStoppingConfig(
                n_max_iterations=settings.n_iterations,
                max_labels_budget=settings.max_labels_budget,
                n_hits=settings.stop_at_n_hits,
                max_consecutive_failures=settings.stop_after_failed_rounds,
            ),
        )

    def _run_simulation(self, model_type: ActiveLearningModelType, embedder_name: str,
                        seed: int,
                        show_progress: bool = True) -> ActiveLearningSingleSimulationResult:
        al_campaign_config = ActiveLearningScreeningCampaignConfig(name="Test",  # TODO
                                                                   embedder_name=embedder_name,
                                                                   model_type=model_type,
                                                                   optimization_mode=self.base_config.optimization_mode,
                                                                   seed=seed,
                                                                   target_lb=self.base_config.target_lb,
                                                                   target_ub=self.base_config.target_ub,
                                                                   target_value=self.base_config.target_value,
                                                                   discrete_targets=self.base_config.discrete_targets)
        al_simulation_config = self.get_simulation_config()
        task = biocentral_api().al_screening_simulation(campaign_config=al_campaign_config,
                                                        simulation_config=al_simulation_config,
                                                        store_predictions=STORE_PREDICTIONS)
        # Concurrent callers pass show_progress=False: several tqdm bars writing to one
        # terminal interleave into noise.
        result = task.run_with_progress() if show_progress else task.run()
        TASK_TIMINGS.append((
            f"{self.base_config.dataset_id.name}/{embedder_name}/{model_type.value}"
            f"/{self.base_config.settings.short_name}/seed{seed}",
            task.timing,
        ))
        if result is None:
            raise RuntimeError("Simulation failed")

        return ActiveLearningSingleSimulationResult(
            dataset_id=self.base_config.dataset_id,
            split_id=self.base_config.split_id,
            al_campaign_config=al_campaign_config,
            al_simulation_config=al_simulation_config,
            simulation_result=result)

    def simulate(self, embedder_name: str, model_type: ActiveLearningModelType,
                 n_rounds: int,
                 show_progress: bool = True) -> ActiveLearningMultipleSimulationResult:
        simulation_results = []
        for iteration_idx in range(n_rounds):
            if show_progress:
                print(f"Running simulation round {iteration_idx + 1}/{n_rounds}...")
            seed = self.base_config.settings.first_seed + iteration_idx
            single_simulation_result = self._run_simulation(model_type=model_type,
                                                            embedder_name=embedder_name,
                                                            seed=seed,
                                                            show_progress=show_progress)
            simulation_results.append(single_simulation_result)
        return ActiveLearningMultipleSimulationResult(simulation_results)


class ActiveLearningSingleSimulationResult:
    def __init__(self,
                 dataset_id: ALSimulatorDataset,
                 al_campaign_config: ActiveLearningScreeningCampaignConfig,
                 al_simulation_config: ActiveLearningScreeningSimulationConfig,
                 simulation_result: ActiveLearningScreeningSimulationResult,
                 split_id: ALSimulatorSplit = ALSimulatorSplit.FULL_POOL):
        self.dataset_id = dataset_id
        self.al_campaign_config = al_campaign_config
        self.al_simulation_config = al_simulation_config
        self.simulation_result = simulation_result
        self.split_id = split_id

    def get_total_number_of_suggestions(self):
        return sum([len(it_res.suggestions) for it_res in self.simulation_result.iteration_results])

    def model_dump_json(self):
        """Serialize to JSON string"""
        return json.dumps({
            'dataset_id': self.dataset_id.value,
            'split_id': self.split_id.value,
            'al_campaign_config': json.loads(self.al_campaign_config.model_dump_json()),
            'al_simulation_config': json.loads(self.al_simulation_config.model_dump_json()),
            'simulation_result': json.loads(self.simulation_result.model_dump_json())
        })

    @classmethod
    def model_validate_json(cls, json_str: str):
        """Create instance from JSON string"""
        data = json.loads(json_str)
        return cls(
            dataset_id=ALSimulatorDataset(data['dataset_id']),
            # Results written before splits existed have no split_id; they are all full-pool runs.
            split_id=ALSimulatorSplit(data.get('split_id', ALSimulatorSplit.FULL_POOL.value)),
            al_campaign_config=ActiveLearningScreeningCampaignConfig.model_validate_json(
                json.dumps(data['al_campaign_config'])),
            al_simulation_config=ActiveLearningScreeningSimulationConfig.model_validate_json(
                json.dumps(data['al_simulation_config'])),
            simulation_result=ActiveLearningScreeningSimulationResult.model_validate_json(
                json.dumps(data['simulation_result']))
        )

    def is_success(self):
        required_n_hits = self.al_simulation_config.stopping_config.n_hits
        if required_n_hits is None:
            return False  # No hit threshold to measure success against
        return sum(map(len, self.simulation_result.iteration_hits or [])) >= required_n_hits

    def n_hits_found(self) -> int:
        return sum(map(len, self.simulation_result.iteration_hits or []))

    def n_measured(self) -> int:
        """Variants the campaign actually had labelled, excluding the starting set."""
        return sum(len(result.suggestions or [])
                   for result in (self.simulation_result.iteration_results or []))

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
        filtered_results_suggestions = [[sugg for sugg in res.results if sugg.entity_id in res.suggestions][0]
                                        for res in result.iteration_results or []]
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
        import al_plots

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
        import al_plots

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
        return {
            "embedder_name": first.al_campaign_config.embedder_name,
            "model_type": getattr(first.al_campaign_config.model_type, "value",
                                  first.al_campaign_config.model_type),
            "optimization_mode": getattr(first.al_campaign_config.optimization_mode, "value",
                                         first.al_campaign_config.optimization_mode),
            "n_simulations": len(self.simulation_results),
            "n_successful": sum(1 for ssr in self.simulation_results if ssr.is_success()),
            "n_hits_threshold": first.al_simulation_config.stopping_config.n_hits,
            "is_discrete": first.al_campaign_config.optimization_mode == ActiveLearningOptimizationMode.DISCRETE,
            "discrete_targets": first.al_campaign_config.discrete_targets,
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
