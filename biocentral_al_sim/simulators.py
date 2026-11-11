from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cache
from typing import ClassVar, List, Optional

from biocentral_api import (
    ActiveLearningEngineeringCampaignConfig,
    ActiveLearningEngineeringSimulationConfig,
    ActiveLearningModelType,
    ActiveLearningScreeningCampaignConfig,
    ActiveLearningScreeningSimulationConfig,
    ActiveLearningStoppingConfig,
    BiocentralAPI,
    SequenceData,
)
from biotrainer_core.input_files import read_FASTA

from .datasets import ALSimulatorDataset
from .results import (
    ActiveLearningEngineeringSingleSimulationResult,
    ActiveLearningMultipleSimulationResult,
    ActiveLearningScreeningSingleSimulationResult,
    ActiveLearningSingleSimulationResult,
)
from .settings import ActiveLearningFixedBaseConfig
from .splits import ALSimulatorSplit, design_space


def get_simulator(
    dataset_id: ALSimulatorDataset,
    split_id: ALSimulatorSplit = ALSimulatorSplit.FULL_POOL,
) -> ActiveLearningScreeningSimulator:
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
    return ActiveLearningScreeningSimulator(al_base_config=base_config)


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


class ActiveLearningSimulator(ABC):
    """Runs simulated campaigns of one type, as set by the subclass."""

    result_cls: ClassVar[type[ActiveLearningSingleSimulationResult]]

    def __init__(self, al_base_config: ActiveLearningFixedBaseConfig):
        self.base_config = al_base_config

    @abstractmethod
    def get_simulation_config(self): ...

    @abstractmethod
    def _campaign_config(
        self, embedder_name: str, model_type: ActiveLearningModelType, seed: int
    ): ...

    @abstractmethod
    def _submit(self, campaign_config, simulation_config): ...

    def _simulation_config_fields(self) -> dict:
        """Fields every simulation config shares."""
        # start_ids and n_start are mutually exclusive: a pinned starting set or a random draw.
        start_ids = self.base_config.start_ids
        settings = self.base_config.settings
        return {
            "simulation_data": self.base_config.simulation_data,
            "n_start": None if start_ids else settings.n_start,
            "start_ids": start_ids,
            "n_suggestions_per_iteration": settings.n_suggestions_per_iteration,
            "stopping_config": ActiveLearningStoppingConfig(
                n_max_iterations=settings.n_iterations,
                max_labels_budget=settings.max_labels_budget,
                n_hits=settings.stop_at_n_hits,
                max_consecutive_failures=settings.stop_after_failed_rounds,
            ),
        }

    def _campaign_config_fields(
        self, embedder_name: str, model_type: ActiveLearningModelType, seed: int
    ) -> dict:
        """Fields every campaign config shares."""
        return {
            "name": "Test",  # TODO
            "embedder_name": embedder_name,
            "model_type": model_type,
            "optimization_mode": self.base_config.optimization_mode,
            "seed": seed,
        }

    def run_replicate(
        self,
        model_type: ActiveLearningModelType,
        embedder_name: str,
        seed: int,
        show_progress: bool = True,
    ) -> ActiveLearningSingleSimulationResult:
        al_campaign_config = self._campaign_config(embedder_name, model_type, seed)
        al_simulation_config = self.get_simulation_config()
        task = self._submit(al_campaign_config, al_simulation_config)
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

        return self.result_cls(
            dataset_id=self.base_config.dataset_id,
            split_id=self.base_config.split_id,
            al_campaign_config=al_campaign_config,
            al_simulation_config=al_simulation_config,
            simulation_result=result,
        )

    def simulate(
        self,
        embedder_name: str,
        model_type: ActiveLearningModelType,
        n_replicates: int,
        show_progress: bool = True,
    ) -> ActiveLearningMultipleSimulationResult:
        simulation_results = []
        for replicate_idx in range(n_replicates):
            if show_progress:
                print(f"Running replicate {replicate_idx + 1}/{n_replicates}...")
            seed = self.base_config.settings.first_seed + replicate_idx
            single_simulation_result = self.run_replicate(
                model_type=model_type,
                embedder_name=embedder_name,
                seed=seed,
                show_progress=show_progress,
            )
            simulation_results.append(single_simulation_result)
        return ActiveLearningMultipleSimulationResult(simulation_results)


class ActiveLearningScreeningSimulator(ActiveLearningSimulator):
    """Simulates a screening campaign: each round ranks the unlabelled pool and measures
    the top suggestions."""

    result_cls = ActiveLearningScreeningSingleSimulationResult

    def get_simulation_config(self) -> ActiveLearningScreeningSimulationConfig:
        return ActiveLearningScreeningSimulationConfig(
            **self._simulation_config_fields()
        )

    def _campaign_config(
        self, embedder_name: str, model_type: ActiveLearningModelType, seed: int
    ) -> ActiveLearningScreeningCampaignConfig:
        return ActiveLearningScreeningCampaignConfig(
            **self._campaign_config_fields(embedder_name, model_type, seed),
            target_lb=self.base_config.target_lb,
            target_ub=self.base_config.target_ub,
            target_value=self.base_config.target_value,
            discrete_targets=self.base_config.discrete_targets,
        )

    def _submit(self, campaign_config, simulation_config):
        return biocentral_api().al_screening_simulation(
            campaign_config=campaign_config, simulation_config=simulation_config
        )


class ActiveLearningEngineeringSimulator(ActiveLearningSimulator):
    """Simulates an engineering campaign: each round mutates the wild type and scores
    candidates drawn from the landscape, so the generator -- and features that only exist
    on the engineering path, such as delta embeddings -- are exercised end to end."""

    result_cls = ActiveLearningEngineeringSingleSimulationResult

    def get_simulation_config(self) -> ActiveLearningEngineeringSimulationConfig:
        settings = self.base_config.settings
        return ActiveLearningEngineeringSimulationConfig(
            **self._simulation_config_fields(),
            n_candidates_per_iteration=settings.n_candidates_per_iteration,
            generate_candidates=settings.generate_candidates,
            n_substitutions=settings.n_substitutions,
        )

    def _campaign_config(
        self, embedder_name: str, model_type: ActiveLearningModelType, seed: int
    ) -> ActiveLearningEngineeringCampaignConfig:
        settings = self.base_config.settings
        reference = self.base_config.dataset_id.reference_sequence()
        return ActiveLearningEngineeringCampaignConfig(
            **self._campaign_config_fields(embedder_name, model_type, seed),
            wildtype_sequence=reference,
            allowed_substitutions=(
                design_space(self.base_config.simulation_data, reference)
                if settings.generate_candidates
                else None
            ),
            use_delta_embeddings=settings.use_delta_embeddings,
        )

    def _submit(self, campaign_config, simulation_config):
        return biocentral_api().al_engineering_simulation(
            campaign_config=campaign_config, simulation_config=simulation_config
        )
