from __future__ import annotations

import hashlib
from typing import List, Optional

from biocentral_api import ActiveLearningOptimizationMode, SequenceData
from pydantic import BaseModel, ConfigDict, Field

from .datasets import ALSimulatorDataset
from .splits import DEFAULT_SEED, ALSimulatorSplit


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
    first_seed: int = Field(
        default=DEFAULT_SEED,
        description="Seed of the first replicate",
    )
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


class EngineeringCampaignSettings(CampaignSettings):
    """Campaign settings for a simulated engineering campaign.

    A subclass rather than extra fields on CampaignSettings: the fingerprint hashes every
    field, so widening the base would rename every existing screening result file.
    """

    n_candidates_per_iteration: int = Field(
        default=1000, description="Unlabelled landscape variants scored per round")
    use_delta_embeddings: bool = Field(
        default=False,
        description="Subtract the wildtype embedding from each variant's (PLM embedders only)")


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

    # Split config (see splits). FULL_POOL is the identity split and changes nothing.
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
