from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from .datasets import ALSimulatorDataset
from .splits import ALSimulatorSplit


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
