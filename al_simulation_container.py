from enum import Enum, auto
from typing import Dict, List, Optional

from pydantic import BaseModel, Field
from biocentral_api import ActiveLearningOptimizationMode


class ALSimulatorDatasetDefinition(BaseModel):
    """Everything that defines a simulation dataset: its data and its optimization target"""
    fasta_path: str = Field(description="Path to the labelled simulation FASTA")
    optimization_mode: ActiveLearningOptimizationMode = Field(description="How the labels are optimized")
    discrete_targets: Optional[List[str]] = Field(default=None, description="Target labels (mode: DISCRETE)")
    target_lb: Optional[float] = Field(default=None, description="Lower bound of the target (mode: INTERVAL)")
    target_ub: Optional[float] = Field(default=None, description="Upper bound of the target (mode: INTERVAL)")
    target_value: Optional[float] = Field(default=None, description="Target value (mode: VALUE)")


class ALSimulatorDataset(Enum):
    # The values are stored in the compressed dashboard data, so do not reorder these members
    MELTOME_MAXIMIZE = auto()
    MELTOME_MINIMIZE = auto()
    SCL = auto()
    AMYLASE = auto()
    PHOT = auto()
    EXOTOX = auto()

    @staticmethod
    def all():
        return [ALSimulatorDataset.MELTOME_MAXIMIZE,
                ALSimulatorDataset.MELTOME_MINIMIZE,
                ALSimulatorDataset.SCL,
                ALSimulatorDataset.AMYLASE,
                ALSimulatorDataset.PHOT,
                ALSimulatorDataset.EXOTOX]

    def definition(self) -> ALSimulatorDatasetDefinition:
        definition = _DATASET_DEFINITIONS.get(self.name)
        if definition is None:
            raise ValueError(f"No dataset definition for {self.name}.")
        return definition

    def to_path(self, path_override: Optional[Dict[str, str]] = None) -> str:
        path = None
        if path_override:
            path = path_override.get(self.name)
        if path is not None:
            return path
        return self.definition().fasta_path


_DATASET_DEFINITIONS: Dict[str, ALSimulatorDatasetDefinition] = {
    ALSimulatorDataset.MELTOME_MAXIMIZE.name: ALSimulatorDatasetDefinition(
        fasta_path="datasets/biotrainer_meltome_mixed_max2000.fasta",
        optimization_mode=ActiveLearningOptimizationMode.MAXIMIZE),
    ALSimulatorDataset.MELTOME_MINIMIZE.name: ALSimulatorDatasetDefinition(
        fasta_path="datasets/biotrainer_meltome_mixed_max2000.fasta",
        optimization_mode=ActiveLearningOptimizationMode.MINIMIZE),
    ALSimulatorDataset.SCL.name: ALSimulatorDatasetDefinition(
        fasta_path="datasets/scl_max2000.fasta",
        optimization_mode=ActiveLearningOptimizationMode.DISCRETE,
        discrete_targets=["Peroxisome"]),
    ALSimulatorDataset.AMYLASE.name: ALSimulatorDatasetDefinition(
        fasta_path="datasets/amylase_pet_max2000.fasta",
        optimization_mode=ActiveLearningOptimizationMode.MAXIMIZE),
    ALSimulatorDataset.PHOT.name: ALSimulatorDatasetDefinition(
        fasta_path="datasets/PHOT_CHLRE_Chen_2023_max2000.fasta",
        optimization_mode=ActiveLearningOptimizationMode.MAXIMIZE),
    ALSimulatorDataset.EXOTOX.name: ALSimulatorDatasetDefinition(
        fasta_path="datasets/exotox_merged_max2000.fasta",
        optimization_mode=ActiveLearningOptimizationMode.DISCRETE,
        discrete_targets=["EXOTOXIN"]),
}
