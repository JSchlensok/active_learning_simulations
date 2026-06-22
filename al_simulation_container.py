from enum import Enum, auto
from typing import Dict, Optional


class ALSimulatorDataset(Enum):
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

    def to_path(self, path_override: Optional[Dict[str, str]] = None) -> str:
        path = None
        if path_override:
            path = path_override.get(self.name)
        if path is not None:
            return path
        default_dict = {
            ALSimulatorDataset.MELTOME_MAXIMIZE.name: "datasets/biotrainer_meltome_mixed_max2000.fasta",
            ALSimulatorDataset.MELTOME_MINIMIZE.name: "datasets/biotrainer_meltome_mixed_max2000.fasta",
            ALSimulatorDataset.SCL.name: "datasets/scl_max2000.fasta",
            ALSimulatorDataset.AMYLASE.name: "datasets/amylase_pet_max2000.fasta",
            ALSimulatorDataset.PHOT.name: "datasets/PHOT_CHLRE_Chen_2023_max2000.fasta",
            ALSimulatorDataset.EXOTOX.name: "datasets/exotox_merged_max2000.fasta",
        }
        return default_dict[self.name]