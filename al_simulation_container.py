from enum import Enum, auto
from biocentral_api import ActiveLearningOptimizationMode

from biotrainer_core.input_files import read_FASTA
from al_simulator import ActiveLearningFixedBaseConfig, ActiveLearningSimulator


class ALSimulatorDataset(Enum):
    MELTOME = auto()
    SCL = auto()
    AMYLASE = auto()

    @staticmethod
    def all():
        return [ALSimulatorDataset.MELTOME, ALSimulatorDataset.SCL, ALSimulatorDataset.AMYLASE]


def get_simulator(dataset_id: ALSimulatorDataset) -> ActiveLearningSimulator:
    match dataset_id:
        case ALSimulatorDataset.MELTOME:
            meltome_base_config = ActiveLearningFixedBaseConfig(
                simulation_data=read_FASTA("datasets/biotrainer_meltome_mixed_max2000.fasta"),
                optimization_mode=ActiveLearningOptimizationMode.MAXIMIZE)
            return ActiveLearningSimulator(al_base_config=meltome_base_config)
        case ALSimulatorDataset.SCL:
            scl_base_config = ActiveLearningFixedBaseConfig(simulation_data=read_FASTA("datasets/scl_max2000.fasta"),
                                                            optimization_mode=ActiveLearningOptimizationMode.DISCRETE,
                                                            discrete_targets=["Peroxisome"])
            return ActiveLearningSimulator(al_base_config=scl_base_config)
        case ALSimulatorDataset.AMYLASE:
            amylase_base_config = ActiveLearningFixedBaseConfig(simulation_data=read_FASTA("datasets/amylase_pet_max2000.fasta"),
                                                                 optimization_mode=ActiveLearningOptimizationMode.MAXIMIZE)
            return ActiveLearningSimulator(al_base_config=amylase_base_config)