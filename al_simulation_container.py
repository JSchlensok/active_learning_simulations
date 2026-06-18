from enum import Enum, auto


class ALSimulatorDataset(Enum):
    MELTOME_MAXIMIZE = auto()
    MELTOME_MINIMIZE = auto()
    SCL = auto()
    AMYLASE = auto()
    PHOT = auto()

    @staticmethod
    def all():
        return [ALSimulatorDataset.MELTOME_MAXIMIZE,
                ALSimulatorDataset.MELTOME_MINIMIZE,
                ALSimulatorDataset.SCL,
                ALSimulatorDataset.AMYLASE,
                ALSimulatorDataset.PHOT]
