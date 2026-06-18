from pathlib import Path
from pydantic import BaseModel, Field
from biocentral_api import ActiveLearningModelType, CommonEmbedder

from al_compress_reports import compress_reports
from al_simulation_container import ALSimulatorDataset
from al_simulator import ActiveLearningMultipleSimulationResult, get_simulator


class ExperimentConstants:
    n_rounds: int = 2  # TODO 5
    result_dir: Path = Path("simulation_v1_results/")


class ExperimentParametersV1(BaseModel):
    dataset_id: ALSimulatorDataset = Field(description="Dataset to use for the simulation")
    embedder_name: str = Field(description="Name of the embedder to use")
    model_type: ActiveLearningModelType = Field(description="Type of the model to use")

    def to_file_name(self):
        embedder_name = self.embedder_name.replace("/", "-")
        return f"al_sim_{self.dataset_id.name}_{embedder_name}_{self.model_type.value}.json"


def _create_experiment_params():
    experiment_params = []
    dataset_ids = ALSimulatorDataset.all()
    embedder_names = [CommonEmbedder.ONE_HOT_ENCODING.value]#, CommonEmbedder.ProtT5.value]
    model_types = [ActiveLearningModelType.GAUSSIAN_PROCESS, ActiveLearningModelType.FNN_MCD,
                   ActiveLearningModelType.RANDOM]
    for dataset_id in dataset_ids:
        for embedder_name in embedder_names:
            for model_type in model_types:
                experiment_params.append(
                    ExperimentParametersV1(dataset_id=dataset_id, embedder_name=embedder_name, model_type=model_type))
    # TODO DEBUG
    #experiment_params = [
    #    ExperimentParametersV1(dataset_id=ALSimulatorDataset.PHOT,
    #                           model_type=ActiveLearningModelType.GAUSSIAN_PROCESS,
    #                           embedder_name=CommonEmbedder.ONE_HOT_ENCODING.value)
    #]
    return experiment_params


def _run_experiment(experiment_params: ExperimentParametersV1):
    if not ExperimentConstants.result_dir.exists():
        ExperimentConstants.result_dir.mkdir(parents=True, exist_ok=True)

    save_dir = ExperimentConstants.result_dir / experiment_params.to_file_name()
    if save_dir.exists():
        sim_result = ActiveLearningMultipleSimulationResult.from_json(save_dir)
    else:
        al_simulator = get_simulator(experiment_params.dataset_id)
        print(f"Running simulation for {experiment_params}..")
        sim_result = al_simulator.simulate(model_type=experiment_params.model_type,
                                           embedder_name=experiment_params.embedder_name,
                                           n_rounds=ExperimentConstants.n_rounds)
        sim_result.save(save_dir)
    sim_result.print_stats()


def main():
    experiment_params = _create_experiment_params()
    for experiment_param in experiment_params:
        _run_experiment(experiment_param)
    print("All simulations completed. Compressing reports...")
    compress_reports()


if __name__ == "__main__":
    main()
