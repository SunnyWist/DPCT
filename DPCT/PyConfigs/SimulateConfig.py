from dataclasses import dataclass, field
from typing import List, Union
import os


@dataclass
class EnvConfig:
    LST: List[Union[int, float]] = field(default_factory=lambda: [4, 8, 16, 32, 64, 128, 256])
    SEP: bool = False


@dataclass
class SimulateConfig:
    SIMULATE_WITH_ONE_ENV: bool = False

    AGENT: EnvConfig = EnvConfig()
    SIZE: EnvConfig = EnvConfig()
    DENSITY: EnvConfig = EnvConfig()
    WALL: EnvConfig = EnvConfig()
    ID_NUM: int = 20

    ENVS_FOLDER: str = "envs"
    ENV_FILES_FOLDER: str = "env_files"
    FIRST_FOLDER_FORMAT = "{}size_{}density_{}wall"
    ENV_FILE_FORMAT = "{}agents_{}size_{}density_{}wall_id{}"

    OUTPUT_FOLDER: str = "my_outputs"
    RESULTS_FOLDER: str = "results"
    MODELS_FOLDER: str = "models"
    MODEL_NAME: str = "Original_7"
    MODEL_TIMESTAMP: str = ""
    MODEL_CONFIG_FILE: str = "config.yaml"

    NUMBER_OF_WORKERS: int = 8
    """Number of simulation workers for parallel processing."""

    PRINT_INFO: bool = True
    RESUME_TESTING: int = 1  # 0: False, 1: True

    ALL_RESULTS_PICKLE_FILE: str = "all_results.pkl"
    ALL_RESULTS_CSV_FILE: str = "all_results.csv"

    USE_NPZ: bool = False

    WITH_EP: bool = False  # エンドポイントを利用するかどうか

    def __post_init__(self):
        self.ENV_FILES_PATH: str = os.path.join(self.ENVS_FOLDER, self.ENV_FILES_FOLDER)

        self.RESULTS_PATH: str = os.path.join(self.OUTPUT_FOLDER, self.RESULTS_FOLDER)
        if len(self.MODEL_TIMESTAMP) == 0:
            specified_models_folder: str = os.path.join(self.OUTPUT_FOLDER, self.MODELS_FOLDER, self.MODEL_NAME)
            model_folders: List[str] = sorted(os.listdir(specified_models_folder))
            self.MODEL_TIMESTAMP: str = model_folders[-1]
            self.MODEL_PATH: str = os.path.join(specified_models_folder, self.MODEL_TIMESTAMP)
        else:
            self.MODEL_PATH: str = os.path.join(
                self.OUTPUT_FOLDER, self.MODELS_FOLDER, self.MODEL_NAME, self.MODEL_TIMESTAMP
            )
        self.MODEL_CONFIG_FILE_PATH: str = os.path.join(self.MODEL_PATH, self.MODEL_CONFIG_FILE)
