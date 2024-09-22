from dataclasses import dataclass, field
from typing import List, Union
import os


@dataclass
class GeneratorConfig:
    # Add Info or Generate New Maps
    ADD_INFO: bool = False

    # For MazeTestInfoAdder
    WITH_EP: bool = False
    SAVE_FOLDER: str = "envs/new_env_files"
    PRINT_INFO: bool = True
    USE_NPZ: bool = False
    ADDED_FOLDER_NAME: str = "added"

    # Only use MazeTestGenerator
    PURE_MAPS: bool = False
    AGENT_LIST: List[int] = field(default_factory=lambda: [4, 8, 16, 32, 64, 128, 256])
    SIZE_LIST: List[int] = field(default_factory=lambda: [20, 40, 80, 160])
    DENSITY_LIST: List[float] = field(default_factory=lambda: [0.3, 0.65])
    WALL_LIST: List[int] = field(default_factory=lambda: [1, 10, 20])
    TESTS_TO_GENERATE: int = 20

    def __post_init__(self):
        self.SAVE_PATH: str = os.path.join(self.SAVE_FOLDER)
        if not os.path.exists(self.SAVE_PATH):
            os.makedirs(self.SAVE_PATH)
