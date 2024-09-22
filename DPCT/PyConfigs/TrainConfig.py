from dataclasses import dataclass, field
from typing import List, Dict
import datetime
import os


@dataclass
class ParamatersConfig:
    PLAN_NUM: int = 0

    # Learning parameters
    FINISH_EPISODE_NUM: int = 50000
    GAMMA: float = 0.95
    LR_Q: float = 2.0e-5
    ADAPT_LR: bool = True
    ADAPT_COEFF: float = 5.0e-5
    EXPERIENCE_BUFFER_SIZE: int = 256
    MAX_EPISODE_LENGTH: int = 256
    IL_MAX_EP_LENGTH: int = 64
    EPISODE_COUNT: int = 0

    # Observer parameters
    OBS_SIZE: int = 7
    SMALL_OBS_SIZE: int = 7
    NUM_FUTURE_STEPS: int = 3

    # Environment parameters
    ENVIRONMENT_SIZE_LIST: List[int] = field(default_factory=lambda: [10, 60])
    WALL_COMPONENTS_LIST: List[int] = field(default_factory=lambda: [1, 21])
    OBSTACLE_DENSITY_LIST: List[float] = field(default_factory=lambda: [0.0, 0.75])
    NUM_CHANNELS: Dict[float, int] = field(default_factory=lambda: {0: 8, 1: 9, 1.2: 9})

    A_SIZE: int = 5
    NUM_META_AGENTS: int = 9
    NUM_IL_META_AGENTS: int = 4
    NUM_THREADS: int = 8
    NUM_BUFFERS: int = 1

    # Training parameters
    SUMMARY_WINDOW: int = 10
    LOAD_MODEL: bool = False
    RESET_TRAINER: bool = False
    OUTPUT_FOLDER: str = "my_outputs"
    TRAINING_VERSION: str = "Original_7"
    MODEL_FOLDER: str = "models"
    OUTPUT_CONFIG_FILE: str = "config.yaml"
    GIFS_FOLDER: str = "train_gifs"
    TB_DATA_FOLDER: str = "tb_data"
    OUTPUT_GIFS: bool = False
    GIFS_FREQUENCY_RL: int = 128
    OUTPUT_IL_GIFS: bool = False
    IL_GIF_PROB: float = 0.0001

    # Imitation options
    PRIMING_LENGTH: int = 0
    MSTAR_CALL_FREQUENCY: int = 1

    # Others
    EPISODE_START: int = 0
    TRAINING: bool = True
    EPISODE_SAMPLES: int = 256
    GLOBAL_NET_SCOPE: str = "global"

    class JOB_OPTIONS:
        getExperience = 1
        getGradient = 2

    class COMPUTE_OPTIONS:
        multiThreaded = 1
        synchronous = 2

    def __post_init__(self):
        """Set the parameters that depend on YAML file"""
        # 現在時刻を取得(日本時間)
        t_delta = datetime.timedelta(hours=9)
        JST = datetime.timezone(t_delta, "JST")
        now = datetime.datetime.now(JST)

        self.ENVIRONMENT_SIZE: tuple = tuple(self.ENVIRONMENT_SIZE_LIST)
        self.WALL_COMPONENTS: tuple = tuple(self.WALL_COMPONENTS_LIST)
        self.OBSTACLE_DENSITY: tuple = tuple(self.OBSTACLE_DENSITY_LIST)

        EXPANDED_OBS_SIZE_LIST: Dict[float, int] = {
            0: None,  # Original
            1: self.OBS_SIZE * 2 - 1,  # Plan1
            1.2: self.OBS_SIZE + self.SMALL_OBS_SIZE - 1,  # Plan1.2
        }
        self.EXPANDED_OBS_SIZE: int = EXPANDED_OBS_SIZE_LIST[self.PLAN_NUM]

        self.TIMESTAMP: str = now.strftime("%Y%m%d_%H%M%S")
        # like "my_outputs/models/Original_7/240321_230831"
        self.MODEL_PATH: str = os.path.join(
            self.OUTPUT_FOLDER, self.MODEL_FOLDER, self.TRAINING_VERSION, self.TIMESTAMP
        )
        self.OUTPUT_CONFIG_FILE_PATH: str = os.path.join(self.MODEL_PATH, self.OUTPUT_CONFIG_FILE)
        self.GIFS_PATH: str = os.path.join(self.OUTPUT_FOLDER, self.GIFS_FOLDER, self.TRAINING_VERSION, self.TIMESTAMP)
        self.TRAIN_PATH: str = os.path.join(
            self.OUTPUT_FOLDER, self.TB_DATA_FOLDER, self.TRAINING_VERSION, self.TIMESTAMP
        )

        # Observation variables
        self.NUM_CHANNEL: int = self.NUM_CHANNELS[self.PLAN_NUM] + self.NUM_FUTURE_STEPS

        self.EPISODE_START: int = self.EPISODE_COUNT
        self.EPISODE_SAMPLES: int = self.EXPERIENCE_BUFFER_SIZE

        self.ENV_PARAMS: List[List[int]] = [
            [
                [self.WALL_COMPONENTS[0], self.WALL_COMPONENTS[1]],
                [self.OBSTACLE_DENSITY[0], self.OBSTACLE_DENSITY[1]],
            ]
            for _ in range(self.NUM_META_AGENTS)
        ]

        self.JOB_TYPE: int = self.JOB_OPTIONS.getGradient
        self.COMPUTE_TYPE: int = self.COMPUTE_OPTIONS.multiThreaded

        """Shared variables"""
        # Shared arrays for training
        self.swarm_reward = [0] * self.NUM_META_AGENTS
        self.swarm_targets = [0] * self.NUM_META_AGENTS

        # Shared arrays for tensorboard
        self.episode_rewards = [[] for _ in range(self.NUM_META_AGENTS)]
        self.episode_finishes = [[] for _ in range(self.NUM_META_AGENTS)]
        self.episode_lengths = [[] for _ in range(self.NUM_META_AGENTS)]
        self.episode_mean_values = [[] for _ in range(self.NUM_META_AGENTS)]
        self.episode_invalid_ops = [[] for _ in range(self.NUM_META_AGENTS)]
        self.episode_stop_ops = [[] for _ in range(self.NUM_META_AGENTS)]
        self.episode_wrong_blocking = [[] for _ in range(self.NUM_META_AGENTS)]
        self.rollouts = [None for _ in range(self.NUM_META_AGENTS)]
        self.GIF_frames = []

        # Joint variables
        self.joint_actions = [{} for _ in range(self.NUM_META_AGENTS)]
        self.joint_env = [None for _ in range(self.NUM_META_AGENTS)]
        self.joint_observations = [{} for _ in range(self.NUM_META_AGENTS)]
        self.joint_rewards = [{} for _ in range(self.NUM_META_AGENTS)]
        self.joint_done = [{} for _ in range(self.NUM_META_AGENTS)]
