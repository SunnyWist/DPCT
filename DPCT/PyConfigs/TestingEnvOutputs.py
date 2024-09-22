from dataclasses import dataclass
from typing import List


@dataclass
class TestingEnvOutputs:
    """Outputs for testing env"""

    target_reached: int
    computing_time_list: List
    num_crash: int
    episode_status: str
    succeed_episode: bool
    step_count: int
