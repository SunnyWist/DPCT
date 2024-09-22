import numpy as np
from typing import List, Dict, Tuple, Union, Any

from Env_Builder import World


def _get_one_hot_for_agent_direction(agent):
    """Returns the agent's direction to one-hot encoding."""
    direction = np.zeros(4)
    direction[agent.direction] = 1
    return direction


class ObservationBuilder:
    """
    ObservationBuilder base class.
    """

    def __init__(self):
        self.world: World
        self.NUM_CHANNELS: int

    def set_env(self, env: World):
        self.world: World = env

    def reset(self):
        """
        Called after each environment reset.
        """
        raise NotImplementedError()

    def get_many(self, handles: Union[List[int], None]) -> Dict[int, List[np.ndarray]]:
        """
        Called whenever an observation has to be computed for the `env` environment, for each agent with handle
        in the `handles` list.

        Parameters
        ----------
        handles : list of handles, optional
            List with the handles of the agents for which to compute the observation vector.

        Returns
        -------
        function
            A dictionary of observation structures, specific to the corresponding environment, with handles from
            `handles` as keys.
        """
        raise NotImplementedError

    def get_obs_size(self) -> int:
        """
        各エージェントの視野の一辺の長さを返す(Original: OBS_SIZE, Plan1 or Plan1.2: EXPANDED_OBS_SIZE)
        Returns the size of the observation vector for each agent.
        If expanded_observation is definedm return the size of the expanded observation.
        """
        raise NotImplementedError()


class DummyObserver(ObservationBuilder):
    """
    DummyObservationBuilder class which returns dummy observations
    This is used in the evaluation service
    """

    def __init__(self):
        super().__init__()
        self.observation_size = 1

    def reset(self):
        pass

    def get_many(self, handles) -> bool:
        return True

    def get(self, handle: int = 0) -> bool:
        return True

    def get_obs_size(self) -> int:
        return self.observation_size
