# Env_Builder
import copy
from operator import sub, add
import gym
import numpy as np
import math
import warnings
import time
from matplotlib.colors import *  # type: ignore
from gym.envs.classic_control import rendering
import imageio
from gym import spaces
from typing import List, Dict, Tuple, Union, Any, Optional, Set, overload

from GroupLock import Lock
from od_mstar3.col_set_addition import OutOfTimeError, NoSolutionError
from od_mstar3 import od_mstar
from Map_Generator import *

try:
    from od_mstar3 import cpp_mstar
except ImportError:
    raise ImportError("cpp_mstar not compiled. Please refer to README")
try:
    from astarlib3 import astarlib

    USE_Cython_ASTAR = True
except ImportError:
    USE_Cython_ASTAR = False
    raise ImportError("cpp_aStar not compiled. Please refer to README.")


def make_gif(images, fname):
    gif = imageio.mimwrite(fname, images, subrectangles=True)
    print("wrote gif")
    return gif


def opposite_actions(action, isDiagonal=False):
    if isDiagonal:
        raise NotImplemented
    else:
        checking_table = {0: -1, 1: 3, 2: 4, 3: 1, 4: 2}
    return checking_table[action]


def action2dir(action: int) -> Tuple[int, int]:
    checking_table = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (-1, 0)}
    return checking_table[action]


def dir2action(direction: Tuple) -> int:
    checking_table = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3, (-1, 0): 4}
    return checking_table[direction]


@overload
def tuple_plus(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int]: ...


@overload
def tuple_plus(a: Tuple, b: Tuple) -> Tuple: ...


def tuple_plus(a: Tuple, b: Tuple) -> Tuple:
    """a + b"""
    return tuple(map(add, a, b))


def tuple_minus(a: Tuple, b: Tuple) -> Tuple:
    """a - b"""
    return tuple(map(sub, a, b))


def _heap(ls, max_length):
    while True:
        if len(ls) > max_length:
            ls.pop(0)
        else:
            return ls


def get_key(dict1: Dict, val) -> List:
    return [k for k, v in dict1.items() if v == val]


def getAstarDistanceMap(map: np.ndarray, start: tuple, goal: tuple, isCPython) -> np.ndarray:
    """
    returns a numpy array of same dims as map with the distance to the goal from each coord
    :param map: a n by m np array, where -1 denotes obstacle
    :param start: start_position
    :param goal: goal_position
    :return: optimal distance map
    """

    def lowestF(fScore, openSet):
        # find entry in openSet with lowest fScore
        assert len(openSet) > 0
        minF = 2**31 - 1
        minNode = None
        for i, j in openSet:
            if (i, j) not in fScore:
                continue
            if fScore[(i, j)] < minF:
                minF = fScore[(i, j)]
                minNode = (i, j)
        return minNode

    def getNeighbors(node):
        # return set of neighbors to the given node
        n_moves = 5
        neighbors = set()
        for move in range(1, n_moves):  # we dont want to include 0 or it will include itself
            direction = action2dir(move)
            dx = direction[0]
            dy = direction[1]
            ax = node[0]
            ay = node[1]
            if ax + dx >= map.shape[0] or ax + dx < 0 or ay + dy >= map.shape[1] or ay + dy < 0:  # out of bounds
                continue
            if map[ax + dx, ay + dy] == -1:  # collide with static obstacle
                continue
            neighbors.add((ax + dx, ay + dy))
        return neighbors

    if not isCPython:
        # NOTE THAT WE REVERSE THE DIRECTION OF SEARCH SO THAT THE GSCORE WILL BE DISTANCE TO GOAL
        start, goal = goal, start
        start, goal = tuple(start), tuple(goal)
        # The set of nodes already evaluated
        closedSet = set()

        # The set of currently discovered nodes that are not evaluated yet.
        # Initially, only the start node is known.
        openSet = set()
        openSet.add(start)

        # For each node, which node it can most efficiently be reached from.
        # If a node can be reached from many nodes, cameFrom will eventually contain the
        # most efficient previous step.
        cameFrom = dict()

        # For each node, the cost of getting from the start node to that node.
        gScore = dict()  # default value infinity

        # The cost of going from start to start is zero.
        gScore[start] = 0

        # For each node, the total cost of getting from the start node to the goal
        # by passing by that node. That value is partly known, partly heuristic.
        fScore = dict()  # default infinity

        # our heuristic is euclidean distance to goal
        heuristic_cost_estimate = lambda x, y: math.hypot(x[0] - y[0], x[1] - y[1])

        # For the first node, that value is completely heuristic.
        fScore[start] = heuristic_cost_estimate(start, goal)

        while len(openSet) != 0:
            # current = the node in openSet having the lowest fScore value
            current = lowestF(fScore, openSet)

            openSet.remove(current)
            closedSet.add(current)
            for neighbor in getNeighbors(current):
                if neighbor in closedSet:
                    continue  # Ignore the neighbor which is already evaluated.

                if neighbor not in openSet:  # Discover a new node
                    openSet.add(neighbor)

                # The distance from start to a neighbor
                # in our case the distance between is always 1
                tentative_gScore = gScore[current] + 1
                if tentative_gScore >= gScore.get(neighbor, 2**31 - 1):
                    continue  # This is not a better path.

                # This path is the best until now. Record it!
                cameFrom[neighbor] = current
                gScore[neighbor] = tentative_gScore
                fScore[neighbor] = gScore[neighbor] + heuristic_cost_estimate(neighbor, goal)

                # parse through the gScores
        Astar_map = map.copy()
        for i, j in gScore:
            Astar_map[i, j] = gScore[i, j]
        return Astar_map
    else:
        planner = astarlib.aStar(array=map)  # where 0 is free space, -1 is obstacle
        return planner.getAstarDistanceMap(goal)  # should give you the distance map for a given goal


def unpack_copy_int_list(value_list: List[int]) -> List[int]:
    """アンパックを利用してint型のリストのディープコピーを行う

    Args:
        value_list (List[int]): コピー元のリスト(int)

    Returns:
        List[int]: コピーしたリスト
    """
    return [*value_list]


def unpack_copy_float_list(value_list: List[float]) -> List[float]:
    """アンパックを利用してfloat型のリストのディープコピーを行う

    Args:
        value_list (List[float]): コピー元のリスト(float)

    Returns:
        List[float]: コピーしたリスト
    """
    return [*value_list]


def unpack_copy_int_set(value_set: Set[int]) -> Set[int]:
    """アンパックを利用してint型のセットのディープコピーを行う

    Args:
        value_set (Set[int]): コピー元のセット

    Returns:
        Set[int]: コピーしたセット
    """
    return {*value_set}


class Agent:
    """
    The agent object that contains agent's position, direction dict and position dict,
    currently only supporting 4-connected region.
    self.distance_map is None here. Assign values in upper class.
    ###########
    WARNING: direction_history[i] means the action taking from i-1 step, resulting in the state of step i,
    such that len(direction_history) == len(position_history)
    ###########
    """

    def __init__(self, isDiagonal=False):
        self._path_count: int = -1
        self.IsDiagonal: bool = isDiagonal
        self.freeze: int = 0

        self.position: Optional[Tuple[int, int]] = None
        self.position_history: List[Tuple[int, int]] = []
        self.ID: Optional[int] = None
        self.direction: Optional[Tuple[int, int]] = None
        self.direction_history: List[Tuple] = [(None, None)]
        self.action_history: List[Optional[int]] = []
        self.goal_pos: Optional[Tuple[int, int]] = None
        self.distanceMap: Optional[np.ndarray] = None
        self.dones: int = 0
        self.status: Optional[int] = None
        self.next_goal: Optional[Tuple[int, int]] = None
        self.next_distanceMap: Optional[np.ndarray] = None

    def reset(self):
        self._path_count = -1
        self.freeze = 0
        (
            self.position,
            self.position_history,
            self.ID,
            self.direction,
            self.direction_history,
            self.action_history,
            self.goal_pos,
            self.distanceMap,
            self.dones,
            self.status,
            self.next_goal,
            self.next_distanceMap,
        ) = (None, [], None, None, [(None, None)], [], None, None, 0, None, None, None)

    def move(self, pos: Union[Tuple[int, int], List[int], None], status: Optional[int] = None):
        """エージェントが移動するための処理

        Args:
            pos (Optional[Tuple[int, int]]): 移動先の座標
            status (Optional[int], optional): エージェントの状態. Defaults to None.
        """
        if pos is None:
            pos = self.position
        if self.position is not None:
            assert pos in [
                self.position,
                tuple_plus(self.position, (0, 1)),
                tuple_plus(self.position, (0, -1)),
                tuple_plus(self.position, (1, 0)),
                tuple_plus(self.position, (-1, 0)),
            ], "only 1 step 1 cell allowed. Previous pos:" + str(self.position)
        assert pos is not None
        tuple_pos = tuple(pos)
        assert len(tuple_pos) == 2
        self.add_history(tuple_pos, status)

    def add_history(self, position: Tuple[int, int], status: Optional[int] = None):
        if position is None:
            raise ValueError("position is None")
        self.status = status
        self._path_count += 1
        self.position = position
        if self._path_count != 0:
            direction = tuple_minus(position, self.position_history[-1])
            action = dir2action(direction)
            assert action in list(range(4 + 1)), "direction not in actionDir, something going wrong"
            self.direction_history.append(direction)
            self.action_history.append(action)
        self.position_history.append(position)

        self.position_history = _heap(self.position_history, 30)
        self.direction_history = _heap(self.direction_history, 30)
        self.action_history = _heap(self.action_history, 30)


class World:
    """
    Include: basic world generation rules, blank map generation and collision checking.
    reset_world:
    Do not add action pruning, reward structure or any other routine for training in this class. Pls add in upper class MAPFEnv
    """

    def __init__(self, map_generator, num_agents, isDiagonal=False):
        self.num_agents = num_agents
        self.manual_world = False
        self.manual_goal = False
        self.goal_generate_distance = 2

        self.map_generator = map_generator
        self.isDiagonal = isDiagonal

        self.agents_init_pos, self.goals_init_pos = None, None
        self.reset_world()
        self.init_agents_and_goals()

        self.goal_candidate_map = None

    def reset_world(self):
        """
        generate/re-generate a world map, and compute its corridor map
        """

        def scan_for_agents(state_map):
            agents = {}
            for i in range(state_map.shape[0]):
                for j in range(state_map.shape[1]):
                    if state_map[i, j] > 0:
                        agentID = state_map[i, j]
                        agents.update({agentID: (i, j)})
            return agents

        self.state, self.goals_map = self.map_generator()
        # detect manual world
        if (self.state > 0).any():
            self.manual_world = True
            self.agents_init_pos = scan_for_agents(self.state)
            if self.num_agents is not None and self.num_agents != len(self.agents_init_pos.keys()):
                warnings.warn(
                    "num_agent does not match the actual agent number in manual map! "
                    "num_agent has been set to be consistent with manual map."
                )
            self.num_agents = len(self.agents_init_pos.keys())
            self.agents = {i: copy.deepcopy(Agent()) for i in range(1, self.num_agents + 1)}
        else:
            assert self.num_agents is not None
            self.agents = {i: copy.deepcopy(Agent()) for i in range(1, self.num_agents + 1)}
        # detect manual goals_map
        if self.goals_map is not None:
            self.manual_goal = True
            self.goals_init_pos = scan_for_agents(self.goals_map) if self.manual_goal else None

        else:
            self.goals_map = np.zeros([self.state.shape[0], self.state.shape[1]])

        self.corridor_map = {}
        self.restrict_init_corridor = True
        self.visited = []
        self.corridors = {}
        self.get_corridors()

    def reset_agent(self):
        """
        remove all the agents (with their travel history) and goals in the env, rebase the env into a blank one
        """
        self.agents = {i: copy.deepcopy(Agent()) for i in range(1, self.num_agents + 1)}
        self.state[self.state > 0] = 0  # remove agents in the map

    def reset_agent_next_goal(self):
        """
        reset the next goal of each agent
        """
        for agent in self.agents.values():
            agent.next_goal = None

    def set_goal_candidate_map(self, goal_candidate_map: np.ndarray):
        self.goal_candidate_map = goal_candidate_map

    def get_corridors(self):
        """
        in corridor_map , output = list:
            list[0] : if In corridor, corridor id , else -1
            list[1] : If Inside Corridor = 1
                      If Corridor Endpoint = 2
                      If Free Cell Outside Corridor = 0
                      If Obstacle = -1
        """
        corridor_count = 1
        # Initialize corridor map
        for i in range(self.state.shape[0]):
            for j in range(self.state.shape[1]):
                if self.state[i, j] >= 0:
                    self.corridor_map[(i, j)] = [-1, 0]
                else:
                    self.corridor_map[(i, j)] = [-1, -1]
        # Compute All Corridors and End-points, store them in self.corridors , update corridor_map
        for i in range(self.state.shape[0]):
            for j in range(self.state.shape[1]):
                positions = self.blank_env_valid_neighbor(i, j)
                if (positions.count(None)) == 2 and (i, j) not in self.visited:
                    allowed = self.check_for_singular_state(positions)
                    if not allowed:
                        continue
                    self.corridors[corridor_count] = {}
                    self.corridors[corridor_count]["Positions"] = [(i, j)]
                    self.corridor_map[(i, j)] = [corridor_count, 1]
                    self.corridors[corridor_count]["EndPoints"] = []
                    self.visited.append((i, j))
                    for num in range(4):
                        pos_num = positions[num]
                        if pos_num is not None:
                            self.visit(pos_num[0], pos_num[1], corridor_count)
                    corridor_count += 1
        # Get Delta X , Delta Y for the computed corridors ( Delta= Displacement to corridor exit)
        for k in range(1, corridor_count):
            if k in self.corridors:
                if len(self.corridors[k]["EndPoints"]) == 2:
                    self.corridors[k]["DeltaX"] = {}
                    self.corridors[k]["DeltaY"] = {}
                    pos_a = self.corridors[k]["EndPoints"][0]
                    pos_b = self.corridors[k]["EndPoints"][1]
                    self.corridors[k]["DeltaX"][pos_a] = pos_a[0] - pos_b[0]  # / (max(1, abs(pos_a[0] - pos_b[0])))
                    self.corridors[k]["DeltaX"][pos_b] = -1 * self.corridors[k]["DeltaX"][pos_a]
                    self.corridors[k]["DeltaY"][pos_a] = pos_a[1] - pos_b[1]  # / (max(1, abs(pos_a[1] - pos_b[1])))
                    self.corridors[k]["DeltaY"][pos_b] = -1 * self.corridors[k]["DeltaY"][pos_a]
            else:
                print("Weird2")

                # Rearrange the computed corridor list such that it becomes easier to iterate over the structure
        # Basically, sort the self.corridors['Positions'] list in a way that the first element of the list is
        # adjacent to Endpoint[0] and the last element of the list is adjacent to EndPoint[1]
        # If there is only 1 endpoint, the sorting doesn't matter since blocking is easy to compute
        for t in range(1, corridor_count):
            positions = self.blank_env_valid_neighbor(
                self.corridors[t]["EndPoints"][0][0], self.corridors[t]["EndPoints"][0][1]
            )
            for position in positions:
                if position is not None and self.corridor_map[position][0] == t:
                    break
            index = self.corridors[t]["Positions"].index(position)

            if index == 0:
                pass
            if index != len(self.corridors[t]["Positions"]) - 1:
                temp_list = self.corridors[t]["Positions"][0 : index + 1]
                temp_list.reverse()
                temp_end = self.corridors[t]["Positions"][index + 1 :]
                self.corridors[t]["Positions"] = []
                self.corridors[t]["Positions"].extend(temp_list)
                self.corridors[t]["Positions"].extend(temp_end)

            elif index == len(self.corridors[t]["Positions"]) - 1 and len(self.corridors[t]["EndPoints"]) == 2:
                positions2 = self.blank_env_valid_neighbor(
                    self.corridors[t]["EndPoints"][1][0], self.corridors[t]["EndPoints"][1][1]
                )
                for position2 in positions2:
                    if position2 is not None and self.corridor_map[position2][0] == t:
                        break
                index2 = self.corridors[t]["Positions"].index(position2)
                temp_list = self.corridors[t]["Positions"][0 : index2 + 1]
                temp_list.reverse()
                temp_end = self.corridors[t]["Positions"][index2 + 1 :]
                self.corridors[t]["Positions"] = []
                self.corridors[t]["Positions"].extend(temp_list)
                self.corridors[t]["Positions"].extend(temp_end)
                self.corridors[t]["Positions"].reverse()
            else:
                if len(self.corridors[t]["EndPoints"]) == 2:
                    print("Weird3")

            self.corridors[t]["StoppingPoints"] = []
            if len(self.corridors[t]["EndPoints"]) == 2:
                position_first = self.corridors[t]["Positions"][0]
                position_last = self.corridors[t]["Positions"][-1]
                self.corridors[t]["StoppingPoints"].append([position_first[0], position_first[1]])
                self.corridors[t]["StoppingPoints"].append([position_last[0], position_last[1]])
            else:
                position_first = self.corridors[t]["Positions"][0]
                if position is None:
                    raise ValueError("position is None")
                self.corridors[t]["StoppingPoints"].append([position[0], position[1]])
                self.corridors[t]["StoppingPoints"].append(None)
        return

    def check_for_singular_state(self, positions):
        counter = 0
        for num in range(4):
            if positions[num] is not None:
                new_positions = self.blank_env_valid_neighbor(positions[num][0], positions[num][1])
                if new_positions.count(None) in [2, 3]:
                    counter += 1
        return counter > 0

    def visit(self, i: int, j: int, corridor_id: int):
        positions = self.blank_env_valid_neighbor(i, j)
        if positions.count(None) in [0, 1]:
            self.corridors[corridor_id]["EndPoints"].append((i, j))
            self.corridor_map[(i, j)] = [corridor_id, 2]
            return
        elif positions.count(None) in [2, 3]:
            self.visited.append((i, j))
            self.corridors[corridor_id]["Positions"].append((i, j))
            self.corridor_map[(i, j)] = [corridor_id, 1]
            for num in range(4):
                pos_num = positions[num]
                if pos_num is not None and pos_num not in self.visited:
                    self.visit(pos_num[0], pos_num[1], corridor_id)
        else:
            print("Weird")

    def blank_env_valid_neighbor(self, i: int, j: int) -> List[Optional[Tuple]]:
        possible_positions: List[Optional[Tuple]] = [None, None, None, None]
        move = [[0, 1], [1, 0], [-1, 0], [0, -1]]
        if self.state[i, j] == -1:
            return possible_positions
        else:
            for num in range(4):
                x = i + move[num][0]
                y = j + move[num][1]
                if 0 <= x < self.state.shape[0] and 0 <= y < self.state.shape[1]:
                    if self.state[x, y] != -1:
                        possible_positions[num] = (x, y)
                        continue
        return possible_positions

    def getPos(self, agent_id: int) -> Tuple:
        position = self.agents[agent_id].position
        if position is None:
            raise ValueError("Agent not in the map")
        return tuple(position)

    def getDone(self, agent_id: int) -> int:
        # get the number of goals that an agent has finished
        return self.agents[agent_id].dones

    def get_history(self, agent_id: int, path_id: Optional[int] = None) -> Tuple:
        """
        :param: path_id: if None, get the last step
        :return: past_pos: (x,y), past_direction: int
        """
        if path_id is None:
            path_id = self.agents[agent_id]._path_count - 1 if self.agents[agent_id]._path_count > 0 else 0
        try:
            return self.agents[agent_id].position_history[path_id], self.agents[agent_id].direction_history[path_id]
        except IndexError:
            print("you are giving an invalid path_id")
            raise ValueError

    def getGoal(self, agent_id) -> Tuple:
        position = self.agents[agent_id].goal_pos
        if position is None:
            raise ValueError("Goal not in the map")
        return tuple(position)

    def init_agents_and_goals(self):
        """
        place all agents and goals in the blank env. If turning on corridor population restriction, only 1 agent is
        allowed to be born in each corridor.
        """

        def corridor_restricted_init_poss(state_map, corridor_map, goal_map, id_list=None):
            """
            generate agent init positions when corridor init population is restricted
            return a dict of positions {agentID:(x,y), ...}
            """
            if id_list is None:
                id_list = list(range(1, self.num_agents + 1))

            free_space1 = list(np.argwhere(state_map == 0))
            free_space1 = [tuple(pos) for pos in free_space1]
            corridors_visited = []
            manual_positions = {}
            break_completely = False
            for idx in id_list:
                if break_completely:
                    return None
                pos_set = False
                agentID = idx
                while not pos_set:
                    try:
                        assert len(free_space1) > 1
                        random_pos = np.random.choice(len(free_space1))
                    except AssertionError or ValueError:
                        print("wrong agent")
                        self.reset_world()
                        self.init_agents_and_goals()
                        break_completely = True
                        if idx == id_list[-1]:
                            return None
                        break
                    position = free_space1[random_pos]
                    cell_info = corridor_map[position[0], position[1]][1]
                    if cell_info in [0, 2]:
                        if goal_map[position[0], position[1]] != agentID:
                            manual_positions.update({idx: (position[0], position[1])})
                            free_space1.remove(position)
                            pos_set = True
                    elif cell_info == 1:
                        corridor_id = corridor_map[position[0], position[1]][0]
                        if corridor_id not in corridors_visited:
                            if goal_map[position[0], position[1]] != agentID:
                                manual_positions.update({idx: (position[0], position[1])})
                                corridors_visited.append(corridor_id)
                                free_space1.remove(position)
                                pos_set = True
                        else:
                            free_space1.remove(position)
                    else:
                        print("Very Weird")
            return manual_positions

        # no corridor population restriction
        if not self.restrict_init_corridor or (self.restrict_init_corridor and self.manual_world):
            self.put_goals(list(range(1, self.num_agents + 1)), self.goals_init_pos)
            self._put_agents(list(range(1, self.num_agents + 1)), self.agents_init_pos)
        # has corridor population restriction
        else:
            check = self.put_goals(list(range(1, self.num_agents + 1)), self.goals_init_pos)
            if check is not None:
                manual_positions = corridor_restricted_init_poss(self.state, self.corridor_map, self.goals_map)
                if manual_positions is not None:
                    self._put_agents(list(range(1, self.num_agents + 1)), manual_positions)

    def _put_agents(self, id_list, manual_pos=None):
        """
        put some agents in the blank env, saved history data in self.agents and self.state
        get distance map for the agents
        :param id_list: a list of agent_id
                manual_pos: a dict of manual positions {agentID: (x,y),...}
        """
        if manual_pos is None:
            # randomly init agents everywhere
            free_space = np.argwhere(np.logical_or(self.state == 0, self.goals_map == 0) == 1)
            new_idx = np.random.choice(len(free_space), size=len(id_list), replace=False)
            init_poss = [free_space[idx] for idx in new_idx]
        else:
            assert len(manual_pos.keys()) == len(id_list)
            init_poss = [manual_pos[agentID] for agentID in id_list]
        assert len(init_poss) == len(id_list)
        self.agents_init_pos = {}
        for idx, agentID in enumerate(id_list):
            self.agents[agentID].ID = agentID
            if (
                self.state[init_poss[idx][0], init_poss[idx][1]] in [0, agentID]
                and self.goals_map[init_poss[idx][0], init_poss[idx][1]] != agentID
            ):
                self.state[init_poss[idx][0], init_poss[idx][1]] = agentID
                self.agents_init_pos.update({agentID: (init_poss[idx][0], init_poss[idx][1])})
            else:
                print(self.state)
                print(init_poss)
                raise ValueError("invalid manual_pos for agent" + str(agentID) + " at: " + str(init_poss[idx]))
            self.agents[agentID].move(init_poss[idx])
            agent_pos = self.agents[agentID].position
            agent_goal = self.agents[agentID].goal_pos
            assert agent_pos is not None and agent_goal is not None
            self.agents[agentID].distanceMap = getAstarDistanceMap(
                self.state, agent_pos, agent_goal, isCPython=USE_Cython_ASTAR
            )

    def put_goals(self, id_list: List[int], manual_pos: Optional[Dict[int, Tuple[int, int]]] = None):
        """エージェントがゴールに到達した場合、新しいゴールを設定する

        Args:
            id_list (List[int]): ゴールにたどり着いたエージェントのIDのリスト
            manual_pos ([type], optional): ゴールの座標. Defaults to None.

        Returns:
            [type]: [description]

        put a goal of single agent in the env, if the goal already exists, remove that goal and put a new one
        :param manual_pos: a dict of manual_pos {agentID: (x, y)}
        :param id_list: a list of agentID
        :return: an Agent object
        """

        def random_goal_pos(previous_goals=None, distance=None):
            next_goal_buffer = {agentID: self.agents[agentID].next_goal for agentID in range(1, self.num_agents + 1)}
            curr_goal_buffer = {agentID: self.agents[agentID].goal_pos for agentID in range(1, self.num_agents + 1)}
            if previous_goals is None:
                previous_goals = {agentID: None for agentID in id_list}
            if distance is None:
                distance = self.goal_generate_distance
            free_for_all = np.logical_and(self.state == 0, self.goals_map == 0)
            if not all(previous_goals.values()):  # they are new born agents
                free_space = np.argwhere(free_for_all == 1)
                init_idx = np.random.choice(len(free_space), size=len(id_list), replace=False)
                new_goals = {agentID: tuple(free_space[init_idx[agentID - 1]]) for agentID in id_list}
                return new_goals
            else:
                new_goals = {}
                for agentID in id_list:
                    free_on_agents = np.logical_and(self.state > 0, self.state != agentID)
                    free_spaces_for_previous_goal = np.logical_or(free_on_agents, free_for_all)
                    # free_spaces_for_previous_goal = np.logical_and(free_spaces_for_previous_goal, self.goals_map==0)
                    if distance > 0:
                        previous_pos = previous_goals[agentID]
                        assert previous_pos is not None
                        previous_x, previous_y = previous_pos
                        x_lower_bound = (previous_x - distance) if (previous_x - distance) > 0 else 0
                        x_upper_bound = previous_x + distance + 1
                        y_lower_bound = (previous_y - distance) if (previous_x - distance) > 0 else 0
                        y_upper_bound = previous_y + distance + 1
                        free_spaces_for_previous_goal[x_lower_bound:x_upper_bound, y_lower_bound:y_upper_bound] = False
                    free_spaces_for_previous_goal = list(np.argwhere(free_spaces_for_previous_goal == 1))
                    free_spaces_for_previous_goal = [pos.tolist() for pos in free_spaces_for_previous_goal]

                    try:
                        unique = False
                        counter = 0
                        while unique == False and counter < 500:
                            init_idx = np.random.choice(len(free_spaces_for_previous_goal))
                            init_pos = free_spaces_for_previous_goal[init_idx]
                            unique = True
                            if (
                                tuple(init_pos) in next_goal_buffer.values()
                                or tuple(init_pos) in curr_goal_buffer.values()
                                or tuple(init_pos) in new_goals.values()
                            ):
                                unique = False
                            if previous_goals is not None:
                                if tuple(init_pos) in previous_goals.values():
                                    unique = False
                            counter += 1
                        if counter >= 500:
                            print("Hard to find Non Conflicting Goal")
                        new_goals.update({agentID: tuple(init_pos)})
                    except ValueError:
                        print("wrong goal")
                        self.reset_world()
                        print(self.agents[1].position)
                        self.init_agents_and_goals()
                        return None
                return new_goals

        def random_candidate_goal_pos(previous_goals=None, distance=None):
            next_goal_buffer = {agentID: self.agents[agentID].next_goal for agentID in range(1, self.num_agents + 1)}
            curr_goal_buffer = {agentID: self.agents[agentID].goal_pos for agentID in range(1, self.num_agents + 1)}
            if previous_goals is None:
                previous_goals = {agentID: None for agentID in id_list}
            if distance is None:
                distance = self.goal_generate_distance
            free_for_all = np.logical_and(self.state == 0, self.goals_map == 0, self.goal_candidate_map == 1)
            if not all(previous_goals.values()):  # they are new born agents
                free_space = np.argwhere(free_for_all == 1)
                init_idx = np.random.choice(len(free_space), size=len(id_list), replace=False)
                new_goals = {agentID: tuple(free_space[init_idx[agentID - 1]]) for agentID in id_list}
                return new_goals
            else:
                new_goals = {}
                for agentID in id_list:
                    free_on_agents = np.logical_and(self.state > 0, self.state != agentID)
                    free_spaces_for_previous_goal = np.logical_or(free_on_agents, free_for_all)
                    # free_spaces_for_previous_goal = np.logical_and(free_spaces_for_previous_goal, self.goals_map==0)
                    if distance > 0:
                        previous_pos = previous_goals[agentID]
                        assert previous_pos is not None
                        previous_x, previous_y = previous_pos
                        x_lower_bound = (previous_x - distance) if (previous_x - distance) > 0 else 0
                        x_upper_bound = previous_x + distance + 1
                        y_lower_bound = (previous_y - distance) if (previous_x - distance) > 0 else 0
                        y_upper_bound = previous_y + distance + 1
                        free_spaces_for_previous_goal[x_lower_bound:x_upper_bound, y_lower_bound:y_upper_bound] = False
                    free_spaces_for_previous_goal = list(np.argwhere(free_spaces_for_previous_goal == 1))
                    free_spaces_for_previous_goal = [pos.tolist() for pos in free_spaces_for_previous_goal]

                    try:
                        unique = False
                        counter = 0
                        while unique == False and counter < 500:
                            init_idx = np.random.choice(len(free_spaces_for_previous_goal))
                            init_pos = free_spaces_for_previous_goal[init_idx]
                            unique = True
                            if (
                                tuple(init_pos) in next_goal_buffer.values()
                                or tuple(init_pos) in curr_goal_buffer.values()
                                or tuple(init_pos) in new_goals.values()
                            ):
                                unique = False
                            if previous_goals is not None:
                                if tuple(init_pos) in previous_goals.values():
                                    unique = False
                            counter += 1
                        if counter >= 500:
                            print("Hard to find Non Conflicting Goal")
                        new_goals.update({agentID: tuple(init_pos)})
                    except ValueError:
                        print("wrong goal")
                        self.reset_world()
                        print(self.agents[1].position)
                        self.init_agents_and_goals()
                        return None
                return new_goals

        # ここから処理
        previous_goals = {agentID: self.agents[agentID].goal_pos for agentID in id_list}
        if manual_pos is not None:
            new_goals = manual_pos
        else:
            new_goals = random_goal_pos(previous_goals, distance=self.goal_generate_distance)
        if new_goals is not None:  # recursive breaker
            refresh_distance_map = False
            for agentID in id_list:
                if (
                    self.state[new_goals[agentID][0], new_goals[agentID][1]] >= 0
                    or self.state[new_goals[agentID][0], new_goals[agentID][1]] == -9
                ):
                    if self.agents[agentID].next_goal is None:  # no next_goal to use
                        # set goals_map
                        self.goals_map[new_goals[agentID][0], new_goals[agentID][1]] = agentID
                        # set agent.goal_pos
                        self.agents[agentID].goal_pos = (new_goals[agentID][0], new_goals[agentID][1])
                        # set agent.next_goal
                        new_next_goals = random_goal_pos(new_goals, distance=self.goal_generate_distance)
                        if new_next_goals is None:
                            return None
                        self.agents[agentID].next_goal = (new_next_goals[agentID][0], new_next_goals[agentID][1])
                        # remove previous goal
                        previous_goal_temp = previous_goals[agentID]
                        if previous_goal_temp is not None:
                            self.goals_map[previous_goal_temp[0], previous_goal_temp[1]] = 0
                    else:  # use next_goal as new goal
                        # set goals_map
                        next_goal_temp = self.agents[agentID].next_goal
                        assert next_goal_temp is not None
                        self.goals_map[next_goal_temp[0], next_goal_temp[1]] = agentID
                        # set agent.goal_pos
                        self.agents[agentID].goal_pos = self.agents[agentID].next_goal
                        # set agent.next_goal
                        self.agents[agentID].next_goal = (
                            new_goals[agentID][0],
                            new_goals[agentID][1],
                        )  # store new goal into next_goal
                        # remove previous goal
                        previous_goal_temp = previous_goals[agentID]
                        if previous_goal_temp is not None:
                            self.goals_map[previous_goal_temp[0], previous_goal_temp[1]] = 0
                else:
                    print(self.state)
                    print(self.goals_map)
                    raise ValueError("invalid manual_pos for goal" + str(agentID) + " at: ", str(new_goals[agentID]))
                if previous_goals[agentID] is not None:  # it has a goal!
                    if previous_goals[agentID] != self.agents[agentID].position:
                        print(self.state)
                        print(self.goals_map)
                        print(previous_goals)
                        raise RuntimeError("agent hasn't finished its goal but asking for a new goal!")

                    refresh_distance_map = True

                # compute distance map
                agent_goal = self.agents[agentID].goal_pos
                agent_next_goal = self.agents[agentID].next_goal
                assert agent_goal is not None and agent_next_goal is not None
                self.agents[agentID].next_distanceMap = getAstarDistanceMap(
                    self.state,
                    agent_goal,
                    agent_next_goal,
                    isCPython=USE_Cython_ASTAR,
                )
                if refresh_distance_map:
                    agent_pos = self.agents[agentID].position
                    agent_goal = self.agents[agentID].goal_pos
                    assert agent_pos is not None and agent_goal is not None
                    self.agents[agentID].distanceMap = getAstarDistanceMap(
                        self.state,
                        agent_pos,
                        agent_goal,
                        isCPython=USE_Cython_ASTAR,
                    )
            return 1
        else:
            return None

    def CheckCollideStatus(
        self, movement_dict: Dict[int, int], check_col=True, epsilon: float = 0
    ) -> Tuple[Dict[int, Optional[int]], Dict[int, Tuple]]:
        """衝突のチェックを行う

        Args:
            movement_dict (Dict[int, int]): エージェントが移動しようとしている方向(action)を示すDict
            check_col (bool, optional): 衝突のチェックを行うかどうか. Defaults to True.

        Returns:
            Tuple[Dict[int, Optional[int]], Dict[int, Tuple]]: エージェントごとの移動結果による状態のDictと次の位置を示したDict

        エージェントの状態:
            1: アクションが実行され、エージェントがゴール上にいる
            0: アクションが実行された
            -1: 環境との衝突（障害物、境界外）
            -2: エージェントとの衝突、すれ違い A|B->B|A
            -3: エージェントとの衝突、同じノードに移動 A| |B->AB

        WARNING: ONLY NON-DIAGONAL IS IMPLEMENTED
        return collision status and predicted next positions, do not move agent directly
        :return:
         1: action executed, and agents standing on its goal.
         0: action executed
        -1: collision with env (obstacles, out of bound)
        -2: collision with robot, swap
        -3: collision with robot, cell-wise
        """

        if self.isDiagonal is True:
            raise NotImplementedError("Diagonal is not implemented yet")
        assumed_newPos_dict: Dict[int, Tuple[int, int]] = {}  # エージェントが移動しようとしている次の位置(tuple)
        determined_newPos_dict: Dict[int, Tuple[int, int]] = {}  # エージェントの確定した次の位置(tuple)
        status_dict: Dict[int, Optional[int]] = {agentID: None for agentID in range(1, self.num_agents + 1)}
        not_determined_ids = set(range(1, self.num_agents + 1))  # まだ経路を決めていないエージェントのIDのセット

        # evaluation/testing only with continuous M*:
        # 衝突をチェックせず、そのままの位置を返す(状態に負値が含まれない)
        if not check_col:
            for agentID in range(1, self.num_agents + 1):
                assumed_direction = action2dir(movement_dict[agentID])
                assumed_newPos = tuple_plus(self.getPos(agentID), assumed_direction)
                assumed_newPos_dict.update({agentID: assumed_newPos})

            # all actions are assumed valid
            original_not_determined_ids = unpack_copy_int_set(not_determined_ids)
            for agentID in original_not_determined_ids:
                status_dict[agentID] = 1 if assumed_newPos_dict[agentID] == self.agents[agentID].goal_pos else 0
                determined_newPos_dict.update({agentID: assumed_newPos_dict[agentID]})
                not_determined_ids.remove(agentID)

            assert not not_determined_ids, "not_checked_list is not empty"

            return status_dict, determined_newPos_dict

        # detect env collision
        # 衝突をチェックする
        # NOTE:
        #   元のコードだと、複数エージェントが同じノードに向かおうとした場合、
        #   優先度が高いエージェントが移動することになっているが、
        #   その移動先に優先度が低いエージェントがおり、そのエージェントの経路計画が失敗したためにその場に留まる場合、
        #   移動する優先度の高いエージェントと経路計画に失敗した優先度の低いエージェントとの間で衝突が発生する
        #   そのため、優先度の高いエージェントはその場にエージェントがいない場合に限り移動するように変更し、
        #   移動先がない場合は反復処理をして、移動先が見つかるまで繰り返すように変更する
        #   PIBTという選択肢もあり？
        original_not_determined_ids = unpack_copy_int_set(not_determined_ids)
        for agentID in original_not_determined_ids:
            assumed_direction = action2dir(movement_dict[agentID])
            assumed_newPos = tuple_plus(self.getPos(agentID), assumed_direction)
            assumed_newPos_dict.update({agentID: assumed_newPos})
            # エージェントが留まろうとしている場合、選択の余地がないのでその場に留まることを確定する
            if assumed_newPos == self.getPos(agentID):
                status_dict[agentID] = 1 if assumed_newPos_dict[agentID] == self.agents[agentID].goal_pos else 0
                determined_newPos_dict.update({agentID: self.getPos(agentID)})
                not_determined_ids.remove(agentID)
            # 環境との衝突（障害物、境界外）がある場合、その場に留まることを確定する
            if (
                assumed_newPos[0] < 0
                or assumed_newPos[0] >= self.state.shape[0]
                or assumed_newPos[1] < 0
                or assumed_newPos[1] >= self.state.shape[1]
                or self.state[assumed_newPos] == -1
            ):
                status_dict[agentID] = -1
                determined_newPos_dict.update({agentID: self.getPos(agentID)})
                assumed_newPos_dict[agentID] = self.getPos(agentID)
                not_determined_ids.remove(agentID)

        # ループの検出を行う
        # 2エージェント間でのループはスワップであるから、衝突扱いにする
        # 3エージェント以上のループは移動できるので、全てのエージェントを移動させる
        # 最終的に、エージェントが移動したいノードにいるエージェントを繋いで木構造に帰着させたい
        original_not_determined_ids = unpack_copy_int_set(not_determined_ids)
        checked_ids: Set[int] = set()  # 確認済みのエージェントのIDのセット
        undetermined_ids: Set[int] = set()  # この段階では経路を確定させないエージェントのIDのセット
        backtrack_dict: Dict[int, List[int]] = (
            {}
        )  # そのエージェントが今いる位置に移動しようとしているエージェントのIDのリストのDict
        root_ids: Set[int] = set()  # 木構造のルートのセット
        for agentID in original_not_determined_ids:
            if agentID in checked_ids:
                # 確認済みのエージェントの場合、スキップ
                continue
            # エージェントが移動したいノードにいるエージェントが移動したいノードを取得することを繰り返し、ループを検出していく
            current_node = assumed_newPos_dict[agentID]
            trajectory_ids: List[int] = [agentID]  # この過程で調べたエージェントのIDのリスト
            while True:
                # 他エージェントがその場所にいるなら、そのエージェントのIDが返ってくる
                current_node_state: int = self.state[current_node]
                if current_node_state <= 0 or current_node_state in undetermined_ids:
                    # 移動したい先のノードにエージェントがいない場合、あるいは経路未確定のエージェントがいる場合
                    # 今のところは経路を確定させない
                    if current_node_state == 0 or current_node_state == -9:
                        # 移動したいノードにエージェントがいない場合、ルートに追加
                        root_ids.add(trajectory_ids[-1])
                    elif current_node_state > 0:
                        # 移動したいノードにエージェントがいる場合、最後のエージェントのIDをtrajectory_idsに追加
                        trajectory_ids.append(current_node_state)
                    else:
                        raise ValueError("Weird Agent: " + str(agentID))
                    for i in range(1, len(trajectory_ids)):
                        backtrack_dict[trajectory_ids[i]] = backtrack_dict.get(trajectory_ids[i], []) + [
                            trajectory_ids[i - 1]
                        ]
                    checked_ids.update(trajectory_ids)
                    undetermined_ids.update(trajectory_ids)
                    break
                elif current_node_state in determined_newPos_dict:
                    # 移動したい先のノードが既に他のエージェントの移動先として確定している場合
                    # 衝突扱いにする
                    for traj_id in trajectory_ids:
                        status_dict[traj_id] = -3
                        determined_newPos_dict.update({traj_id: self.getPos(traj_id)})
                        assumed_newPos_dict[traj_id] = self.getPos(traj_id)
                        not_determined_ids.remove(traj_id)
                    checked_ids.update(trajectory_ids)
                    break
                elif current_node_state in trajectory_ids:
                    # ループを検知
                    # ループを構成しているエージェントのIDのリスト
                    loop_agent_ids = trajectory_ids[trajectory_ids.index(current_node_state) :]
                    # ループに繋がっているが、ループを構成していないエージェントのIDのリスト
                    sub_loop_agent_ids = trajectory_ids[: trajectory_ids.index(current_node_state)]
                    if len(loop_agent_ids) == 2:
                        # 2エージェント間のループはスワップであるから、お互いを衝突扱いにする
                        # 自分を衝突扱いにし、その場に留まる
                        status_dict[loop_agent_ids[0]] = -2
                        determined_newPos_dict.update({loop_agent_ids[0]: self.getPos(loop_agent_ids[0])})
                        assumed_newPos_dict[loop_agent_ids[0]] = self.getPos(loop_agent_ids[0])
                        not_determined_ids.remove(loop_agent_ids[0])
                        # 相手を衝突扱いにし、その場に留まる
                        status_dict[loop_agent_ids[1]] = -2
                        determined_newPos_dict.update({loop_agent_ids[1]: self.getPos(loop_agent_ids[1])})
                        assumed_newPos_dict[loop_agent_ids[1]] = self.getPos(loop_agent_ids[1])
                        not_determined_ids.remove(loop_agent_ids[1])
                    elif len(loop_agent_ids) > 2:
                        # 3エージェント以上のループは移動できるので、全てのエージェントを移動させる
                        for loop_agent_id in loop_agent_ids:
                            status_dict[loop_agent_id] = (
                                1 if assumed_newPos_dict[loop_agent_id] == self.agents[loop_agent_id].goal_pos else 0
                            )
                            determined_newPos_dict.update({loop_agent_id: assumed_newPos_dict[loop_agent_id]})
                            not_determined_ids.remove(loop_agent_id)
                    else:
                        raise ValueError("Weird Agent: " + str(agentID))
                    # ループに繋がったエージェントを全て衝突として扱う
                    for sub_id in sub_loop_agent_ids:
                        status_dict[sub_id] = -3
                        determined_newPos_dict.update({sub_id: self.getPos(sub_id)})
                        assumed_newPos_dict[sub_id] = self.getPos(sub_id)
                        not_determined_ids.remove(sub_id)
                    checked_ids.update(trajectory_ids)
                    break
                else:
                    # current_nodeを更新
                    current_node = assumed_newPos_dict[current_node_state]
                    trajectory_ids.append(current_node_state)

        determined_newPos_set: Set[Tuple[int, int]] = set(determined_newPos_dict.values())
        while root_ids:
            # ルートから木構造を辿って、エージェントを移動させる
            root_id = root_ids.pop()
            root_Assumed_newPos: Tuple[int, int] = assumed_newPos_dict[root_id]
            other_roots_coming_agents = [
                other_root_id for other_root_id in root_ids if assumed_newPos_dict[other_root_id] == root_Assumed_newPos
            ]
            if root_Assumed_newPos not in determined_newPos_set and (
                len(other_roots_coming_agents) == 0 or root_id < min(other_roots_coming_agents)
            ):
                # 移動したい先のノードに他のエージェントがおらず、かつ、自分の優先度が高い場合、移動する
                status_dict[root_id] = 1 if assumed_newPos_dict[root_id] == self.agents[root_id].goal_pos else 0
                determined_newPos_dict.update({root_id: assumed_newPos_dict[root_id]})
                determined_newPos_set.add(assumed_newPos_dict[root_id])
                not_determined_ids.remove(root_id)
                if root_id in backtrack_dict:
                    # バックトラック
                    root_ids.update(backtrack_dict[root_id])
            else:
                # 他に優先して移動するエージェントがいるので、衝突扱いにする
                # バックトラックしながら、このエージェントに移動しようとしているエージェントを全て衝突扱いにする
                collided_agent_ids = {root_id}
                while collided_agent_ids:
                    collided_agent_id = collided_agent_ids.pop()
                    # 衝突扱いにする
                    status_dict[collided_agent_id] = -3
                    determined_newPos_dict.update({collided_agent_id: self.getPos(collided_agent_id)})
                    determined_newPos_set.add(self.getPos(collided_agent_id))
                    assumed_newPos_dict[collided_agent_id] = self.getPos(collided_agent_id)
                    not_determined_ids.remove(collided_agent_id)
                    if collided_agent_id in backtrack_dict:
                        collided_agent_ids.update(backtrack_dict[collided_agent_id])

        assert not not_determined_ids, "not_determined_ids is not empty" + str(not_determined_ids)
        assert len(determined_newPos_dict.keys()) == self.num_agents, "some agents are not determined"
        assert len(determined_newPos_set) == self.num_agents, "some agents are not determined"

        return status_dict, determined_newPos_dict


class TestWorld(World):
    def __init__(self, map_generator, world_info, isDiagonal=False, isConventional=False, with_ep: bool = False):
        super().__init__(map_generator, num_agents=None, isDiagonal=isDiagonal)
        if with_ep:
            (
                [self.state, self.goals_map],
                self.agents_init_pos,
                self.corridor_map,
                self.corridors,
                self.agents,
                self.ep_map,
            ) = world_info
        else:
            [self.state, self.goals_map], self.agents_init_pos, self.corridor_map, self.corridors, self.agents = (
                world_info
            )
        self.corridor_map, self.corridors = self.corridor_map[()], self.corridors[()]
        self.num_agents = len(self.agents_init_pos.keys())
        self.isConventional = isConventional
        self.with_ep = with_ep

    def reset_world(self):
        pass

    def init_agents_and_goals(self):
        pass

    def put_goals(self, id_list, manual_pos=None):
        """
        NO DISTANCE MAPS FOR MSTAR!!
        """

        def random_goal_pos(previous_goals=None, distance=None):
            next_goal_buffer = {agentID: self.agents[agentID].next_goal for agentID in range(1, self.num_agents + 1)}
            curr_goal_buffer = {agentID: self.agents[agentID].goal_pos for agentID in range(1, self.num_agents + 1)}
            if previous_goals is None:
                previous_goals = {agentID: None for agentID in id_list}
            if distance is None:
                distance = self.goal_generate_distance
            free_for_all = np.logical_and(self.state == 0, self.goals_map == 0)
            if not all(previous_goals.values()):  # they are new born agents
                free_space = np.argwhere(free_for_all == 1)
                init_idx = np.random.choice(len(free_space), size=len(id_list), replace=False)
                new_goals = {agentID: tuple(free_space[init_idx[agentID - 1]]) for agentID in id_list}
                return new_goals
            else:
                new_goals = {}
                for agentID in id_list:
                    free_on_agents = np.logical_and(self.state > 0, self.state != agentID)
                    free_spaces_for_previous_goal = np.logical_or(free_on_agents, free_for_all)
                    if distance > 0:
                        previous_pos = previous_goals[agentID]
                        assert previous_pos is not None
                        previous_x, previous_y = previous_pos
                        x_lower_bound = (previous_x - distance) if (previous_x - distance) > 0 else 0
                        x_upper_bound = previous_x + distance + 1
                        y_lower_bound = (previous_y - distance) if (previous_x - distance) > 0 else 0
                        y_upper_bound = previous_y + distance + 1
                        free_spaces_for_previous_goal[x_lower_bound:x_upper_bound, y_lower_bound:y_upper_bound] = False
                    free_spaces_for_previous_goal = list(np.argwhere(free_spaces_for_previous_goal == 1))
                    free_spaces_for_previous_goal = [pos.tolist() for pos in free_spaces_for_previous_goal]

                    try:
                        unique = False
                        counter = 0
                        while unique == False and counter < 500:
                            init_idx = np.random.choice(len(free_spaces_for_previous_goal))
                            init_pos = free_spaces_for_previous_goal[init_idx]
                            unique = True
                            if (
                                tuple(init_pos) in next_goal_buffer.values()
                                or tuple(init_pos) in curr_goal_buffer.values()
                                or tuple(init_pos) in new_goals.values()
                            ):
                                unique = False
                            if previous_goals is not None:
                                if tuple(init_pos) in previous_goals.values():
                                    unique = False
                            counter += 1
                        if counter >= 500:
                            print("Hard to find Non Conflicting Goal")
                        new_goals.update({agentID: tuple(init_pos)})
                    except ValueError:
                        print("wrong goal")
                        self.reset_world()
                        print(self.agents[1].position)
                        self.init_agents_and_goals()
                        return None
                return new_goals

        def random_goal_pos_with_ep(previous_goals=None, distance=None, ep_map=None):
            next_goal_buffer = {agentID: self.agents[agentID].next_goal for agentID in range(1, self.num_agents + 1)}
            curr_goal_buffer = {agentID: self.agents[agentID].goal_pos for agentID in range(1, self.num_agents + 1)}
            if previous_goals is None:
                previous_goals = {agentID: None for agentID in id_list}
            if distance is None:
                distance = self.goal_generate_distance
            # self.stateが0で、goals_mapが0で、ep_mapが1の場所をTrueとする
            free_for_all = np.logical_and(self.state == 0, self.goals_map == 0)
            free_for_all = np.logical_and(free_for_all, ep_map == 1)
            if not all(previous_goals.values()):  # they are new born agents
                free_space = np.argwhere(free_for_all == 1)
                init_idx = np.random.choice(len(free_space), size=len(id_list), replace=False)
                new_goals = {agentID: tuple(free_space[init_idx[agentID - 1]]) for agentID in id_list}
                return new_goals
            else:
                new_goals = {}
                for agentID in id_list:
                    free_on_agents = np.logical_and(self.state > 0, self.state != agentID)
                    free_on_agents = np.logical_and(free_on_agents, ep_map == 1)
                    free_spaces_for_previous_goal = np.logical_or(free_on_agents, free_for_all)
                    if distance > 0:
                        previous_pos = previous_goals[agentID]
                        assert previous_pos is not None
                        previous_x, previous_y = previous_pos
                        x_lower_bound = (previous_x - distance) if (previous_x - distance) > 0 else 0
                        x_upper_bound = previous_x + distance + 1
                        y_lower_bound = (previous_y - distance) if (previous_x - distance) > 0 else 0
                        y_upper_bound = previous_y + distance + 1
                        free_spaces_for_previous_goal[x_lower_bound:x_upper_bound, y_lower_bound:y_upper_bound] = False
                    free_spaces_for_previous_goal = list(np.argwhere(free_spaces_for_previous_goal == 1))
                    free_spaces_for_previous_goal = [pos.tolist() for pos in free_spaces_for_previous_goal]

                    try:
                        unique = False
                        counter = 0
                        while unique == False and counter < 500:
                            init_idx = np.random.choice(len(free_spaces_for_previous_goal))
                            init_pos = free_spaces_for_previous_goal[init_idx]
                            unique = True
                            if (
                                tuple(init_pos) in next_goal_buffer.values()
                                or tuple(init_pos) in curr_goal_buffer.values()
                                or tuple(init_pos) in new_goals.values()
                            ):
                                unique = False
                            if previous_goals is not None:
                                if tuple(init_pos) in previous_goals.values():
                                    unique = False
                            counter += 1
                        if counter >= 500:
                            print("Hard to find Non Conflicting Goal")
                        new_goals.update({agentID: tuple(init_pos)})
                    except ValueError:
                        print("wrong goal")
                        self.reset_world()
                        print(self.agents[1].position)
                        self.init_agents_and_goals()
                        return None
                return new_goals

        previous_goals = {agentID: self.agents[agentID].goal_pos for agentID in id_list}
        if manual_pos is not None:
            print("manual_pos")
            new_goals = manual_pos
        elif self.with_ep:
            new_goals = random_goal_pos_with_ep(
                previous_goals, distance=self.goal_generate_distance, ep_map=self.ep_map
            )
        else:
            new_goals = random_goal_pos(previous_goals, distance=self.goal_generate_distance)
        if new_goals is not None:  # recursive breaker
            refresh_distance_map = False
            for agentID in id_list:
                if (
                    self.state[new_goals[agentID][0], new_goals[agentID][1]] >= 0
                    and self.ep_map[new_goals[agentID][0], new_goals[agentID][1]] == 1
                ):
                    if self.agents[agentID].next_goal is None:  # no next_goal to use
                        # set goals_map
                        self.goals_map[new_goals[agentID][0], new_goals[agentID][1]] = agentID
                        # set agent.goal_pos
                        self.agents[agentID].goal_pos = (new_goals[agentID][0], new_goals[agentID][1])
                        # set agent.next_goal
                        if self.with_ep:
                            new_next_goals = random_goal_pos_with_ep(
                                new_goals, distance=self.goal_generate_distance, ep_map=self.ep_map
                            )
                        else:
                            new_next_goals = random_goal_pos(new_goals, distance=self.goal_generate_distance)
                        if new_next_goals is None:
                            return None
                        self.agents[agentID].next_goal = (new_next_goals[agentID][0], new_next_goals[agentID][1])
                        # remove previous goal
                        if previous_goals[agentID] is not None:
                            self.goals_map[previous_goals[agentID][0], previous_goals[agentID][1]] = 0
                    else:  # use next_goal as new goal
                        # set goals_map
                        self.goals_map[self.agents[agentID].next_goal[0], self.agents[agentID].next_goal[1]] = agentID
                        # set agent.goal_pos
                        self.agents[agentID].goal_pos = self.agents[agentID].next_goal
                        # set agent.next_goal
                        self.agents[agentID].next_goal = (
                            new_goals[agentID][0],
                            new_goals[agentID][1],
                        )  # store new goal into next_goal
                        # remove previous goal
                        if previous_goals[agentID] is not None:
                            self.goals_map[previous_goals[agentID][0], previous_goals[agentID][1]] = 0
                else:
                    print(self.state)
                    print(self.goals_map)
                    raise ValueError("invalid manual_pos for goal" + str(agentID) + " at: ", str(new_goals[agentID]))
                if previous_goals[agentID] is not None:  # it has a goal!
                    if previous_goals[agentID] != self.agents[agentID].position:
                        print(self.state)
                        print(self.goals_map)
                        print(previous_goals)
                        raise RuntimeError("agent hasn't finished its goal but asking for a new goal!")

                    refresh_distance_map = True

                # compute distance map
                if not self.isConventional:
                    self.agents[agentID].next_distanceMap = getAstarDistanceMap(
                        self.state,
                        self.agents[agentID].goal_pos,
                        self.agents[agentID].next_goal,
                        isCPython=USE_Cython_ASTAR,
                    )
                    if refresh_distance_map:
                        self.agents[agentID].distanceMap = getAstarDistanceMap(
                            self.state,
                            self.agents[agentID].position,
                            self.agents[agentID].goal_pos,
                            isCPython=USE_Cython_ASTAR,
                        )
            return 1
        else:
            return None


class MAPFEnv(gym.Env):
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(
        self,
        observer,
        map_generator,
        num_agents,
        IsDiagonal=False,
        frozen_steps=0,
        isOneShot=False,
        should_use_cpp_mstar=True,
    ):
        self.observer = observer
        self.map_generator = map_generator
        self.viewer = None

        self.should_use_cpp_mstar = should_use_cpp_mstar

        self.isOneShot = isOneShot
        self.frozen_steps = frozen_steps
        self.num_agents = num_agents
        self.IsDiagonal = IsDiagonal
        self.set_world()
        self.obs_size = self.observer.get_obs_size()
        self.isStandingOnGoal = {i: False for i in range(1, self.num_agents + 1)}

        self.individual_rewards = {i: 0.0 for i in range(1, self.num_agents + 1)}
        self.mutex = Lock()
        self.GIF_frame = []
        if IsDiagonal:
            self.action_space = spaces.Tuple([spaces.Discrete(self.num_agents), spaces.Discrete(9)])
        else:
            self.action_space = spaces.Tuple([spaces.Discrete(self.num_agents), spaces.Discrete(5)])

        # 報酬値の設定
        self.ACTION_COST = -0.3
        self.GOAL_REWARD = None  # 0.4*map_size
        # self.ENV_COLLISION_REWARD = -15.0
        self.COLLISION_REWARD = -3.0

    def getObstacleMap(self):
        return (self.world.state == -1).astype(int)

    def getMapSize(self) -> int:
        return self.world.state.shape[0]

    def getGoals(self) -> Dict[int, Optional[Tuple[int, int]]]:
        return {i: self.world.agents[i].goal_pos for i in range(1, self.num_agents + 1)}

    def getStatus(self) -> Dict[int, Optional[int]]:
        return {i: self.world.agents[i].status for i in range(1, self.num_agents + 1)}

    def getPositions(self) -> Dict[int, Optional[Tuple[int, int]]]:
        return {i: self.world.agents[i].position for i in range(1, self.num_agents + 1)}

    def getLastMovements(self) -> Dict[int, Tuple[int, int]]:
        return {i: self.world.agents[i].position_history[-1] for i in range(1, self.num_agents + 1)}

    def getDone(self):
        if self.isOneShot:
            return (
                sum([self.world.getDone(agentID) > 0 for agentID in range(1, self.num_agents + 1)]) == self.num_agents
            )
        else:
            raise NotImplementedError("there is no definition of getDone in continuous world")

    def set_world(self):
        self.world = World(self.map_generator, num_agents=self.num_agents, isDiagonal=self.IsDiagonal)
        self.num_agents = self.world.num_agents
        self.observer.set_env(self.world)

    def _reset(self, *args, **kwargs):
        raise NotImplementedError

    def isInCorridor(self, agentID):
        """
        :param agentID: start from 1 not 0!
        :return: isIn: bool, corridor_ID: int
        """
        agent_pos = self.world.getPos(agentID)
        if self.world.corridor_map[(agent_pos[0], agent_pos[1])][1] in [-1, 2]:
            return False, None
        else:
            return True, self.world.corridor_map[(agent_pos[0], agent_pos[1])][0]

    def _observe(self, handles=None):
        """
        Returns Dict of observation {agentid:[], ...}
        """
        if handles is None:
            self.obs_dict = self.observer.get_many(list(range(1, self.num_agents + 1)))
        elif handles in list(range(1, self.num_agents + 1)):
            self.obs_dict = self.observer.get_many([handles])
        elif set(handles) == set(handles) & set(list(range(1, self.num_agents + 1))):
            self.obs_dict = self.observer.get_many(handles)
        else:
            raise ValueError("Invalid agent_id given")
        return self.obs_dict

    def step_all(self, movement_dict, observe=True, check_col=True) -> Tuple:
        """
        Agents are forced to freeze self.frozen_steps steps if they are standing on their goals.
        The new goal will be generated at the FIRST step it remains on its goal.

        :param movement_dict: {agentID_starting_from_1: action:int 0-4, ...}
                              unmentioned agent will be considered as taking standing still
        :return: obs_of_all:dict, reward_of_single_step:dict
        """

        for agentID in range(1, self.num_agents + 1):
            if self.world.agents[agentID].freeze > self.frozen_steps:  # set frozen agents free
                self.world.agents[agentID].freeze = 0

            if agentID not in movement_dict.keys() or self.world.agents[agentID].freeze:
                movement_dict.update({agentID: 0})
            else:
                assert (
                    movement_dict[agentID] in list(range(5)) if self.IsDiagonal else list(range(9))
                ), "action not in action space"

        status_dict, newPos_dict = self.world.CheckCollideStatus(movement_dict, check_col=check_col)
        # マップ上のエージェントを一旦全部削除
        self.world.state[self.world.state > 0] = 0  # remove agents in the map
        put_goal_list: List[int] = []
        freeze_list = []
        for agentID in range(1, self.num_agents + 1):
            if self.isOneShot and self.world.getDone(agentID) > 0:
                continue
            # エージェントの位置を更新
            newPos = newPos_dict[agentID]
            self.world.state[newPos] = agentID
            self.world.agents[agentID].move(newPos, status_dict[agentID])
            self.give_moving_reward(agentID)
            if status_dict[agentID] == 1:
                # エージェントがゴールに到達した場合
                if not self.isOneShot:
                    if self.world.agents[agentID].freeze == 0:
                        # ゴールに今たどり着いたエージェントをリストに追加
                        put_goal_list.append(agentID)
                    if self.world.agents[agentID].action_history[-1] == 0:  # standing still on goal
                        # ゴール上で立ち止まっているエージェントをリストに追加
                        freeze_list.append(agentID)
                    self.world.agents[agentID].freeze += 1
                else:
                    raise NotImplementedError("OneShot is not implemented yet")
                    self.world.agents[agentID].status = 2
                    self.world.state[newPos] = 0
                    self.world.goals_map[newPos] = 0
        free_agents = list(range(1, self.num_agents + 1))

        if put_goal_list and not self.isOneShot:
            self.world.put_goals(put_goal_list)

            # remove obs for frozen agents:

            for frozen_agent in freeze_list:
                free_agents.remove(frozen_agent)
        if observe:
            return self._observe(free_agents), self.individual_rewards
        elif check_col:
            return None, self.individual_rewards
        else:
            raise NotImplementedError

    def give_moving_reward(self, agentID):
        raise NotImplementedError

    def listValidActions(self, agent_ID, agent_obs):
        raise NotImplementedError

    def expert_until_first_goal(self, inflation=2.0, time_limit=60.0):
        world = self.getObstacleMap()
        start_positions = []
        goals = []
        start_positions_dir = self.getPositions()
        goals_dir = self.getGoals()
        for i in range(1, self.world.num_agents + 1):
            start_positions.append(start_positions_dir[i])
            goals.append(goals_dir[i])
        mstar_path = None
        start_time = time.perf_counter()
        if self.should_use_cpp_mstar:
            try:
                mstar_path = cpp_mstar.find_path(world, start_positions, goals, inflation, time_limit / 5.0)
            except OutOfTimeError:
                # M* timed out
                print("timeout(cpp_mstar)")
                print("World", world)
                print("Start Pos", start_positions)
                print("Goals", goals)
                print("time", time.perf_counter() - start_time)
            except NoSolutionError:
                print("nosol????")
                print("World", world)
                print("Start Pos", start_positions)
                print("Goals", goals)
            except:
                print("Unknown bug?!")
                print("raise Exception(cpp_mstar)")
        else:
            try:
                mstar_path = od_mstar.find_path(
                    world, start_positions, goals, inflation=inflation, time_limit=time_limit
                )
            except OutOfTimeError:
                # M* timed out
                print("timeout(od_mstar)")
                print("World", world)
                print("Start Pos", start_positions)
                print("Goals", goals)
                print("time", time.perf_counter() - start_time)
            except NoSolutionError:
                print("nosol????")
                print("World", world)
                print("Start Pos", start_positions)
                print("Goals", goals)
            except:
                print("Unknown bug?!")
                print("raise Exception(od_mstar)")
        return mstar_path

    def _add_rendering_entry(self, entry, permanent=False):
        assert self.viewer is not None
        if permanent:
            self.viewer.add_geom(entry)
        else:
            self.viewer.add_onetime(entry)

    def _render(self, mode="human", close=False, screen_width=800, screen_height=800):
        def painter(state_map, agents_dict, goals_dict):
            def initColors(num_agents):
                c = {a + 1: hsv_to_rgb(np.array([a / float(num_agents), 1, 1])) for a in range(num_agents)}
                return c

            def create_rectangle(x, y, width, height, fill):
                ps = [(x, y), ((x + width), y), ((x + width), (y + height)), (x, (y + height))]
                rect = rendering.FilledPolygon(ps)
                rect.set_color(fill[0], fill[1], fill[2])
                rect.add_attr(rendering.Transform())
                return rect

            def drawStar(centerX, centerY, diameter, numPoints, color):
                entry_list = []
                outerRad = diameter // 2
                innerRad = int(outerRad * 3 / 8)
                # fill the center of the star
                angleBetween = 2 * math.pi / numPoints  # angle between star points in radians
                for i in range(numPoints):
                    # p1 and p3 are on the inner radius, and p2 is the point
                    pointAngle = math.pi / 2 + i * angleBetween
                    p1X = centerX + innerRad * math.cos(pointAngle - angleBetween / 2)
                    p1Y = centerY - innerRad * math.sin(pointAngle - angleBetween / 2)
                    p2X = centerX + outerRad * math.cos(pointAngle)
                    p2Y = centerY - outerRad * math.sin(pointAngle)
                    p3X = centerX + innerRad * math.cos(pointAngle + angleBetween / 2)
                    p3Y = centerY - innerRad * math.sin(pointAngle + angleBetween / 2)
                    # draw the triangle for each tip.
                    poly = rendering.FilledPolygon([(p1X, p1Y), (p2X, p2Y), (p3X, p3Y)])
                    poly.set_color(color[0], color[1], color[2])
                    poly.add_attr(rendering.Transform())
                    entry_list.append(poly)
                return entry_list

            def create_circle(x, y, diameter, world_size, fill, resolution=20):
                c = (x + world_size / 2, y + world_size / 2)
                dr = math.pi * 2 / resolution
                ps = []
                for i in range(resolution):
                    x = c[0] + math.cos(i * dr) * diameter / 2
                    y = c[1] + math.sin(i * dr) * diameter / 2
                    ps.append((x, y))
                circ = rendering.FilledPolygon(ps)
                circ.set_color(fill[0], fill[1], fill[2])
                circ.add_attr(rendering.Transform())
                return circ

            assert len(goals_dict) == len(agents_dict)
            num_agents = len(goals_dict)
            world_shape = state_map.shape
            world_size = screen_width / max(*world_shape)
            colors = initColors(num_agents)
            if self.viewer is None:
                self.viewer = rendering.Viewer(screen_width, screen_height)
                rect = create_rectangle(0, 0, screen_width, screen_height, (0.6, 0.6, 0.6))
                self._add_rendering_entry(rect, permanent=True)
                for i in range(world_shape[0]):
                    start = 0
                    end = 1
                    scanning = False
                    write = False
                    for j in range(world_shape[1]):
                        if state_map[i, j] != -1 and not scanning:  # free
                            start = j
                            scanning = True
                        if (j == world_shape[1] - 1 or state_map[i, j] == -1) and scanning:
                            end = j + 1 if j == world_shape[1] - 1 else j
                            scanning = False
                            write = True
                        if write:
                            x = i * world_size
                            y = start * world_size
                            rect = create_rectangle(x, y, world_size, world_size * (end - start), (1, 1, 1))
                            self._add_rendering_entry(rect, permanent=True)
                            write = False
            for agent in range(1, num_agents + 1):
                i, j = agents_dict[agent]
                x = i * world_size
                y = j * world_size
                color = colors[state_map[i, j]]
                rect = create_rectangle(x, y, world_size, world_size, color)
                self._add_rendering_entry(rect)

                i, j = goals_dict[agent]
                x = i * world_size
                y = j * world_size
                color = colors[agent]
                circ = create_circle(x, y, world_size, world_size, color)
                self._add_rendering_entry(circ)
                if agents_dict[agent][0] == goals_dict[agent][0] and agents_dict[agent][1] == goals_dict[agent][1]:
                    color = (0, 0, 0)
                    circ = create_circle(x, y, world_size, world_size, color)
                    self._add_rendering_entry(circ)
            assert self.viewer is not None
            result = self.viewer.render(return_rgb_array=True)
            return result

        frame = painter(self.world.state, self.getPositions(), self.getGoals())
        return frame

