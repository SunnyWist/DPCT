import numpy as np
import copy
import time
from typing import List, Dict, Tuple, Union

from Observer_Builder import ObservationBuilder
from Env_Builder import *


# 提案手法1: 視野内のエージェントのそのまた視野内のエージェントから現在地と目的地の情報を得る
# 諸々のMapのサイズをobs_size * obs_sizeから、(obs_size * 2 - 1) * (obs_size * 2 - 1)に変更
# 見えない部分は、pathlength_mapは-1で、それ以外は0で埋める
class PRIMAL2_Observer(ObservationBuilder):
    """
    obs shape: (9 + num_future_steps * (obs_size * 2 - 1) * (obs_size * 2 - 1) )
    map order: poss_map, goal_map, goals_map, obs_map, pathlength_map, blocking_map, deltax_map, deltay_map, visible_map ,astar maps
    """

    def __init__(
        self,
        observation_size: int = 11,
        num_future_steps: int = 3,
        printTime: bool = False,
        small_observation_size: int = 7,
        parameter_kinds: int = 9,
    ):
        super(PRIMAL2_Observer, self).__init__()
        self.small_size = small_observation_size  # unused
        # 基本的な視野の一辺の長さ、環境、エージェントともに観測可能な範囲
        self.observation_size = observation_size
        # 最大視野、基本的な視野内にいるエージェントの視野から観測できる最大の範囲
        self.expanded_size = self.observation_size * 2 - 1
        self.num_future_steps = num_future_steps
        self.NUM_CHANNELS = parameter_kinds + self.num_future_steps
        self.printTime = printTime

    def set_world(self, world: World):
        super().set_env(world)

    def get_next_positions(self, agent_id: int) -> List[List[int]]:
        """エージェントの次の次の位置を取得する

        Args:
            agent_id (int): エージェントID

        Returns:
            List[List[int]]: エージェントの次の次の位置のリスト
        """
        agent_pos = self.world.getPos(agent_id)
        positions: List[List[int]] = []
        current_pos = [agent_pos[0], agent_pos[1]]
        next_positions = self.world.blank_env_valid_neighbor(current_pos[0], current_pos[1])
        for position in next_positions:
            if position is not None and position != agent_pos:
                positions.append([position[0], position[1]])
                next_next_positions = self.world.blank_env_valid_neighbor(position[0], position[1])
                for pos in next_next_positions:
                    if pos is not None and pos not in positions and pos != agent_pos:
                        positions.append([pos[0], pos[1]])
        return positions

    def _get(self, agent_id: int, all_astar_maps: np.ndarray) -> Tuple[np.ndarray, List[float], np.ndarray]:
        start_time = time.time()

        assert agent_id > 0
        agent_pos = self.world.getPos(agent_id)
        top_left = (
            agent_pos[0] - self.observation_size // 2,
            agent_pos[1] - self.observation_size // 2,
        )
        bottom_right = (
            top_left[0] + self.observation_size,
            top_left[1] + self.observation_size,
        )
        centre = (self.observation_size - 1) / 2
        obs_shape = (self.observation_size, self.observation_size)

        # 視野内の他のエージェントの視野から観測できる範囲(expanded_size * expanded_size)
        expanded_top_left = (
            agent_pos[0] - self.expanded_size // 2,
            agent_pos[1] - self.expanded_size // 2,
        )
        expanded_bottom_right = (
            expanded_top_left[0] + self.expanded_size,
            expanded_top_left[1] + self.expanded_size,
        )
        expanded_centre = (self.expanded_size - 1) / 2
        expanded_obs_shape = (self.expanded_size, self.expanded_size)  # マップのサイズ使用するマップのサイズを定義

        # 最大視野に合わせて行列サイズを定義

        # 4つのバイナリマップ(PRIMAL2では)
        # 自分のゴール位置がどこかを示すマップ (1: ゴール位置, 0: それ以外)
        goal_map = np.zeros(expanded_obs_shape)
        # 他のエージェントの位置がどこかを示すマップ (1: エージェントの位置, 0: それ以外)
        poss_map = np.zeros(expanded_obs_shape)
        # 他のエージェントのゴール位置がどこかを示すマップ (1: ゴール位置, 0: それ以外)
        goals_map = np.zeros(expanded_obs_shape)
        # 障害物の位置がどこかを示すマップ (1: 障害物の位置, 0: それ以外)
        obs_map = np.zeros(expanded_obs_shape)

        # 3ステップ分のA*による他エージェントの予測位置を示すマップ
        # A*による視野内のエージェントの3ステップ分の予測位置を示すマップ (>0: 予測されるエージェントの数, 0: それ以外)
        astar_map = np.zeros(
            [
                self.num_future_steps,
                self.expanded_size,
                self.expanded_size,
            ]
        )
        # A*による全エージェントの3ステップ分の予測位置を示すマップ (>0: 予測されるエージェントの数, 0: それ以外)
        astar_map_unpadded = np.zeros(
            [
                self.num_future_steps,
                self.world.state.shape[0],
                self.world.state.shape[1],
            ]
        )

        # 自分のゴール位置までの距離(A*)を示すマップ
        # 各マスのゴールまでの距離(A*)を示すマップ (-1: 障害物or観測不可能エリア, >0: ゴールまでの距離)
        pathlength_map = np.zeros(expanded_obs_shape)

        # コリドーに関する3つのマップ
        # コリドーのx方向の変位を示すマップ (>0: 変位量, 0: エンドポイントではない)
        deltax_map = np.zeros(expanded_obs_shape)
        # コリドーのy方向の変位を示すマップ (>0: 変位量, 0: エンドポイントではない)
        deltay_map = np.zeros(expanded_obs_shape)
        # コリドーでこちら側に向かってくるエージェントがいるかどうかを示すマップ (1: エンドポイントかつ向かってくるエージェントがいる, 0: それ以外)
        blocking_map = np.zeros(expanded_obs_shape)

        # 提案手法1で追加する視野制限に関するマップ
        # そのマスの環境が観測可能かを示すマップ (1: 観測可能, 0: 観測不可能)
        visible_map = np.zeros(expanded_obs_shape)
        # 近隣エージェントによって拡張された視野内の他のエージェントのゴールのマップ (1: ゴール位置)
        expanded_goals_map = np.zeros(expanded_obs_shape)

        time1 = time.time() - start_time
        start_time = time.time()

        # concatenate all_astar maps
        other_agents = list(range(self.world.num_agents))  # needs to be 0-indexed for numpy magic below
        other_agents.remove(agent_id - 1)  # 0-indexing again
        astar_map_unpadded = np.zeros(
            [
                self.num_future_steps,
                self.world.state.shape[0],
                self.world.state.shape[1],
            ]
        )
        astar_map_unpadded[
            : self.num_future_steps,
            max(0, expanded_top_left[0]) : min(expanded_bottom_right[0], self.world.state.shape[0]),
            max(0, expanded_top_left[1]) : min(expanded_bottom_right[1], self.world.state.shape[1]),
        ] = np.sum(
            all_astar_maps[
                other_agents,
                : self.num_future_steps,
                max(0, expanded_top_left[0]) : min(expanded_bottom_right[0], self.world.state.shape[0]),
                max(0, expanded_top_left[1]) : min(expanded_bottom_right[1], self.world.state.shape[1]),
            ],
            axis=0,
        )

        time2 = time.time() - start_time
        start_time = time.time()

        # original layers from PRIMAL1
        # 各マップの状況をリストに埋めていく
        # 見えない部分は、obs_map, pathlength_mapは-1で、それ以外は0で埋める
        # まず自分の本来の視野内にあるエージェントを求めて、そのエージェントの視野をvisible_mapに記録する

        # 自力で見える部分の処理
        own_visible_agents = []
        for i in range(top_left[0], top_left[0] + self.observation_size):
            for j in range(top_left[1], top_left[1] + self.observation_size):
                if i >= self.world.state.shape[0] or i < 0 or j >= self.world.state.shape[1] or j < 0:
                    # out of bounds, just treat as an obstacle
                    # マップ外
                    continue
                visible_map[i - expanded_top_left[0], j - expanded_top_left[1]] = 1
                if self.world.state[i, j] > 0 and self.world.state[i, j] != agent_id:
                    # other agents' positions
                    own_visible_agents.append(self.world.state[i, j])
        # 視野内にいるエージェントを通して見える部分の処理
        for agent in own_visible_agents:
            x, y = self.world.getPos(agent)
            for i in range(x - self.observation_size // 2, x + self.observation_size // 2 + 1):
                for j in range(y - self.observation_size // 2, y + self.observation_size // 2 + 1):
                    if i >= self.world.state.shape[0] or i < 0 or j >= self.world.state.shape[1] or j < 0:
                        # out of bounds, just treat as an obstacle
                        # マップ外
                        continue
                    visible_map[i - expanded_top_left[0], j - expanded_top_left[1]] = 1

        visible_agents = []
        # 見える部分に関して、各マップの状況をリストに埋めていく
        for i in range(expanded_top_left[0], expanded_top_left[0] + self.expanded_size):
            for j in range(expanded_top_left[1], expanded_top_left[1] + self.expanded_size):
                if i >= self.world.state.shape[0] or i < 0 or j >= self.world.state.shape[1] or j < 0:
                    # out of bounds, just treat as an obstacle
                    # マップ外のため、障害物として扱う
                    obs_map[i - expanded_top_left[0], j - expanded_top_left[1]] = 1
                    pathlength_map[i - expanded_top_left[0], j - expanded_top_left[1]] = -1
                    continue
                if visible_map[i - expanded_top_left[0], j - expanded_top_left[1]] == 0:
                    # 見えない範囲のため、pathlength_mapは-1で、それ以外は0で埋める
                    # obs_map[i - expanded_top_left[0], j - expanded_top_left[1]] = -1
                    pathlength_map[i - expanded_top_left[0], j - expanded_top_left[1]] = -1
                    continue

                astar_map[: self.num_future_steps, i - expanded_top_left[0], j - expanded_top_left[1]] = (
                    astar_map_unpadded[: self.num_future_steps, i, j]
                )
                if self.world.state[i, j] == -1:
                    # obstacles
                    obs_map[i - expanded_top_left[0], j - expanded_top_left[1]] = 1
                if self.world.state[i, j] == agent_id:
                    # agent's position
                    poss_map[i - expanded_top_left[0], j - expanded_top_left[1]] = 1
                    # updated_poss_map[i - top_left[0], j - top_left[1]] = 0
                if self.world.goals_map[i, j] == agent_id:
                    # agent's goal
                    goal_map[i - expanded_top_left[0], j - expanded_top_left[1]] = 1
                if self.world.state[i, j] > 0 and self.world.state[i, j] != agent_id:
                    # other agents' positions
                    visible_agents.append(self.world.state[i, j])
                    poss_map[i - expanded_top_left[0], j - expanded_top_left[1]] = 1
                    # updated_poss_map[i - top_left[0], j - top_left[1]] = self.world.state[i, j]

                # we can keep this map even if on goal,
                # since observation is computed after the refresh of new distance map
                my_distance_map = self.world.agents[agent_id].distanceMap
                if my_distance_map is None:
                    raise ValueError("Distance map is not computed.")
                pathlength_map[i - expanded_top_left[0], j - expanded_top_left[1]] = my_distance_map[i, j]

        time3 = time.time() - start_time
        start_time = time.time()

        for agent in visible_agents:
            x, y = self.world.getGoal(agent)
            if (
                expanded_top_left[0] <= x < expanded_bottom_right[0]
                and expanded_top_left[1] <= y < expanded_bottom_right[1]
            ):
                # ゴールがマップ内で、
                if visible_map[x - expanded_top_left[0], y - expanded_top_left[1]] == 1:
                    # ゴールが見える場合は、expanded_goals_mapのゴール位置の場所に1を埋める
                    expanded_goals_map[x - expanded_top_left[0], y - expanded_top_left[1]] = 1
            # goals_mapのエージェント本来の視野内のゴール位置の場所に最も近いところに1を埋める
            min_node = (
                max(top_left[0], min(top_left[0] + self.observation_size - 1, x)),
                max(top_left[1], min(top_left[1] + self.observation_size - 1, y)),
            )
            goals_map[min_node[0] - top_left[0], min_node[1] - top_left[1]] = 1

        dx = self.world.getGoal(agent_id)[0] - agent_pos[0]
        dy = self.world.getGoal(agent_id)[1] - agent_pos[1]
        mag = (dx**2 + dy**2) ** 0.5
        if mag != 0:
            dx = dx / mag
            dy = dy / mag
        if mag > 60:
            mag = 60

        time4 = time.time() - start_time
        start_time = time.time()

        current_corridor_id = -1
        current_corridor = self.world.corridor_map[self.world.getPos(agent_id)[0], self.world.getPos(agent_id)[1]][1]
        if current_corridor == 1:
            current_corridor_id = self.world.corridor_map[
                self.world.getPos(agent_id)[0], self.world.getPos(agent_id)[1]
            ][0]

        positions = self.get_next_positions(agent_id)
        for position in positions:
            cell_info = self.world.corridor_map[position[0], position[1]]
            if cell_info[1] == 1:
                corridor_id = cell_info[0]
                if corridor_id != current_corridor_id:
                    if len(self.world.corridors[corridor_id]["EndPoints"]) == 1:
                        if [position[0], position[1]] == self.world.corridors[corridor_id]["StoppingPoints"][0]:
                            blocking_map[position[0] - expanded_top_left[0], position[1] - expanded_top_left[1]] = (
                                self.get_blocking(corridor_id, 0, agent_id, 1)
                            )
                    elif [position[0], position[1]] == self.world.corridors[corridor_id]["StoppingPoints"][0]:
                        end_point_pos = self.world.corridors[corridor_id]["EndPoints"][0]
                        deltax_map[
                            position[0] - expanded_top_left[0], position[1] - expanded_top_left[1]
                        ] = self.world.corridors[corridor_id]["DeltaX"][
                            (end_point_pos[0], end_point_pos[1])
                        ]  # / max(mag, 1)
                        deltay_map[
                            position[0] - expanded_top_left[0], position[1] - expanded_top_left[1]
                        ] = self.world.corridors[corridor_id]["DeltaY"][
                            (end_point_pos[0], end_point_pos[1])
                        ]  # / max(mag, 1)
                        blocking_map[position[0] - expanded_top_left[0], position[1] - expanded_top_left[1]] = (
                            self.get_blocking(corridor_id, 0, agent_id, 2)
                        )
                    elif [position[0], position[1]] == self.world.corridors[corridor_id]["StoppingPoints"][1]:
                        end_point_pos = self.world.corridors[corridor_id]["EndPoints"][1]
                        deltax_map[
                            position[0] - expanded_top_left[0], position[1] - expanded_top_left[1]
                        ] = self.world.corridors[corridor_id]["DeltaX"][
                            (end_point_pos[0], end_point_pos[1])
                        ]  # / max(mag, 1)
                        deltay_map[
                            position[0] - expanded_top_left[0], position[1] - expanded_top_left[1]
                        ] = self.world.corridors[corridor_id]["DeltaY"][
                            (end_point_pos[0], end_point_pos[1])
                        ]  # / max(mag, 1)
                        blocking_map[position[0] - expanded_top_left[0], position[1] - expanded_top_left[1]] = (
                            self.get_blocking(corridor_id, 1, agent_id, 2)
                        )
                    else:
                        pass

        time5 = time.time() - start_time
        start_time = time.time()

        # pathlength_mapを正規化
        free_spaces = list(np.argwhere(pathlength_map > 0))
        distance_list = []
        for arg in free_spaces:
            dist = pathlength_map[arg[0], arg[1]]
            if dist not in distance_list:
                distance_list.append(dist)
        distance_list.sort()
        step_size = 1 / len(distance_list)
        for i in range(self.expanded_size):
            for j in range(self.expanded_size):
                dist_mag = pathlength_map[i, j]
                if dist_mag > 0:
                    index = distance_list.index(dist_mag)
                    pathlength_map[i, j] = (index + 1) * step_size

        state = np.array(
            [
                poss_map,
                goal_map,
                goals_map,
                obs_map,
                pathlength_map,
                blocking_map,
                deltax_map,
                deltay_map,
                visible_map,
            ]
        )
        state = np.concatenate((state, astar_map), axis=0)

        time6 = time.time() - start_time
        start_time = time.time()

        if type(state) is not np.ndarray:
            raise ValueError("State is not np.ndarray. State is of type: " + str(type(state)))
        return (
            state,
            [dx, dy, mag],
            np.array([time1, time2, time3, time4, time5, time6]),
        )

    def get_many(self, handles: Union[List[int], None] = None) -> Dict[int, List[Union[np.ndarray, List[float]]]]:
        """各エージェントの観測を取得する

        Args:
            handles (Union[List[int], None], optional): 観測を取得するエージェントのIDのリスト. Defaults to None.

        Returns:
            Dict[int, List[Union[np.ndarray, List[float]]]]: 各エージェントの観測の辞書
        """
        observations: Dict[int, List[Union[np.ndarray, List[float]]]] = {}
        all_astar_maps = self.get_astar_map()
        if handles is None:
            handles = list(range(1, self.world.num_agents + 1))

        times = np.zeros((1, 6))

        for h in handles:
            state, vector, time = self._get(h, all_astar_maps)
            observations[h] = [state, vector]
            times += time
        if self.printTime:
            print(times)
        return observations

    def get_astar_map(self) -> np.ndarray:
        """各エージェントに対する、向こう3ステップ分のA*による最短経路を使った予測マップを取得する

        Returns:
            np.ndarray: 各エージェントに対する、向こう3ステップ分のA*による最短経路を使った予測マップ

        :return: a dict of 3D np arrays. Each astar_maps[agentID] is a num_future_steps * obs_size * obs_size matrix.
        """

        def get_single_astar_path(distance_map, start_position, path_len) -> List[List]:
            """
            :param distance_map:
            :param start_position:
            :param path_len:
            :return: [[(x,y), ...],..] a list of lists of positions from start_position, the length of the return can be
            smaller than num_future_steps. Index of the return: list[step][0-n] = tuple(x, y)
            """

            def get_astar_one_step(position):
                next_astar_cell = []
                h = self.world.state.shape[0]
                w = self.world.state.shape[1]
                for direction in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                    new_pos = tuple_plus(position, direction)
                    if 0 < new_pos[0] <= h and 0 < new_pos[1] <= w:
                        if distance_map[new_pos] == distance_map[position] - 1 and distance_map[new_pos] >= 0:
                            next_astar_cell.append(new_pos)
                return next_astar_cell

            path_counter = 0
            astar_list = [[start_position]]
            while path_counter < path_len:
                last_step_cells = astar_list[-1]
                next_step_cells = []
                for cells_per_step in last_step_cells:
                    new_cell_list = get_astar_one_step(cells_per_step)
                    if not new_cell_list:  # no next step, should be standing on goal
                        astar_list.pop(0)
                        return astar_list
                    next_step_cells.extend(new_cell_list)
                next_step_cells = list(set(next_step_cells))  # remove repeated positions
                astar_list.append(next_step_cells)
                path_counter += 1

            astar_list.pop(0)
            return astar_list

        astar_maps = {}
        for agentID in range(1, self.world.num_agents + 1):
            astar_maps.update(
                {
                    agentID: np.zeros(
                        [
                            self.num_future_steps,
                            self.world.state.shape[0],
                            self.world.state.shape[1],
                        ]
                    )
                }
            )

            distance_map0, start_pos0 = (
                self.world.agents[agentID].distanceMap,
                self.world.agents[agentID].position,
            )
            astar_path = get_single_astar_path(distance_map0, start_pos0, self.num_future_steps)

            if not len(astar_path) == self.num_future_steps:  # this agent reaches its goal during future steps
                distance_map1, start_pos1 = (
                    self.world.agents[agentID].next_distanceMap,
                    self.world.agents[agentID].goal_pos,
                )
                astar_path.extend(
                    get_single_astar_path(
                        distance_map1,
                        start_pos1,
                        self.num_future_steps - len(astar_path),
                    )
                )

            for i in range(self.num_future_steps - len(astar_path)):  # only happen when min_distance not sufficient
                try:
                    astar_path.extend([[astar_path[-1][-1]]])  # stay at the last pos
                except IndexError:
                    raise ImportError(
                        "Invalid astar path."
                        "If Cython-astar is on, it means that Cython is not correctly compiled."
                        "Refer to README for compilation."
                    )

            assert len(astar_path) == self.num_future_steps
            for step in range(self.num_future_steps):
                for cell in astar_path[step]:
                    astar_maps[agentID][step, cell[0], cell[1]] = 1

        return np.asarray([astar_maps[i] for i in range(1, self.world.num_agents + 1)])  # type: ignore

    def get_blocking(self, corridor_id: int, reverse: int, agent_id: int, dead_end: int) -> int:
        def get_last_pos(agentID, position):
            history_list = copy.deepcopy(self.world.agents[agentID].position_history)
            history_list.reverse()
            assert history_list[0] == self.world.getPos(agentID)
            history_list.pop(0)
            if history_list == []:
                return None
            for pos in history_list:
                if pos != position:
                    return pos
            return None

        positions_to_check = copy.deepcopy(self.world.corridors[corridor_id]["Positions"])
        if reverse:
            positions_to_check.reverse()
        idx = -1
        for position in positions_to_check:
            idx += 1
            state = self.world.state[position[0], position[1]]
            if state > 0 and state != agent_id:
                if dead_end == 1:
                    return 1
                if idx == 0:
                    return 1
                last_pos = get_last_pos(state, position)
                if last_pos == None:
                    return 1
                if idx == len(positions_to_check) - 1:
                    if last_pos != positions_to_check[idx - 1]:
                        return 1
                    break
                if last_pos == positions_to_check[idx + 1]:
                    return 1
        if dead_end == 2:
            if not reverse:
                other_endpoint = self.world.corridors[corridor_id]["EndPoints"][1]
            else:
                other_endpoint = self.world.corridors[corridor_id]["EndPoints"][0]
            state_endpoint = self.world.state[other_endpoint[0], other_endpoint[1]]
            if state_endpoint > 0 and state_endpoint != agent_id:
                return -1
        return 0

    def get_obs_size(self) -> int:
        return self.expanded_size


if __name__ == "__main__":
    pass
