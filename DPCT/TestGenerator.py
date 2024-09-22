import multiprocessing as mp
from pathos.multiprocessing import ProcessPool as Pool
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import itertools
from typing import List, Dict, Tuple, Union
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

from PRIMAL2_Env import *
from Observer_Builder import DummyObserver
from Map_Generator import ManualGenerator
from Env_Builder import *
from PyConfigs.GeneratorConfig import GeneratorConfig


class MazeTestGenerator:
    """
    テストを実行するための環境を生成するクラス

    Args:
        env_dir (str): 保存先のディレクトリ
        printInfo (bool): 進捗を表示するかどうか
        pureMaps (bool): state_mapとgoals_mapのみ保存するかどうか
    """

    def __init__(self, env_dir: str, printInfo: bool, use_npz: bool = False, pureMaps: bool = False):
        self.env_dir: str = env_dir
        self.num_core: int = mp.cpu_count()
        self.parallel_pool: Pool = Pool()
        self.printInfo: bool = printInfo
        self.pureMaps: bool = pureMaps  # if true, only save state_map and goals_map, else save the whole world
        self.use_npz: bool = use_npz

    def make_name(self, n: int, s: int, d: float, w: int, id: int, extension: str, dirname: str, extra="") -> str:
        """
        ファイルパスを生成するメソッド

        Args:
            n (int): エージェントの数
            s (int): 環境の一辺サイズ
            d (float): 障害物の密度
            w (int): コリドーの平均の長さ
            id (int): テストのID
            extension (str): 拡張子
            dirname (str): 保存先のディレクトリパス
            extra (str): 追加の情報

        Returns:
            str: ファイル名
        """
        if extra == "":
            file_name = "{}agents_{}size_{}density_{}wall_id{}{}".format(n, s, d, w, id, extension)
        else:
            file_name = "{}agents_{}size_{}density_{}wall_id{}_{}{}".format(n, s, d, w, id, extra, extension)
        return os.path.join(dirname, file_name)

    def create_map(self, num_agents: int, env_size: int, obs_dense: float, wall_component: int, id: int):
        obs_dense = round(obs_dense, 2)
        if self.use_npz:
            extension = ".npz"
        else:
            extension = ".npy"
        file_name = self.make_name(
            num_agents, env_size, obs_dense, wall_component, id, dirname=self.env_dir, extension=extension
        )
        if os.path.exists(file_name):
            if self.printInfo:
                print("skip env:" + file_name)
            return

        gameEnv = DummyEnv(
            num_agents=num_agents,
            observer=DummyObserver(),
            map_generator=MazeGenerator()(
                env_size=env_size, wall_components=wall_component, obstacle_density=obs_dense
            ),
        )
        state = np.array(gameEnv.world.state)
        goals = np.array(gameEnv.world.goals_map)
        if self.pureMaps:
            info = np.array([state, goals])
        else:
            if gameEnv.world.agents_init_pos is None:
                raise ValueError("agents_init_pos is None")
            agents_init_pos: Dict = gameEnv.world.agents_init_pos
            corridor_map = np.array(gameEnv.world.corridor_map)
            corridors = np.array(gameEnv.world.corridors)
            agents_object: Dict = gameEnv.world.agents
            info = np.array([[state, goals], agents_init_pos, corridor_map, corridors, agents_object])

        if self.use_npz:
            np.savez_compressed(file_name, info)
        else:
            np.save(file_name, info)

        if self.printInfo:
            print("finish env: " + file_name)

    def run_mazeMap_creator(
        self,
        num_agents_list: List[int],
        env_size_list: List[int],
        obs_dense_list: List[float],
        wall_component_list: List[int],
        num_tests: int,
        multiProcessing: bool = True,
    ):
        """
        テスト環境を生成するメソッド

        Args:
            num_agents_list (List[int]): エージェント数のリスト
            env_size_list (List[int]): 環境の一辺サイズのリスト
            obs_dense_list (List[float]): 障害物の密度のリスト
            wall_component_list (List[int]): コリドーの平均の長さのリスト
            num_tests (int): 生成するテストの数
            multiProcessing (bool): マルチプロセスで実行するかどうか
        """
        if not os.path.exists(self.env_dir):
            os.makedirs(self.env_dir)
        if multiProcessing:
            print("Multi-processing activated, you are using {:d} processes".format(self.num_core))
        else:
            print("Single-processing activated, you are using 1 processes")

        total_tests = (
            len(num_agents_list) * len(env_size_list) * len(obs_dense_list) * len(wall_component_list) * num_tests
        )
        print("There are {} tests in total. Start Working!".format(total_tests))

        allResults = []

        parameters_combinations = itertools.product(
            num_agents_list, env_size_list, obs_dense_list, wall_component_list, range(num_tests)
        )

        for num_agents, env_size, obs_dense, wall_component, i in parameters_combinations:
            if env_size <= 20 and num_agents >= 128:
                continue
            if env_size <= 40 and num_agents >= 256:
                continue

            if multiProcessing:
                result = self.parallel_pool.apipe(self.create_map, num_agents, env_size, obs_dense, wall_component, i)
                allResults.append(result)
            else:
                self.create_map(num_agents, env_size, obs_dense, wall_component, i)

        totalJobs = len(allResults)
        jobsCompleted = 0
        while len(allResults) > 0:
            for i in range(len(allResults)):
                if allResults[i].ready():
                    jobsCompleted += 1
                    print("{} / {}".format(jobsCompleted, totalJobs))
                    allResults[i].get()
                    allResults.pop(i)
                    break
        self.parallel_pool.close()
        print("finish all envs!")


class MazeTestInfoAdder(MazeTestGenerator):
    """
    add info to previous testing env.
    Info in the npy file  in FIXED ORDER!!!:
    [
    [state_map, goal_map], <----- previous info
    agents_init_pos
    goals_init_pos
    corridor_map
    corridors
    world.agents
    ]
    """

    def __init__(self, env_dir: str, printInfo: bool = False, use_npz: bool = False):
        super(MazeTestInfoAdder, self).__init__(env_dir=env_dir, printInfo=printInfo, use_npz=use_npz)

    def read_envs(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        print("loading testing env...")
        maps_dict: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        for root, dirs, files in os.walk(self.env_dir, topdown=False):
            for name in files:
                if not name.endswith(".npy") and not name.endswith(".npz"):
                    continue
                try:
                    load_file_path = Path(os.path.join(root, name))
                    maps = np.load(load_file_path, allow_pickle=True)
                    if name.endswith(".npz"):
                        maps = maps["arr_0"]
                    print(maps)
                    if len(maps) < 2:
                        continue
                except ValueError:
                    print(
                        root + name,
                        "is a broken file that numpy cannot read, possibly due to the forced "
                        "suspension of generation code. Automatically skip this env...",
                    )
                    continue
                if len(maps) > 2:  # notice that only pure maps will be processed
                    continue
                new_file_name = str(load_file_path.relative_to(self.env_dir))
                maps_dict.update({new_file_name: (maps[0], maps[1])})
        print("There are " + str(len(maps_dict.keys())) + " tests detected")
        return maps_dict

    def add_info(self, state_map: np.ndarray, goals_map: np.ndarray, file_name: str):
        gameEnv = DummyEnv(
            num_agents=1,
            observer=DummyObserver(),
            map_generator=ManualGenerator()(state_map=state_map, goals_map=goals_map),
        )
        state = np.array(gameEnv.world.state)
        goals = np.array(gameEnv.world.goals_map)
        agents_init_pos = gameEnv.world.agents_init_pos
        corridor_map = np.array(gameEnv.world.corridor_map)
        corridors = np.array(gameEnv.world.corridors)
        agents_object = gameEnv.world.agents

        info = np.array([[state, goals], agents_init_pos, corridor_map, corridors, agents_object])
        if self.use_npz:
            file_name = file_name.replace(".npy", ".npz")
            save_path = os.path.join(self.env_dir, "added", file_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez_compressed(save_path, info)
        else:
            file_name = file_name.replace(".npz", ".npy")
            save_path = os.path.join(self.env_dir, "added", file_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, info)

        if self.printInfo:
            print("finish env:" + file_name)

    def run_mazeMap_infoAdder(self, multiProcessing: bool = True):
        """
        最小限のテスト環境にエージェント、ゴール、コリドーの情報を追加するメソッド

        Args:
            multiProcessing (bool): マルチプロセスで実行するかどうか
        """
        if not os.path.exists(self.env_dir):
            os.makedirs(self.env_dir)
        if multiProcessing:
            print("Multi-processing activated, you are using {:d} processes".format(self.num_core))
        else:
            print("Single-processing activated, you are using 1 processes")

        map_dict = self.read_envs()
        print("There are " + str(len(map_dict.keys())) + " tests in total. Start Working!")

        allResults = []
        for file_name, maps in map_dict.items():
            if multiProcessing:
                result = self.parallel_pool.apipe(self.add_info, maps[0], maps[1], file_name)
                allResults.append(result)
            else:
                self.add_info(maps[0], maps[1], file_name)

        totalJobs = len(allResults)
        jobsCompleted = 0
        while len(allResults) > 0:
            for i in range(len(allResults)):
                if allResults[i].ready():
                    jobsCompleted += 1
                    print("{} / {}".format(jobsCompleted, totalJobs))
                    allResults[i].get()
                    allResults.pop(i)
                    break
        self.parallel_pool.close()
        print("finish all envs!")


class MazeTestInfoAdderWithEP(MazeTestGenerator):
    """
    生成したRawMap(エンドポイントのマップ込み)からテスト環境を生成するクラス
    RawMap
    [state_map, goal_map, ep_map]

    テスト環境の情報
    [
    [state_map, goal_map], <----- previous info
    agents_init_pos,
    corridor_map,
    corridors,
    world.agents,
    ep_map, <----- MazeTestInfoAdderとの違い
    ]
    """

    def __init__(self, env_dir: str, printInfo: bool = False, use_npz: bool = False, added_folder_name: str = "added"):
        super(MazeTestInfoAdderWithEP, self).__init__(env_dir=env_dir, printInfo=printInfo, use_npz=use_npz)
        self.added_folder_name: str = added_folder_name

    def read_envs(self) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        print("loading testing env...")
        maps_dict: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        for root, dirs, files in os.walk(self.env_dir, topdown=False):
            for name in files:
                if not name.endswith(".npy") and not name.endswith(".npz"):
                    continue
                try:
                    load_file_path = Path(os.path.join(root, name))
                    maps = np.load(load_file_path, allow_pickle=True)
                    if name.endswith(".npz"):
                        maps = maps["arr_0"]
                    print(maps)
                    if len(maps) < 3:
                        continue
                except ValueError:
                    print(
                        root + name,
                        "is a broken file that numpy cannot read, possibly due to the forced "
                        "suspension of generation code. Automatically skip this env...",
                    )
                    continue
                if len(maps) > 3:  # notice that only pure maps will be processed
                    continue
                new_file_name = str(load_file_path.relative_to(self.env_dir))
                maps_dict.update({new_file_name: (maps[0], maps[1], maps[2])})
        print("There are " + str(len(maps_dict.keys())) + " tests detected")
        return maps_dict

    def add_info(self, state_map: np.ndarray, goals_map: np.ndarray, ep_map: np.ndarray, file_name: str):
        """RawMapからテスト環境を生成するメソッド

        Args:
            state_map (np.ndarray): stateを記録したマップ
            goals_map (np.ndarray): ゴール位置のマップ
            ep_map (np.ndarray): エンドポイントのマップ
            file_name (str): 保存先のファイル名
        """
        gameEnv = DummyEnv(
            num_agents=1,
            observer=DummyObserver(),
            map_generator=ManualGenerator()(state_map=state_map, goals_map=goals_map),
        )
        state = np.array(gameEnv.world.state)
        goals = np.array(gameEnv.world.goals_map)
        agents_init_pos = gameEnv.world.agents_init_pos
        corridor_map = np.array(gameEnv.world.corridor_map)
        corridors = np.array(gameEnv.world.corridors)
        agents_object = gameEnv.world.agents
        ep_map = np.array(ep_map)
        print("state_map: ", state)
        print("goals_map: ", goals)
        print("ep_map: ", ep_map)

        info = np.array([[state, goals], agents_init_pos, corridor_map, corridors, agents_object, ep_map])
        if self.use_npz:
            file_name = file_name.replace(".npy", ".npz")
            save_path = os.path.join(self.env_dir, self.added_folder_name, file_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez_compressed(save_path, info)
        else:
            file_name = file_name.replace(".npz", ".npy")
            save_path = os.path.join(self.env_dir, self.added_folder_name, file_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, info)

        if self.printInfo:
            print("finish env:" + file_name)

    def run_mazeMap_infoAdder(self, multiProcessing: bool = True):
        """
        最小限のテスト環境(エンドポイント込み)にエージェント、ゴール、コリドーの情報を追加するメソッド

        Args:
            multiProcessing (bool): マルチプロセスで実行するかどうか
        """
        if not os.path.exists(self.env_dir):
            os.makedirs(self.env_dir)
        if multiProcessing:
            print("Multi-processing activated, you are using {:d} processes".format(self.num_core))
        else:
            print("Single-processing activated, you are using 1 processes")

        map_dict = self.read_envs()
        print("There are " + str(len(map_dict.keys())) + " tests in total. Start Working!")

        allResults = []
        for file_name, maps in map_dict.items():
            if multiProcessing:
                result = self.parallel_pool.apipe(self.add_info, maps[0], maps[1], maps[2], file_name)
                allResults.append(result)
            else:
                self.add_info(maps[0], maps[1], maps[2], file_name)

        totalJobs = len(allResults)
        jobsCompleted = 0
        while len(allResults) > 0:
            for i in range(len(allResults)):
                if allResults[i].ready():
                    jobsCompleted += 1
                    print("{} / {}".format(jobsCompleted, totalJobs))
                    allResults[i].get()
                    allResults.pop(i)
                    break
        self.parallel_pool.close()
        print("finish all envs!")


@hydra.main(version_base=None, config_path="../configs", config_name="generator")
def main(dcfg: DictConfig):
    # dcfgを展開して、GeneratorConfigのインスタンスを作成
    conf: GeneratorConfig = GeneratorConfig(**dcfg)  # type: ignore
    print("conf: ", conf)
    if conf.ADD_INFO:
        if conf.WITH_EP:
            adder = MazeTestInfoAdderWithEP(
                conf.SAVE_PATH,
                printInfo=conf.PRINT_INFO,
                use_npz=conf.USE_NPZ,
                added_folder_name=conf.ADDED_FOLDER_NAME,
            )
        else:
            adder = MazeTestInfoAdder(
                conf.SAVE_PATH,
                printInfo=conf.PRINT_INFO,
                use_npz=conf.USE_NPZ,
            )
        adder.run_mazeMap_infoAdder()
    else:
        generator = MazeTestGenerator(
            conf.SAVE_PATH, printInfo=conf.PRINT_INFO, use_npz=conf.USE_NPZ, pureMaps=conf.PURE_MAPS
        )
        generator.run_mazeMap_creator(
            conf.AGENT_LIST, conf.SIZE_LIST, conf.DENSITY_LIST, conf.WALL_LIST, conf.TESTS_TO_GENERATE
        )


if __name__ == "__main__":
    main()
