import itertools
import os
import datetime
from omegaconf import DictConfig, OmegaConf
import hydra
import pickle
import pandas as pd
import numpy as np
import ray
from typing import List, Tuple

from PyConfigs.SimulateConfig import SimulateConfig
from PyConfigs.TrainConfig import ParamatersConfig
from PyConfigs.TestingEnvOutputs import TestingEnvOutputs


# 現在時刻を取得(日本時間)
t_delta = datetime.timedelta(hours=9)
JST = datetime.timezone(t_delta, "JST")
now = datetime.datetime.now(JST)


@hydra.main(version_base=None, config_path="../configs", config_name="simulate")
def main(dcfg: DictConfig):
    # pyvirtualdisplayのために後からimport
    from TestingEnv import ContinuousTestsRunner, RL_Planner

    @ray.remote(num_cpus=3, num_gpus=1 / 8)
    class RLRayTester:
        def __init__(self, tester_id: int):
            self.tester_id = tester_id

        def reset(
            self,
            env_file_path: str,
            result_folder_path: str,
            observation_size: int,
            small_observation_size: int,
            num_future_steps: int,
            model_path: str,
            plan_num: float,
            num_channel: int,
            is_oneshot: bool,
            save_gif: bool,
            with_ep: bool,
        ):
            if plan_num == 0:
                from PRIMAL2_Observer import PRIMAL2_Observer
            elif plan_num == 1:
                from PRIMAL2_ObserverPlan1 import PRIMAL2_Observer
            else:
                raise Exception("Invalid planner number: " + str(plan_num))

            self.runner = ContinuousTestsRunner(
                env_file_path=env_file_path,
                result_folder_path=result_folder_path,
                Planner=RL_Planner(
                    observer=PRIMAL2_Observer(
                        observation_size=observation_size,
                        num_future_steps=num_future_steps,
                        small_observation_size=small_observation_size,
                    ),
                    model_path=model_path,
                    plan_num=plan_num,
                    num_channel=num_channel,
                    isOneShot=is_oneshot,
                    with_ep=with_ep,
                ),
                save_gif=save_gif,
            )

        def run_1_test(self) -> int:
            self.runner.run_1_test()
            return self.tester_id

    class SpecifiedEnvFolder:
        """
        Represents a folder containing environment files for each parameters

        それぞれのパラメータを持つ環境ファイルが格納されているフォルダを表す
        """

        def __init__(
            self,
            conf: SimulateConfig,
            model_conf: ParamatersConfig,
            agent: int,
            size: int,
            density: float,
            wall: int,
            time_stamp: str,
        ) -> None:
            self.conf: SimulateConfig = conf
            self.model_conf: ParamatersConfig = model_conf

            self.agent: int = agent
            self.size: int = size
            self.density: float = density
            self.wall: int = wall

            self.common_result_path: str = os.path.join(
                conf.RESULTS_PATH,
                conf.MODEL_NAME,
                "trained_" + conf.MODEL_TIMESTAMP,
                time_stamp,
            )  # like "my_outputs/results/Original_7/trained_20240321_230831/20210322_081028"

            self.first_folder_name: str = conf.FIRST_FOLDER_FORMAT.format(self.size, self.density, self.wall)

            self.result_path: str = os.path.join(
                self.common_result_path,
                self.first_folder_name,
                str(agent),
            )  # like "my_outputs/results/Original_7/trained_20240321_230831/20210322_081028/20size_0.3density_1wall/4"

            self.pre_path: str = os.path.join(
                conf.ENV_FILES_PATH, self.first_folder_name
            )  # like "envs/env_files/20size_0.3density_1wall"

            self.path: str = os.path.join(self.pre_path, str(agent))  # like "envs/env_files/20size_0.3density_1wall/4"

        def get_file_name(self, id_num: int) -> str:
            if self.conf.USE_NPZ:
                return self.conf.ENV_FILE_FORMAT.format(self.agent, self.size, self.density, self.wall, id_num) + ".npz"
            return self.conf.ENV_FILE_FORMAT.format(self.agent, self.size, self.density, self.wall, id_num) + ".npy"

        def get_file_path(self, id_num: int) -> str:
            return os.path.join(self.path, self.get_file_name(id_num))

        def get_pickle_file_name(self, id_num: int) -> str:
            return self.conf.ENV_FILE_FORMAT.format(self.agent, self.size, self.density, self.wall, id_num) + ".pkl"

        def get_one_env_file_path(self, id_num: int) -> str:
            return os.path.join(self.conf.ENV_FILES_PATH, self.get_file_name(id_num))

    def get_all_env_folders_path(
        conf: SimulateConfig,
        model_conf: ParamatersConfig,
        time_stamp: str = now.strftime("%Y%m%d_%H%M%S"),
    ) -> List[List[SpecifiedEnvFolder]]:
        """
        return a list of list of SpecifiedEnvFolder for each combination of SIZE, DENSITY, WALL

        configで指定された環境のパラメータに対して、SIZE, DENSITY, WALLの組み合わせごとに環境をSpecifiedEnvFolderのリストでまとめたリストを返す
        """

        ret_paths: list[list[SpecifiedEnvFolder]] = []

        for params_comb in itertools.product(conf.SIZE.LST, conf.DENSITY.LST, conf.WALL.LST):
            paths_list: list[SpecifiedEnvFolder] = []
            for agent in conf.AGENT.LST:
                specified_env_folder = SpecifiedEnvFolder(conf, model_conf, agent, *params_comb, time_stamp=time_stamp)  # type: ignore
                if specified_env_folder.agent >= 128 and specified_env_folder.size <= 20:
                    continue
                if specified_env_folder.agent >= 256 and specified_env_folder.size <= 40:
                    continue
                if not os.path.exists(specified_env_folder.path):
                    raise Exception("There is no folder at " + specified_env_folder.path)
                paths_list.append(specified_env_folder)
            ret_paths.append(paths_list)

        return ret_paths

    def reset_tester(
        tester: ray.ObjectRef, folder: SpecifiedEnvFolder, id_num: int, one_env: bool = False, with_ep: bool = False
    ):
        """
        アクターであるRLRayTestsRunnerをリセットする
        """
        if one_env:
            file_path = folder.get_one_env_file_path(id_num)
        else:
            file_path = folder.get_file_path(id_num)
        if not os.path.exists(os.path.join(os.getcwd(), file_path)):
            raise Exception("There is no file at " + os.path.join(os.getcwd(), file_path))

        save_gif: bool = id_num == 0

        plan_num = folder.model_conf.PLAN_NUM
        if plan_num == -1:
            raise NotImplementedError("MstarContinuousPlanner is not implemented yet.")
        elif plan_num in [0, 1]:
            reset_obj = tester.reset.remote(  # type: ignore
                file_path,
                folder.result_path,
                folder.model_conf.OBS_SIZE,
                folder.model_conf.SMALL_OBS_SIZE,
                folder.model_conf.NUM_FUTURE_STEPS,
                folder.conf.MODEL_PATH,
                plan_num,
                folder.model_conf.NUM_CHANNEL,
                False,
                save_gif,
                with_ep=with_ep,
            )
            ray.get(reset_obj)
        else:
            raise Exception("Invalid planner number: " + str(plan_num))

    def create_and_wait_for_ray_testers(
        number_of_workers: int, env_folders: List[SpecifiedEnvFolder], one_env: bool = False, with_ep: bool = False
    ):
        # テスターのリストを作成
        testers: List[ray.ObjectRef] = []
        for i in range(number_of_workers):
            testers.append(RLRayTester.remote(i))

        # まだ実行していないパターンのリストを作成
        not_yet_patterns: List[Tuple[SpecifiedEnvFolder, int]] = []
        for env_folder in env_folders:
            not_yet_patterns += [(env_folder, i) for i in range(env_folder.conf.ID_NUM)]

        # テストを実行
        job_list = []
        for tester in testers:
            if len(not_yet_patterns) > 0:
                env_f, id_n = not_yet_patterns.pop(0)
                reset_tester(tester, env_f, id_n, one_env, with_ep)
                job_list.append(tester.run_1_test.remote())  # type: ignore

        # テストが終わるまで待機し、終わったら次のテストを実行
        done_job_count: int = 0
        try:
            while len(job_list) > 0:
                done_id, job_list = ray.wait(job_list)
                r_id = ray.get(done_id)[0]
                if len(not_yet_patterns) > 0:
                    env_f, id_n = not_yet_patterns.pop(0)
                    reset_tester(testers[r_id], env_f, id_n, one_env, with_ep)
                    job_list.append(testers[r_id].run_1_test.remote())  # type: ignore
                done_job_count += 1
        except Exception as e:
            print("Error in ray wait: ", e)

    def make_pickle_all(folders_list: List[List[SpecifiedEnvFolder]], conf: SimulateConfig):
        """
        Merges all results saved in each folder into one pickle file

        各フォルダに保存された結果を一つのpickleファイルにまとめる
        """
        columns = [
            "filename",
            "agent",
            "size",
            "density",
            "wall",
            "id",
            "targed_reached",
            "computing_time_list",
            "num_crash",
            "status",
            "isSuccessful",
            "steps",
        ]
        result_df = pd.DataFrame(columns=columns)
        for folder_list in folders_list:
            for folder in folder_list:
                for i in range(conf.ID_NUM):
                    result_file_path = os.path.join(folder.result_path, folder.get_pickle_file_name(i))
                    if not os.path.exists(result_file_path):
                        raise Exception("There is no file at " + result_file_path)
                    outputs = pickle.load(open(result_file_path, "rb"))
                    if not isinstance(outputs, TestingEnvOutputs):
                        raise Exception("The file at " + result_file_path + " is not a TestingEnvOutputs object.")
                    df_append = pd.DataFrame(
                        {
                            "filename": folder.get_file_name(i),
                            "agent": folder.agent,
                            "size": folder.size,
                            "density": folder.density,
                            "wall": folder.wall,
                            "id": i,
                            "targed_reached": outputs.target_reached,
                            "computing_time_list": np.mean(outputs.computing_time_list),
                            "num_crash": outputs.num_crash,
                            "status": outputs.episode_status,
                            "isSuccessful": outputs.succeed_episode,
                            "steps": outputs.step_count,
                        },
                        columns=columns,
                        index=[0],
                    )
                    result_df = pd.concat([result_df, df_append], ignore_index=True)
        # pickleファイルを作成
        result_df.to_pickle(os.path.join(folders_list[0][0].common_result_path, conf.ALL_RESULTS_PICKLE_FILE))
        # csvファイルも（一応）作成
        result_df.to_csv(os.path.join(folders_list[0][0].common_result_path, conf.ALL_RESULTS_CSV_FILE))
        return

    def simulate_with_env_config(conf: SimulateConfig, model_conf: ParamatersConfig):
        """
        Simulate with the specified config

        指定されたconfigでシミュレーションを行う
        """
        print("Simulating with following parameters...")
        print("Agents:  {}, separate = {}".format(conf.AGENT.LST, conf.AGENT.SEP))
        print("Size:    {}, separate = {}".format(conf.SIZE.LST, conf.SIZE.SEP))
        print("Density: {}, separate = {}".format(conf.DENSITY.LST, conf.DENSITY.SEP))
        print("Wall:    {}, separate = {}".format(conf.WALL.LST, conf.WALL.SEP))
        print("ID_NUM:  {}".format(conf.ID_NUM))

        # シミュレーション対象の環境フォルダのパスを持ったSpecifiedEnvFolderのリストを取得
        specified_env_folders: list[list[SpecifiedEnvFolder]] = get_all_env_folders_path(conf, model_conf)

        # 並列処理の準備
        print("start testing with " + str(conf.NUMBER_OF_WORKERS) + " processes...")

        all_results = []
        folder_count = 0
        all_folder_count = len(specified_env_folders)

        # env_folders_path内の各フォルダに対して、シミュレーションを行う
        all_specified_env_folders: list[SpecifiedEnvFolder] = []
        for env_folders in specified_env_folders:
            if len(env_folders) == 0:
                continue
            print("Simulating folder: {} ({}/{})".format(env_folders[0].pre_path, folder_count + 1, all_folder_count))
            all_specified_env_folders += env_folders
            folder_count += 1
        create_and_wait_for_ray_testers(
            conf.NUMBER_OF_WORKERS, all_specified_env_folders, one_env=False, with_ep=conf.WITH_EP
        )

        # 全ての結果をpickleファイルにまとめる
        make_pickle_all(specified_env_folders, conf)

        print("There are {} folders and {} tests.".format(folder_count, len(all_results)))

    def simulate_with_only_one_env(conf: SimulateConfig, model_conf: ParamatersConfig):
        """
        Simulate with only one environment

        一つの環境でシミュレーションを行う
        """
        # 並列処理の準備
        print("start testing with " + str(conf.NUMBER_OF_WORKERS) + " processes...")

        all_results = []

        # env_folders_path内のyamlファイルを取得し読み込み
        param_yaml_path = os.path.join(conf.ENV_FILES_PATH, "param.yaml")
        if not os.path.exists(param_yaml_path):
            raise Exception("There is no param.yaml file at " + param_yaml_path)
        param_dcfg = OmegaConf.load(param_yaml_path)

        time_stamp = now.strftime("%Y%m%d_%H%M%S")

        agent_num: int = int(param_dcfg["agents_num"])  # type: ignore
        size_num: int = int(param_dcfg["size_num"])  # type: ignore
        density_num: float = float(param_dcfg["density_num"])  # type: ignore
        wall_num: int = int(param_dcfg["wall_num"])  # type: ignore

        env_folder: SpecifiedEnvFolder = SpecifiedEnvFolder(
            conf,
            model_conf,
            agent_num,
            size_num,
            density_num,
            wall_num,
            time_stamp,
        )

        print("Simulating folder: {}".format(env_folder.pre_path))
        create_and_wait_for_ray_testers(conf.NUMBER_OF_WORKERS, [env_folder], one_env=True)

        # 全ての結果をpickleファイルにまとめる
        make_pickle_all([[env_folder]], conf)

        print("There are {} folders and {} tests.".format(1, len(all_results)))

    # ここから処理開始
    conf: SimulateConfig = SimulateConfig(**dcfg)  # type: ignore

    # 環境ファイルが格納されてるフォルダ(ENV_FILES_PATH)の存在確認
    if not os.path.exists(conf.ENV_FILES_PATH):
        raise Exception("There is no ENVS_FOLDER or ENV_FILES_FOLDER.")

    # 指定したモデルが格納されているフォルダ(MODEL_PATH)の存在確認
    if not os.path.exists(conf.MODEL_PATH):
        raise Exception("There is no model at MODEL_PATH: " + conf.MODEL_PATH)

    # モデルのconfigファイルの読み込み
    model_dcfg = OmegaConf.load(conf.MODEL_CONFIG_FILE_PATH)
    model_conf: ParamatersConfig = ParamatersConfig(**model_dcfg)  # type: ignore

    # RESULTS_FOLDERフォルダを作成(存在しなかった場合)
    os.makedirs(conf.RESULTS_PATH, exist_ok=True)

    dcfg_dict = OmegaConf.to_container(dcfg, resolve=True)

    print("Run with planner = ", model_conf.PLAN_NUM)
    print("Run with obs_size = ", model_conf.OBS_SIZE)
    print("Run with small_obs_size = ", model_conf.SMALL_OBS_SIZE)

    if conf.SIMULATE_WITH_ONE_ENV:
        simulate_with_only_one_env(conf, model_conf)
    else:
        simulate_with_env_config(conf, model_conf)


if __name__ == "__main__":
    # 仮装ディスプレイを使用するための設定
    from pyvirtualdisplay.display import Display

    display = Display(visible=False, size=(1400, 900))
    display.start()

    main()
