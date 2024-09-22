import os
import tensorflow as tf
from parse import parse
import pickle
from typing import List

from Observer_Builder import DummyObserver, ObservationBuilder
from ACNet import ACNet
from Map_Generator import *
from Env_Builder import *
from PyConfigs.TestingEnvOutputs import TestingEnvOutputs

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=Warning)

ENV_FILE_PARSER = "{agents:d}agents_{size:d}size_{density:f}density_{wall:d}wall_id{ID:d}"


class BasePlanner(metaclass=ABCMeta):
    """Base class for planners"""

    @abstractmethod
    def find_path(self, max_length: int, saveImage: bool, time_limit: float = np.Inf) -> Tuple[TestingEnvOutputs, list]:
        raise NotImplementedError

    @abstractmethod
    def set_world(self):
        raise NotImplementedError

    @abstractmethod
    def give_moving_reward(self, agentID: int):
        raise NotImplementedError

    @abstractmethod
    def listValidActions(self, agent_ID: int, agent_obs: List):
        raise NotImplementedError

    @abstractmethod
    def _reset(self, map_generator: Callable, worldInfo=None):
        raise NotImplementedError


class RL_Planner(MAPFEnv, BasePlanner):
    """
    result saved for NN Continuous Planner:
        target_reached      [int ]: num_target that is reached during the episode.
                                    Affected by timeout or non-solution
        computing_time_list [list]: a computing time record of each run of M*
        num_crash           [int ]: number of crash during the episode
        episode_status      [str ]: whether the episode is 'succeed', 'timeout' or 'no-solution'
        succeed_episode     [bool]: whether the episode is successful (i.e. no timeout, no non-solution) or not
        step_count          [int ]: num_step taken during the episode. The 64 timeout step is included
        frames              [list]: list of GIP frames
    """

    def __init__(
        self,
        observer: ObservationBuilder,
        model_path: str,
        plan_num: float,
        num_channel: int,
        IsDiagonal=False,
        isOneShot=False,
        frozen_steps: int = 0,
        gpu_fraction: float = 0.04,
        with_ep: bool = False,
    ):
        super().__init__(
            observer=observer,
            map_generator=DummyGenerator()(),
            num_agents=1,
            IsDiagonal=IsDiagonal,
            frozen_steps=frozen_steps,
            isOneShot=isOneShot,
        )
        self.plan_num = plan_num
        self.num_channel = num_channel
        self.with_ep = with_ep

        self._set_testType()
        self._set_tensorflow(model_path, gpu_fraction)

    def _set_testType(self):
        self.ACTION_COST, self.GOAL_REWARD, self.COLLISION_REWARD = 0, 0.5, 1

    def _set_tensorflow(self, model_path: str, gpu_fraction: float):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
        self.sess = tf.Session(config=config)

        # todo:HAS TO BE ENTERED MANUALLY TO MATCH THE MODEL, to be read from DRLMAPF...

        self.network = ACNet(
            "global",
            a_size=5,
            trainer=None,
            TRAINING=False,
            NUM_CHANNEL=self.num_channel,
            OBS_SIZE=self.observer.observation_size,
            GLOBAL_NET_SCOPE="global",
            PLAN_NUM=self.plan_num,
            SMALL_OBS_SIZE=self.observer.small_size,
            reuse=tf.AUTO_REUSE,
        )

        # load the weights from the checkpoint (only the global ones!)
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver = tf.train.Saver()
        saver.restore(self.sess, ckpt.model_checkpoint_path)

        self.agent_states = []
        for i in range(self.num_agents):
            rnn_state = self.network.state_init
            self.agent_states.append(rnn_state)

    def set_world(self):
        return

    def give_moving_reward(self, agentID: int):
        """移動後の状態に応じて報酬を与える

        Args:
            agentID (int): エージェントID
        """
        collision_status = self.world.agents[agentID].status
        if collision_status == 0:
            # 衝突なし、ゴールに到達していない
            reward = self.ACTION_COST  # 0
            self.isStandingOnGoal[agentID] = False
        elif collision_status == 1:
            # ゴールに到達
            reward = self.ACTION_COST + self.GOAL_REWARD  # 0.5
            self.isStandingOnGoal[agentID] = True
            self.world.agents[agentID].dones += 1
        else:
            # 衝突
            reward = self.ACTION_COST + self.COLLISION_REWARD  # 1
            self.isStandingOnGoal[agentID] = False
        self.individual_rewards[agentID] = reward

    def listValidActions(self, agent_ID, agent_obs):
        return

    def _reset(self, map_generator=None, worldInfo=None):
        self.map_generator = map_generator
        if worldInfo is not None:
            self.world = TestWorld(
                self.map_generator,
                world_info=worldInfo,
                isDiagonal=self.IsDiagonal,
                isConventional=False,
                with_ep=self.with_ep,
            )
            self.world.reset_agent_next_goal()
        else:
            raise Exception("worldInfo is required")
        self.num_agents = self.world.num_agents
        self.observer.set_env(self.world)
        self.fresh = True
        if self.viewer is not None:
            self.viewer = None
        self.agent_states = []
        for i in range(self.num_agents):
            rnn_state = self.network.state_init
            self.agent_states.append(rnn_state)

    def step_greedily(self, o):
        def run_network(o):
            # 各々エージェントにおいて、それぞれの視野(o)を元に次の移動方向(action_dict)を決定する
            inputs, goal_pos, rnn_out = [], [], []

            for agentID in range(1, self.num_agents + 1):
                agent_obs = o[agentID]
                inputs.append(agent_obs[0])
                goal_pos.append(agent_obs[1])
            # compute up to LSTM in parallel
            h3_vec = self.sess.run(
                [self.network.h3], feed_dict={self.network.inputs: inputs, self.network.goal_pos: goal_pos}
            )
            h3_vec = h3_vec[0]
            # now go all the way past the lstm sequentially feeding the rnn_state
            for a in range(0, self.num_agents):
                rnn_state = self.agent_states[a]
                lstm_output, state = self.sess.run(
                    [self.network.rnn_out, self.network.state_out],
                    feed_dict={
                        self.network.inputs: [inputs[a]],
                        self.network.h3: [h3_vec[a]],
                        self.network.state_in[0]: rnn_state[0],
                        self.network.state_in[1]: rnn_state[1],
                    },
                )
                rnn_out.append(lstm_output[0])
                self.agent_states[a] = state
            # now finish in parallel
            policy_vec = self.sess.run([self.network.policy], feed_dict={self.network.rnn_out: rnn_out})
            policy_vec = policy_vec[0]
            action_dict = {agentID: np.argmax(policy_vec[agentID - 1]) for agentID in range(1, self.num_agents + 1)}
            return action_dict

        numCrashedAgents, computing_time = 0, 0

        start_time = time.time()
        action_dict = run_network(o)
        computing_time = time.time() - start_time

        next_o, reward = self.step_all(action_dict)

        for agentID in reward.keys():
            if reward[agentID] // 1 != 0:
                numCrashedAgents += 1
        assert numCrashedAgents <= self.num_agents

        return numCrashedAgents, computing_time, next_o

    def find_path(self, max_length, saveImage, time_limit=np.Inf) -> Tuple[TestingEnvOutputs, list]:
        assert max_length > 0
        step_count, num_crash, computing_time_list, frames = 0, 0, [], []
        episode_status = "no early stop"

        obs = self._observe()
        for step in range(1, max_length + 1):
            if saveImage:
                frames.append(self._render(mode="rgb_array"))
            numCrash_AStep, computing_time, obs = self.step_greedily(obs)

            computing_time_list.append(computing_time)
            num_crash += numCrash_AStep
            step_count = step

            if time_limit < computing_time:
                episode_status = "timeout"
                break

        if saveImage:
            frames.append(self._render(mode="rgb_array"))

        target_reached = 0
        for agentID in range(1, self.num_agents + 1):
            target_reached += self.world.getDone(agentID)
        outputs = TestingEnvOutputs(
            target_reached=target_reached,
            computing_time_list=computing_time_list,
            num_crash=num_crash,
            episode_status=episode_status,
            succeed_episode=episode_status == "no early stop",
            step_count=step_count,
        )
        return outputs, frames


class MstarContinuousPlanner(MAPFEnv, BasePlanner):
    def __init__(self, IsDiagonal=False, frozen_steps=0):
        super().__init__(
            observer=DummyObserver(),
            map_generator=DummyGenerator()(),
            num_agents=1,
            IsDiagonal=IsDiagonal,
            frozen_steps=frozen_steps,
            isOneShot=False,
        )
        self._set_testType()

    def set_world(self):
        return

    def give_moving_reward(self, agentID):
        collision_status = self.world.agents[agentID].status
        if collision_status == 0:
            reward = self.ACTION_COST
            self.isStandingOnGoal[agentID] = False
        elif collision_status == 1:
            reward = self.ACTION_COST + self.GOAL_REWARD
            self.isStandingOnGoal[agentID] = True
            self.world.agents[agentID].dones += 1
        else:
            reward = self.ACTION_COST + self.COLLISION_REWARD
            self.isStandingOnGoal[agentID] = False
        self.individual_rewards[agentID] = reward

    def listValidActions(self, agent_ID, agent_obs):
        return

    def _set_testType(self):
        self.ACTION_COST, self.GOAL_REWARD, self.COLLISION_REWARD = 0, 0.5, 1
        self.test_type = "continuous"
        self.method = "_" + self.test_type + "mstar"

    def _reset(self, map_generator=None, worldInfo=None):
        self.map_generator = map_generator
        if worldInfo is not None:
            self.world = TestWorld(
                self.map_generator, world_info=worldInfo, isDiagonal=self.IsDiagonal, isConventional=True
            )
        else:
            self.world = World(self.map_generator, num_agents=self.num_agents, isDiagonal=self.IsDiagonal)
        self.num_agents = self.world.num_agents
        self.observer.set_env(self.world)
        self.fresh = True
        if self.viewer is not None:
            self.viewer = None

    def find_path(self, max_length, saveImage, time_limit=300):
        """
        end episode when 1. max_length is reached immediately, or
                         2. 64 steps after the first timeout, or
                         3. non-solution occurs immediately

        target_reached      [int ]: num_target that is reached during the episode.
                                    Affected by timeout or non-solution
        computing_time_list [list]: a computing time record of each run of M*
        num_crash           [int ]: zero crash in M* mode
        episode_status      [str ]: whether the episode is 'succeed', 'timeout' or 'no-solution'
        succeed_episode     [bool]: whether the episode is successful or not
        step_count          [int ]: num_step taken during the episode. The 64 timeout step is included
        frames              [list]: list of GIP frames
        """

        def parse_path(path, step_count):
            on_goal = False
            path_step = 0
            while step_count < max_length and not on_goal:
                actions = {}
                for i in range(self.num_agents):
                    agent_id = i + 1
                    next_pos = path[path_step][i]
                    diff = tuple_minus(next_pos, self.world.getPos(agent_id))
                    actions[agent_id] = dir2action(diff)

                    if self.world.agents[agent_id].goal_pos == next_pos and not on_goal:
                        on_goal = True

                self.step_all(actions, check_col=False)
                if saveImage:
                    frames.append(self._render(mode="rgb_array"))

                step_count += 1
                path_step += 1
            return step_count if step_count <= max_length else max_length

        def compute_path_piece(time_limit):
            succeed = True
            start_time = time.time()
            path = self.expert_until_first_goal(inflation=3.0, time_limit=time_limit / 5.0)
            # /5 bc we first try C++ M* with 5x less time, then fall back on python if need be where we remultiply by 5
            c_time = time.time() - start_time
            if c_time > time_limit or path is None:
                succeed = False
            return path, succeed, c_time

        assert max_length > 0
        frames, computing_time_list = [], []
        target_reached, step_count, episode_status = 0, 0, "succeed"

        while step_count < max_length:
            path_piece, succeed_piece, c_time = compute_path_piece(time_limit)
            computing_time_list.append(c_time)
            if not succeed_piece:  # no solution, skip out of loop
                if c_time > time_limit:  # timeout, make a last computation and skip out of the loop
                    episode_status = "timeout"
                    break
                else:  # no solution
                    episode_status = "no-solution"
                    break
            else:
                step_count = parse_path(path_piece, step_count)

        for agentID in range(1, self.num_agents + 1):
            target_reached += self.world.getDone(agentID)

        return target_reached, computing_time_list, 0, episode_status, episode_status == "succeed", step_count, frames


class ContinuousTestsRunner:
    def __init__(self, env_file_path: str, result_folder_path: str, Planner: BasePlanner, save_gif: bool):
        """ContinuousTestsRunnerのコンストラクタ

        Args:
            env_file_path (str): 環境ファイルのパス
            result_file_path (str): resultファイルのパス
            Planner (BasePlanner): プランナー
            save_gif (bool): GIF画像を保存するかどうか
        """
        self.env_file_path = env_file_path
        self.result_folder_path = result_folder_path
        self.save_gif: bool = save_gif
        self.worker = Planner

        if not os.path.exists(self.result_folder_path):
            os.makedirs(self.result_folder_path, exist_ok=True)
        self.maps = self.__read_single_env()

    def __read_single_env(self) -> np.ndarray:
        """環境ファイル(.npy or .npz)を読み込む

        Returns:
            np.ndarray: 環境ファイル
        """
        if self.env_file_path is None:
            raise ValueError("env_file_path is not set")
        assert self.env_file_path.endswith(".npy") or self.env_file_path.endswith(".npz")
        maps = np.load(self.env_file_path, allow_pickle=True)
        if self.env_file_path.endswith(".npz"):
            maps = maps["arr_0"]
        return maps

    def run_1_test(self):
        """テストを実行する"""

        def get_maxLength(env_size: int) -> int:
            if env_size <= 40:
                return 128
            elif env_size <= 80:
                return 192
            return 256

        def make_gif_continuous(image: List, env_name: str, result_path: str):
            if image:
                gif_file_name = env_name + ".gif"
                gif_path = os.path.join(result_path, gif_file_name)
                images = np.array(image)
                make_gif(images, gif_path)

        def write_files(outputs: TestingEnvOutputs, max_length: int, env_name: str, result_path: str):
            txt_file_name = env_name + ".txt"
            txt_path = os.path.join(result_path, txt_file_name)
            f = open(txt_path, "w")
            outputs_dict = outputs.__dict__
            for key in outputs_dict.keys():
                if key == "frames":
                    continue
                f.write(key + ": " + str(outputs_dict[key]) + "\n")
            f.write("max_length: " + str(max_length) + "\n")
            f.close()

        def write_pickle(outputs: TestingEnvOutputs, env_name: str, result_path: str):
            pickle_file_name = env_name + ".pkl"
            pickle_path = os.path.join(result_path, pickle_file_name)
            with open(pickle_path, "wb") as f:
                pickle.dump(outputs, f)

        self.worker._reset(map_generator=ManualGenerator()(self.maps[0][0], self.maps[0][1]), worldInfo=self.maps)
        env_file_name = os.path.basename(self.env_file_path)
        env_name = env_file_name[: env_file_name.rfind(".")]
        env_parsed = parse(ENV_FILE_PARSER, env_name)
        if env_parsed is None:
            raise NameError("invalid env name")
        env_parsed_named = env_parsed.named  # type: ignore
        env_size = env_parsed_named["size"]
        max_length = get_maxLength(env_size)

        print("working on " + env_name)

        outputs, frames = self.worker.find_path(max_length=int(max_length), saveImage=self.save_gif)

        make_gif_continuous(frames, env_name, self.result_folder_path)
        write_files(outputs, max_length, env_name, self.result_folder_path)
        write_pickle(outputs, env_name, self.result_folder_path)

        return
