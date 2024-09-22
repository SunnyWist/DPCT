import threading
import numpy as np
import ray

from ACNet import ACNet
import GroupLock
from PRIMAL2_Env import PRIMAL2_Env
from Map_Generator import MazeGenerator
from Worker import Worker

from PyConfigs.TrainConfig import ParamatersConfig


class Runner(object):
    """Actor object to start running simulation on workers.
    Gradient computation is also executed on this object."""

    def __init__(self, metaAgentID: int, conf: ParamatersConfig):
        # パラメータ設定
        self.conf = conf

        # 選択したPlanによるPRIMAL2_Observerの選択
        if self.conf.PLAN_NUM == 1:
            from PRIMAL2_ObserverPlan1 import PRIMAL2_Observer
        else:
            from PRIMAL2_Observer import PRIMAL2_Observer

        # tensorflow must be imported within the constructor
        # because this class will be instantiated on a remote ray node
        import tensorflow as tf

        num_agents = self.conf.NUM_THREADS
        self.env = PRIMAL2_Env(
            num_agents=num_agents,
            observer=PRIMAL2_Observer(
                observation_size=self.conf.OBS_SIZE,
                num_future_steps=self.conf.NUM_FUTURE_STEPS,
                small_observation_size=self.conf.SMALL_OBS_SIZE,
            ),
            map_generator=MazeGenerator()(
                env_size=self.conf.ENVIRONMENT_SIZE,
                wall_components=self.conf.WALL_COMPONENTS,
                obstacle_density=self.conf.OBSTACLE_DENSITY,
            ),
        )

        self.metaAgentID = metaAgentID

        trainer = None
        self.localNetwork = ACNet(
            self.conf.GLOBAL_NET_SCOPE,
            self.conf.A_SIZE,
            trainer,
            True,
            self.conf.NUM_CHANNEL,
            self.conf.OBS_SIZE,
            GLOBAL_NET_SCOPE=self.conf.GLOBAL_NET_SCOPE,
            PLAN_NUM=self.conf.PLAN_NUM,
            SMALL_OBS_SIZE=self.conf.SMALL_OBS_SIZE,
            GLOBAL_NETWORK=False,
        )
        self.currEpisode = int(metaAgentID)

        self.global_step = tf.placeholder(tf.float32)

        # first `NUM_IL_META_AGENTS` only use IL and don't need gpu/tensorflow
        if self.metaAgentID < self.conf.NUM_IL_META_AGENTS:
            config = tf.ConfigProto(allow_soft_placement=True, device_count={"GPU": 0})
            self.coord = None
            self.saver = None

        else:
            # set up tf session
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.per_process_gpu_memory_fraction = 1.0 / (
                self.conf.NUM_META_AGENTS - self.conf.NUM_IL_META_AGENTS + 1
            )
            config.gpu_options.allow_growth = True

            self.saver = tf.train.Saver(max_to_keep=1)
            self.coord = tf.train.Coordinator()

        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        self.weightVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        weights = self.sess.run(self.weightVars)
        self.weightSetters = [tf.placeholder(shape=w.shape, dtype=tf.float32) for w in weights]
        self.set_weights_ops = [var.assign(w) for var, w in zip(self.weightVars, self.weightSetters)]

    def set_weights(self, weights):
        feed_dict = {self.weightSetters[i]: w for i, w in enumerate(weights)}
        self.sess.run([self.set_weights_ops], feed_dict=feed_dict)

    def multiThreadedJob(self, episodeNumber):
        workers = []
        worker_threads = []
        workerNames = ["worker_" + str(i + 1) for i in range(self.conf.NUM_THREADS)]
        groupLock = GroupLock.GroupLock([workerNames, workerNames])  # TODO

        workersPerMetaAgent = self.conf.NUM_THREADS

        for a in range(self.conf.NUM_THREADS):
            agentID = a + 1

            workers.append(
                Worker(
                    self.metaAgentID,
                    agentID,
                    workersPerMetaAgent,
                    self.env,
                    self.localNetwork,
                    self.sess,
                    groupLock,
                    learningAgent=True,
                    global_step=self.global_step,
                    conf=self.conf,
                )
            )

        for w in workers:
            groupLock.acquire(0, w.name)
            worker_work = lambda: w.work(episodeNumber, self.coord, self.saver, self.weightVars)
            t = threading.Thread(target=(worker_work))
            t.start()

            worker_threads.append(t)

        self.coord.join(worker_threads)

        jobResults = []
        loss_metrics = []
        perf_metrics = []
        is_imitation = None

        # from .Parameter import JOB_OPTIONS

        for w in workers:
            if w.learningAgent:
                if self.conf.JOB_TYPE == self.conf.JOB_OPTIONS.getGradient:
                    jobResults = jobResults + w.allGradients
                elif self.conf.JOB_TYPE == self.conf.JOB_OPTIONS.getExperience:
                    jobResults.append(w.experienceBuffer)

            is_imitation = False  # w.is_imitation

            loss_metrics.append(w.loss_metrics)
            perf_metrics.append(w.perf_metrics)

        avg_loss_metrics = list(np.mean(np.array(loss_metrics), axis=0))

        if not is_imitation:
            # perf_metrics structure:
            #
            # w.perf_metrics = [
            #    episode_step_count,
            #    episode_values,
            #    episode_inv_count,
            #    episode_stop_count,
            #    episode_reward,
            #    targets_done
            # ]

            perf_metrics = np.array(perf_metrics)
            avg_perf_metrics = np.mean(perf_metrics[:, :4], axis=0)
            episode_reward = np.sum(perf_metrics[:, 4])
            targets_done = np.sum(perf_metrics[:, 5])
            avg_perf_metrics = list(avg_perf_metrics) + [episode_reward, targets_done]
            all_metrics = avg_loss_metrics + avg_perf_metrics
        else:
            all_metrics = avg_loss_metrics

        return jobResults, all_metrics, is_imitation

    def imitationLearningJob(self, episodeNumber):
        workersPerMetaAgent = self.conf.NUM_THREADS
        agentID = None
        groupLock = None

        worker = Worker(
            self.metaAgentID,
            agentID,
            workersPerMetaAgent,
            self.env,
            self.localNetwork,
            self.sess,
            None,
            learningAgent=True,
            global_step=self.global_step,
            conf=self.conf,
        )

        gradients, losses = worker.imitation_learning_only(episodeNumber)
        mean_imitation_loss = [np.mean(losses)]

        is_imitation = True

        return gradients, mean_imitation_loss, is_imitation

    def job(self, global_weights, episodeNumber):
        print("starting episode {} on metaAgent {}".format(episodeNumber, self.metaAgentID))

        # set the local weights to the global weight values from the master network
        self.set_weights(global_weights)

        # from .Parameter import COMPUTE_OPTIONS

        # set first `NUM_IL_META_AGENTS` to perform imitation learning
        if self.metaAgentID < self.conf.NUM_IL_META_AGENTS:
            print("running imitation job")
            jobResults, metrics, is_imitation = self.imitationLearningJob(episodeNumber)

        elif self.conf.COMPUTE_TYPE == self.conf.COMPUTE_OPTIONS.multiThreaded:
            jobResults, metrics, is_imitation = self.multiThreadedJob(episodeNumber)

        elif self.conf.COMPUTE_TYPE == self.conf.COMPUTE_OPTIONS.synchronous:
            print("not implemented")
            assert 1 == 0

        # Get the job results from the learning agents
        # and send them back to the master network
        info = {
            "id": self.metaAgentID,
            "episode_number": episodeNumber,
            "is_imitation": is_imitation,
        }

        return jobResults, metrics, info


@ray.remote(num_cpus=3, num_gpus=1.0 / 6.0)
class RLRunner(Runner):
    def __init__(self, metaAgentID, conf: ParamatersConfig):
        super().__init__(metaAgentID, conf)


@ray.remote(num_cpus=1, num_gpus=0)
class imitationRunner(Runner):
    def __init__(self, metaAgentID, conf: ParamatersConfig):
        super().__init__(metaAgentID, conf)
