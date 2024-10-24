import numpy as np
import tensorflow as tf
import os
import ray
import hydra
from omegaconf import DictConfig, OmegaConf

# WARNINGメッセージを非表示にする
import warnings
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=Warning)
tf.get_logger().setLevel("INFO")
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel(logging.ERROR)

from PyConfigs.TrainConfig import ParamatersConfig


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(dcfg: DictConfig):
    from ACNet import ACNet
    from Runner import imitationRunner, RLRunner

    conf: ParamatersConfig = ParamatersConfig(**dcfg)  # type: ignore
    print("Type of ENVIROMENT_SIZE: ", type(conf.ENVIRONMENT_SIZE))
    print("ENVIROMENT_SIZE: ", conf.ENVIRONMENT_SIZE)

    dcfg_dict = OmegaConf.to_container(dcfg, resolve=True)

    ray.init(num_gpus=1)

    tf.reset_default_graph()
    print("Hello World")

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 1.0 / (conf.NUM_META_AGENTS - conf.NUM_IL_META_AGENTS + 1)
    config.gpu_options.allow_growth = True

    # Create directories
    if not os.path.exists(conf.MODEL_PATH):
        os.makedirs(conf.MODEL_PATH)
    if not os.path.exists(conf.GIFS_PATH):
        os.makedirs(conf.GIFS_PATH)

    global_step = tf.placeholder(tf.float32)

    if conf.ADAPT_LR:
        # computes LR_Q/sqrt(ADAPT_COEFF*steps+1)
        # we need the +1 so that lr at step 0 is defined
        lr = tf.divide(
            tf.constant(conf.LR_Q),
            tf.sqrt(tf.add(1.0, tf.multiply(tf.constant(conf.ADAPT_COEFF), global_step))),
        )
    else:
        lr = tf.constant(conf.LR_Q)

    def apply_gradients(global_network, gradients, sess, curr_episode):
        feed_dict = {global_network.tempGradients[i]: g for i, g in enumerate(gradients)}
        feed_dict[global_step] = curr_episode

        sess.run([global_network.apply_grads], feed_dict=feed_dict)

    def writeImitationDataToTensorboard(global_summary, metrics, curr_episode):
        summary = tf.Summary()
        summary.value.add(tag="Losses/Imitation loss", simple_value=metrics[0])
        global_summary.add_summary(summary, curr_episode)
        global_summary.flush()

    def writeEpisodeRatio(global_summary, numIL, numRL, sess, curr_episode):
        summary = tf.Summary()

        current_learning_rate = sess.run(lr, feed_dict={global_step: curr_episode})

        RL_IL_Ratio = numRL / (numRL + numIL)
        summary.value.add(tag="Perf/Num IL Ep.", simple_value=numIL)
        summary.value.add(tag="Perf/Num RL Ep.", simple_value=numRL)
        summary.value.add(tag="Perf/ RL IL ratio Ep.", simple_value=RL_IL_Ratio)
        summary.value.add(tag="Perf/Learning Rate", simple_value=current_learning_rate)
        global_summary.add_summary(summary, curr_episode)
        global_summary.flush()

    def writeToTensorBoard(global_summary, tensorboardData, curr_episode, plotMeans=True):
        # each row in tensorboardData represents an episode
        # each column is a specific metric

        if plotMeans == True:
            tensorboardData = np.array(tensorboardData)
            tensorboardData = list(np.mean(tensorboardData, axis=0))

            (
                valueLoss,
                policyLoss,
                validLoss,
                entropyLoss,
                gradNorm,
                varNorm,
                mean_length,
                mean_value,
                mean_invalid,
                mean_stop,
                mean_reward,
                mean_finishes,
            ) = tensorboardData

        else:
            firstEpisode = tensorboardData[0]
            (
                valueLoss,
                policyLoss,
                validLoss,
                entropyLoss,
                gradNorm,
                varNorm,
                mean_length,
                mean_value,
                mean_invalid,
                mean_stop,
                mean_reward,
                mean_finishes,
            ) = firstEpisode

        summary = tf.Summary()

        summary.value.add(tag="Perf/Reward", simple_value=mean_reward)
        summary.value.add(tag="Perf/Targets Done", simple_value=mean_finishes)
        summary.value.add(tag="Perf/Length", simple_value=mean_length)
        summary.value.add(tag="Perf/Valid Rate", simple_value=(mean_length - mean_invalid) / mean_length)
        summary.value.add(tag="Perf/Stop Rate", simple_value=(mean_stop) / mean_length)

        summary.value.add(tag="Losses/Value Loss", simple_value=valueLoss)
        summary.value.add(tag="Losses/Policy Loss", simple_value=policyLoss)
        summary.value.add(tag="Losses/Valid Loss", simple_value=validLoss)
        summary.value.add(tag="Losses/Entropy Loss", simple_value=entropyLoss)
        summary.value.add(tag="Losses/Grad Norm", simple_value=gradNorm)
        summary.value.add(tag="Losses/Var Norm", simple_value=varNorm)

        global_summary.add_summary(summary, int(curr_episode - len(tensorboardData)))
        global_summary.flush()

    print("Run with PLAN_NUM = ", conf.PLAN_NUM)
    print("Run with OBS_SIZE = ", conf.OBS_SIZE)
    print("Run with SMALL_OBS_SIZE = ", conf.SMALL_OBS_SIZE)
    print("Run with EXPANDED_OBS_SIZE = ", conf.EXPANDED_OBS_SIZE)

    # MODEL_PATH内のyamlファイルにconfigを保存
    OmegaConf.save(dcfg, conf.OUTPUT_CONFIG_FILE_PATH)

    with tf.device("/gpu:0"):
        trainer = tf.contrib.opt.NadamOptimizer(learning_rate=lr, use_locking=True)
        global_network = ACNet(
            conf.GLOBAL_NET_SCOPE,
            conf.A_SIZE,
            trainer,
            False,
            conf.NUM_CHANNEL,
            conf.OBS_SIZE,
            conf.GLOBAL_NET_SCOPE,
            PLAN_NUM=conf.PLAN_NUM,
            SMALL_OBS_SIZE=conf.SMALL_OBS_SIZE,
            GLOBAL_NETWORK=True,
        )

        global_summary = tf.summary.FileWriter(conf.TRAIN_PATH)
        saver = tf.train.Saver(max_to_keep=1)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if conf.LOAD_MODEL:
            print("Loading Model...")
            ckpt = tf.train.get_checkpoint_state(conf.MODEL_PATH)
            p = ckpt.model_checkpoint_path
            p = p[p.find("-") + 1 :]
            p = p[: p.find(".")]
            curr_episode = int(p)

            saver.restore(sess, ckpt.model_checkpoint_path)
            print("curr_episode set to ", curr_episode)
        else:
            curr_episode = 0

        # launch all of the threads:

        il_agents = [imitationRunner.remote(i, conf) for i in range(conf.NUM_IL_META_AGENTS)]
        rl_agents = [RLRunner.remote(i, conf) for i in range(conf.NUM_IL_META_AGENTS, conf.NUM_META_AGENTS)]
        meta_agents = il_agents + rl_agents

        # get the initial weights from the global network
        weight_names = tf.trainable_variables()
        weights = sess.run(weight_names)  # Gets weights in numpy arrays CHECK

        weightVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        # launch the first job (e.g. getGradient) on each runner
        jobList = []  # Ray ObjectIDs
        for i, meta_agent in enumerate(meta_agents):
            jobList.append(meta_agent.job.remote(weights, curr_episode))  # type: ignore
            curr_episode += 1

        tensorboardData = []

        IDs = [None] * conf.NUM_META_AGENTS

        numImitationEpisodes = 0
        numRLEpisodes = 0

        try:
            while True:
                # wait for any job to be completed - unblock as soon as the earliest arrives
                done_id, jobList = ray.wait(jobList)

                # get the results of the task from the object store
                jobResults, metrics, info = ray.get(done_id)[0]

                # imitation episodes write different data to tensorboard
                if info["is_imitation"]:
                    if jobResults:
                        writeImitationDataToTensorboard(global_summary, metrics, curr_episode)
                        numImitationEpisodes += 1
                else:
                    if jobResults:
                        tensorboardData.append(metrics)
                        numRLEpisodes += 1

                # Write ratio of RL to IL episodes to tensorboard
                writeEpisodeRatio(
                    global_summary,
                    numImitationEpisodes,
                    numRLEpisodes,
                    sess,
                    curr_episode,
                )

                if conf.JOB_TYPE == ParamatersConfig.JOB_OPTIONS.getGradient:
                    if jobResults:
                        for gradient in jobResults:
                            apply_gradients(global_network, gradient, sess, curr_episode)

                elif conf.JOB_TYPE == ParamatersConfig.JOB_OPTIONS.getExperience:
                    print("not implemented")
                    assert 1 == 0
                else:
                    print("not implemented")
                    assert 1 == 0

                # Every `SUMMARY_WINDOW` RL episodes, write RL episodes to tensorboard
                if len(tensorboardData) >= conf.SUMMARY_WINDOW:
                    writeToTensorBoard(global_summary, tensorboardData, curr_episode)
                    tensorboardData = []

                # get the updated weights from the global network
                weight_names = tf.trainable_variables()
                weights = sess.run(weight_names)
                curr_episode += 1

                # start a new job on the recently completed agent with the updated weights
                jobList.extend([meta_agents[info["id"]].job.remote(weights, curr_episode)])

                if curr_episode % 100 == 0:
                    print("Saving Model", end="\n")
                    saver.save(sess, conf.MODEL_PATH + "/model-" + str(int(curr_episode)) + ".cptk")
                    print("Saved Model", end="\n")
                if curr_episode == conf.FINISH_EPISODE_NUM:
                    break

            print("Finished Training")
            ray.shutdown()

        except KeyboardInterrupt:
            print("CTRL-C pressed. killing remote workers")
            for a in meta_agents:
                ray.kill(a)
        except Exception as e:
            print(e)
            print("Exception raised. killing remote workers")
            for a in meta_agents:
                ray.kill(a)


if __name__ == "__main__":
    # 仮想ディスプレイを使用するための設定
    from pyvirtualdisplay.display import Display

    display = Display(visible=False, size=(1400, 900))
    display.start()

    main()
