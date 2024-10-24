# 通常のACNet
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# ignore warnings
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow.contrib.layers as layers  # type: ignore
import numpy as np

# parameters for training
GRAD_CLIP = 10.0
KEEP_PROB1 = 1  # was 0.5
KEEP_PROB2 = 1  # was 0.7
RNN_SIZE = 512
GOAL_REPR_SIZE = 12

VAR_SCOPE_NAME = "/qvalues"


# Used to initialize weights for policy and value output layers (Do we need to use that? Maybe not now)
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)  # type: ignore
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


class ACNet:
    def __init__(
        self,
        scope,
        a_size,
        trainer,
        TRAINING,
        NUM_CHANNEL,
        OBS_SIZE,
        GLOBAL_NET_SCOPE,
        PLAN_NUM,
        SMALL_OBS_SIZE,
        GLOBAL_NETWORK=False,
        reuse=False,
    ):
        self.plan_num = PLAN_NUM
        self.original_obs_size = OBS_SIZE
        self.small_obs_size = SMALL_OBS_SIZE
        if self.plan_num == 0:
            self.obs_size = self.original_obs_size
        elif self.plan_num == 1:
            self.obs_size = self.original_obs_size * 2 - 1
        else:
            raise ValueError("Not Implemented plan_num")
        with tf.variable_scope(str(scope) + VAR_SCOPE_NAME, reuse=reuse):
            self.trainer = trainer
            # The input size may require more work to fit the interface.
            self.inputs = tf.placeholder(
                shape=[None, NUM_CHANNEL, self.obs_size, self.obs_size],
                dtype=tf.float32,
            )
            self.goal_pos = tf.placeholder(shape=[None, 3], dtype=tf.float32)
            self.myinput = tf.transpose(self.inputs, perm=[0, 2, 3, 1])
            (
                self.policy,
                self.value,
                self.state_out,
                self.state_in,
                self.state_init,
                self.valids,
            ) = self._build_net(self.myinput, self.goal_pos, RNN_SIZE, TRAINING, a_size)
        if TRAINING:
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
            self.train_valid = tf.placeholder(shape=[None, a_size], dtype=tf.float32)
            self.target_v = tf.placeholder(tf.float32, [None], "Vtarget")
            self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

            self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])
            self.train_value = tf.placeholder(tf.float32, [None])
            self.train_policy = tf.placeholder(tf.float32, [None])
            self.train_imitation = tf.placeholder(tf.float32, [None])
            self.optimal_actions = tf.placeholder(tf.int32, [None])
            self.optimal_actions_onehot = tf.one_hot(self.optimal_actions, a_size, dtype=tf.float32)
            self.train_valids = tf.placeholder(tf.float32, [None, 1])

            # Loss Functions
            self.value_loss = 0.1 * tf.reduce_mean(
                self.train_value * tf.square(self.target_v - tf.reshape(self.value, shape=[-1]))
            )

            self.entropy = -tf.reduce_mean(self.policy * tf.log(tf.clip_by_value(self.policy, 1e-10, 1.0)))

            self.policy_loss = -0.5 * tf.reduce_mean(
                self.train_policy * tf.log(tf.clip_by_value(self.responsible_outputs, 1e-15, 1.0)) * self.advantages
            )

            self.valid_loss = -16 * tf.reduce_mean(
                self.train_valids * tf.log(tf.clip_by_value(self.valids, 1e-10, 1.0)) * self.train_valid
                + tf.log(tf.clip_by_value(1 - self.valids, 1e-10, 1.0)) * (1 - self.train_valid)
            )

            self.loss = self.value_loss + self.policy_loss + self.valid_loss - self.entropy * 0.01

            # IMPORTANT: 0 * self.value_loss is important so we can
            #            fetch the gradients properly
            self.imitation_loss = 0 * self.value_loss + tf.reduce_mean(
                self.train_imitation
                * tf.keras.backend.categorical_crossentropy(self.optimal_actions_onehot, self.policy)
            )

            # Get gradients from local network using local losses and
            # normalize the gradients using clipping
            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope + VAR_SCOPE_NAME)
            self.gradients = tf.gradients(self.loss, local_vars)
            self.var_norms = tf.global_norm(local_vars)
            self.grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, GRAD_CLIP)

            # Apply local gradients to global network
            # global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, GLOBAL_NET_SCOPE + VAR_SCOPE_NAME)
            # if self.trainer:
            #     self.apply_grads = self.trainer.apply_gradients(zip(self.grads, global_vars))
            self.local_vars = local_vars

            # now the gradients for imitation loss
            self.i_gradients = tf.gradients(self.imitation_loss, local_vars)
            self.i_var_norms = tf.global_norm(local_vars)
            self.i_grads, self.i_grad_norms = tf.clip_by_global_norm(self.i_gradients, GRAD_CLIP)

            # Apply local gradients to global network
            # if self.trainer:
            #     self.apply_imitation_grads = self.trainer.apply_gradients(zip(self.i_grads, global_vars))

        if GLOBAL_NETWORK:
            print("\n\n\n\n is a global network\n\n\n\n")
            weightVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self.tempGradients = [tf.placeholder(shape=w.get_shape(), dtype=tf.float32) for w in weightVars]
            self.apply_grads = self.trainer.apply_gradients(zip(self.tempGradients, weightVars))

        print("Hello World... From  " + str(scope))  # :)

    def _build_net(self, inputs, goal_pos, RNN_SIZE, TRAINING, a_size):
        # VGG layer(畳み込みを3回した後に、プーリングする)
        def VGG_Block(inputs):
            def conv_2d(inputs, kernal_size, output_size):
                conv = layers.conv2d(
                    inputs=inputs,
                    padding="SAME",
                    num_outputs=output_size,
                    kernel_size=[kernal_size[0], kernal_size[1]],
                    stride=1,
                    data_format="NHWC",
                    weights_initializer=w_init,
                    activation_fn=tf.nn.relu,
                )

                return conv

            conv1 = conv_2d(inputs, [3, 3], RNN_SIZE // 4)
            conv1a = conv_2d(conv1, [3, 3], RNN_SIZE // 4)
            conv1b = conv_2d(conv1a, [3, 3], RNN_SIZE // 4)
            pool1 = layers.max_pool2d(inputs=conv1b, kernel_size=[2, 2])
            return pool1

        # VGGを複数回通して、最後に畳み込みを行う
        def VGGs(inputs, obs_size):
            while obs_size > 3:
                inputs = VGG_Block(inputs)
                obs_size = obs_size // 2
            if obs_size == 3:
                return layers.conv2d(
                    inputs=inputs,
                    padding="VALID",
                    num_outputs=RNN_SIZE - GOAL_REPR_SIZE,
                    kernel_size=[3, 3],
                    stride=1,
                    data_format="NHWC",
                    weights_initializer=w_init,
                    activation_fn=None,
                )
            elif obs_size == 2:
                return layers.conv2d(
                    inputs=inputs,
                    padding="VALID",
                    num_outputs=RNN_SIZE - GOAL_REPR_SIZE,
                    kernel_size=[2, 2],
                    stride=1,
                    data_format="NHWC",
                    weights_initializer=w_init,
                    activation_fn=None,
                )
            else:
                raise ValueError("VGGs error")

        w_init = layers.variance_scaling_initializer()
        after_VGGs = VGGs(inputs, self.obs_size)

        flat = tf.nn.relu(layers.flatten(after_VGGs))
        goal_layer = layers.fully_connected(inputs=goal_pos, num_outputs=GOAL_REPR_SIZE)
        hidden_input = tf.concat([flat, goal_layer], 1)
        h1 = layers.fully_connected(inputs=hidden_input, num_outputs=RNN_SIZE)
        d1 = layers.dropout(h1, keep_prob=KEEP_PROB1, is_training=TRAINING)
        h2 = layers.fully_connected(inputs=d1, num_outputs=RNN_SIZE, activation_fn=None)
        d2 = layers.dropout(h2, keep_prob=KEEP_PROB2, is_training=TRAINING)
        self.h3 = tf.nn.relu(d2 + hidden_input)
        # Recurrent network for temporal dependencies
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(RNN_SIZE, state_is_tuple=True)
        c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
        h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
        state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
        state_in = (c_in, h_in)
        rnn_in = tf.expand_dims(self.h3, [0])
        step_size = tf.shape(inputs)[:1]
        state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm_cell,
            rnn_in,
            initial_state=state_in,
            sequence_length=step_size,
            time_major=False,
        )
        lstm_c, lstm_h = lstm_state
        state_out = (lstm_c[:1, :], lstm_h[:1, :])
        self.rnn_out = tf.reshape(lstm_outputs, [-1, RNN_SIZE])

        policy_layer = layers.fully_connected(
            inputs=self.rnn_out,
            num_outputs=a_size,
            weights_initializer=normalized_columns_initializer(1.0 / float(a_size)),
            biases_initializer=None,
            activation_fn=None,
        )
        policy = tf.nn.softmax(policy_layer)
        policy_sig = tf.sigmoid(policy_layer)
        value = layers.fully_connected(
            inputs=self.rnn_out,
            num_outputs=1,
            weights_initializer=normalized_columns_initializer(1.0),
            biases_initializer=None,
            activation_fn=None,
        )

        return policy, value, state_out, state_in, state_init, policy_sig
