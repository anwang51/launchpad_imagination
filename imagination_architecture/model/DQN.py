import tensorflow as tf
from collections import deque
import numpy as np
import random
import gym
from math import log
import math
import os

# POLE-SPECIFIC
max_time = 500
VALID_ACTIONS = [0, 1, 2, 3]
gamma = 0.9
class DQNNet:
    def __init__(self, input_height, input_width, channels, action_num, sess=None):
        #with tf.Graph.as_default():
        #with tf.device("/gpu:0"):
        #tf.reset_default_graph()
        tf.reset_default_graph()
        self.sess = sess
        self.X_pl = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
        # The TD target value
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")
        X = tf.to_float(self.X_pl) / 255.0
        batch_size = tf.shape(self.X_pl)[0]
        # Three convolutional layers
        conv1 = tf.contrib.layers.conv2d(
            X, 32, 8, 4, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(
            conv1, 64, 4, 2, activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(
            conv2, 64, 3, 1, activation_fn=tf.nn.relu)
        # Fully connected layers
        flattened = tf.contrib.layers.flatten(conv3)
        print(flattened.shape)
        fc1 = tf.contrib.layers.fully_connected(flattened, 512)
        self.predictions = tf.contrib.layers.fully_connected(fc1, len(VALID_ACTIONS))
        # Get the predictions for the chosen actions only
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)
        # Calcualte the loss
        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)
        # Optimizer Parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())
        # Summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("loss_hist", self.losses),
            tf.summary.histogram("q_values_hist", self.predictions),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))
        ])

    def update(self, state, action, reward, next_state, done):
        reward_vecs = self.sess.run(self.predictions, {self.X_pl: next_state})
        q_vals = reward + gamma*np.amax(reward_vecs, 1)*(1-done)
        self.sess.run(self.train_op, {self.X_pl: state, self.y_pl: q_vals, self.actions_pl: action})

    def action(self, state):
        return self.sess.run(self.predictions, {self.X_pl: state})

# Deep Q-learning Agent
class DQNAgent:
    def __init__(self, state_width, state_height, channels, action_size, env, sess=None):
        self.sess = sess
        self.state_width = state_width
        self.state_height = state_height
        self.channels = channels
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.epsilon = 0.0  # exploration rate
        self.epsilon_min = 0.0
        self.epsilon_decay = 0.995
        self.model = self._build_model()
        self.episodes = 3000
        self.training_result = []
        self.checkpoint_path = 'experiments/Breakout-v0/checkpoints/'
        self.env = env
        self.restore_session()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        tf.reset_default_graph()
        model = DQNNet(self.state_height, self.state_width, self.channels, self.action_size, self.sess)
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.action(np.array([state]))
        act_values = np.argmax(act_values)
        return act_values  # returns action

    def action(self, state):
        act_values = self.model.action(np.array([state]))
        act_values = np.argmax(act_values)
        return act_values  # returns action

    def reward_vec(self, state):
        act_values = self.model.action(np.array([state]))
        return act_values  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for tup in minibatch:
            states.append(tup[0])
            actions.append(tup[1])
            rewards.append(tup[2])
            next_states.append(tup[3])
            dones.append(tup[4])
        states = np.array(states)
        # actions = np.eye(self.action_size)[actions]
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        self.model.update(states, actions, rewards, next_states, dones)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def restore_session(self):
        experiment_dir = os.path.abspath("./experiments/{}".format(self.env.spec.id))
        # Create directories for checkpoints and summaries
        checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
        checkpoint_path = os.path.join(checkpoint_dir, "model")
        monitor_path = os.path.join(experiment_dir, "monitor")

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not os.path.exists(monitor_path):
            os.makedirs(monitor_path)
        saver = tf.train.Saver(max_to_keep=100)
        # Load a previous checkpoint if we find one
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        print(checkpoint_dir)
        print(latest_checkpoint)
        if latest_checkpoint:
            print("Loading model checkpoint {}...\n".format(latest_checkpoint))
            print(saver.saver_def.filename_tensor_name)
            print(saver.saver_def.restore_op_name)
            saver.restore(self.sess, latest_checkpoint)

    # Should be def train(self, agent_action)
    def train(self, restore_session = True):
        if restore_session:
            self.restore_session()

        env = gym.make('Breakout-v0')
        epis = 0
        f = open("performance_timeseries", "a")
        # Iterate the game
        while True:
            while True:
                try:
                    state = env.reset()
                    #state = env.render(mode='rgb_array')
                except RuntimeWarning:
                    print("RuntimeWarning caught: retrying")
                    continue
                except RuntimeError:
                    print("RuntimeError caught: retrying")
                    continue
                else:
                    break
            # state = env.render(mode='rgb_array')
            performance_score = 0
            done = False
            while not done:
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                performance_score += reward
                if done:
                    reward = -2
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    break
            if epis % 1000 == 0:
                self.model.saver.save(self.model.sess, self.checkpoint_path+'model')
                print('Model {} saved'.format(epis))

            out_str = str(performance_score) + " "
            f.write(out_str)
            f.flush()
            print("episode: {}/{}, score: {}"
                          .format(epis, float("inf"), performance_score))
            # train the agent with the experience of the episode
            num_mem = len(agent.memory)
            if num_mem > 32:
                num_mem = 32
            agent.replay(num_mem)
            epis += 1


#agent = DQNAgent(600, 400, 2)
