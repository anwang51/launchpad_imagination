import tensorflow as tf
from collections import deque
import numpy as np
import random
import gym
from math import log
import math
# POLE-SPECIFIC
max_time = 500

gamma = 0.9
class DQNNet:
    def __init__(self, input_height, input_width, channels, action_num, sess=None):
        #with tf.Graph.as_default():
        #with tf.device("/gpu:0"):
        #tf.reset_default_graph()
        self.x = tf.placeholder("float32", [None, input_height, input_width, channels])

        conv1 = tf.contrib.layers.conv2d(
            self.x, 32, 8, 4, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(
            conv1, 64, 4, 2, activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(
            conv2, 64, 3, 1, activation_fn=tf.nn.relu)

        flat = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(flat, 512)

        self.y_hat = tf.nn.elu(tf.contrib.layers.fully_connected(fc1, action_num))
        self.q_val = tf.placeholder("float32", [None]) #Proper q-vals as calculated by the bellman equation
        self.actions = tf.placeholder("float32", [None, action_num]) #Actions stored as one-hot vectors
        q_val_hat = tf.reduce_sum(tf.multiply(self.y_hat, self.actions), 1) #The q-vals for the actions selected in game
        loss = tf.reduce_mean(tf.squared_difference(self.q_val, q_val_hat))
        self.train = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6).minimize(loss)
        self.saver = tf.train.Saver(max_to_keep = 5, keep_checkpoint_every_n_hours =1)
        if(sess is None):
            print("BEEP BOP BOOOP")
            self.sess = tf.Session()
        else:
            print("WEEP WOP WOOP")
            self.sess = sess
        #self.sess.run(tf.global_variables_initializer())

    def update(self, state, action, reward, next_state, done):
        reward_vecs = self.sess.run(self.y_hat, {self.x: next_state})
        q_vals = reward + gamma*np.amax(reward_vecs, 1)*(1-done)
        self.sess.run(self.train, {self.x: state, self.q_val: q_vals, self.actions: action})

    def action(self, state):
        return self.sess.run(self.y_hat, {self.x: state})



# Deep Q-learning Agent
class DQNAgent:
    def __init__(self, state_width, state_height, action_size, sess=None):
        self.sess = sess
        self.state_width = state_width
        self.state_height = state_height
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.epsilon = 0.0  # exploration rate
        self.epsilon_min = 0.0
        self.epsilon_decay = 0.995
        self.model = self._build_model()
        self.episodes = 3000
        self.training_result = []
        

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        # tf.reset_default_graph()
        model = DQNNet(self.state_height, self.state_width, self.action_size, self.sess)
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
        print("shape", states.shape)
        actions = np.eye(self.action_size)[actions]
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        self.model.update(states, actions, rewards, next_states, dones)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def restore_session(self):
        path = tf.train.get_checkpoint_state('./DQNcheckpoints/')
        if path is None:
            raise IOError('No checkpoint to restore in ' + './DQNcheckpoints/')
        else:
            self.model.saver.restore(self.model.sess, path.model_checkpoint_path)
            #global_step = int(path.model_checkpoint_path.split('-')[-1])
        print(self.model.sess.run(self.model.temp))

    # Should be def train(self, agent_action)
    def train(self, restore_session = False):
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
                self.model.saver.save(self.model.sess, './DQNcheckpoints/'+'model')
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
