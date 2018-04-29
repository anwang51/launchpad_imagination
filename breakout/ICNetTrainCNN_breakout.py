import tensorflow as tf
from collections import deque
import numpy as np
import random
import gym
from math import log
import math

record = open("performance", "w")
savefile = "./savefile.h5"

# POLE-SPECIFIC
max_time = 500
gamma = 0.9

class StateProcessor():
    """
    Processes raw Atari images. Resizes it to 84x84 and converts it to grayscale.

    """
    def __init__(self):
        # Builds the Tensorflow graph
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8) #
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(
                self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)

    def process(self, sess, state):
        """
        Args:
            sess: A Tensorflow session object
            state: A [210, 160, 3] Atari RGB State
        Returns:
            A processed [84, 84, 1] state representing grayscale values.
        """
        return sess.run(self.output, { self.input_state: state })


class ICNet:
    def __init__(self, sess, input_height, input_width, action_num):
        self.sess = sess
        #with tf.device("/gpu:0"):
        self.x = tf.placeholder("float32", [None, 84, 84, 4])
        layer1 = tf.to_float(self.x) / 255.0
        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(
        inputs=layer1,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

        # Convolutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

        # Convolutional Layer #2 and Pooling Layer #2
        conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

        # Flattening
        flat = tf.contrib.layers.flatten(conv3)
        
        # Dense Layer 1
        W1 = tf.Variable(tf.random_uniform([451584, 512], 0, 1))
        b1 = tf.Variable(tf.random_uniform([512], 0, 1))
        dense1 = tf.nn.relu(tf.matmul(flat, W1)+b1)

        # Dense Layer 2
        W2 = tf.Variable(tf.random_uniform([512, action_num], 0, 1))
        b2 = tf.Variable(tf.random_uniform([action_num], 0, 1))

        self.y_hat = tf.nn.elu(tf.matmul(dense1, W2)+b2)

        #self.y = tf.placeholder("float", [None, action_num])
        self.q_val = tf.placeholder("float32", [None]) #Proper q-vals as calculated by the bellman equation
        self.actions = tf.placeholder("float32", [None, action_num]) #Actions stored as one-hot vectors
        q_val_hat = tf.reduce_sum(tf.multiply(self.y_hat, self.actions), 1) #The q-vals for the actions selected in game
        #loss = tf.reduce_sum(tf.square(self.q_val - q_val_hat))
        loss = tf.losses.mean_squared_error(self.q_val, q_val_hat)
        self.train = tf.train.AdamOptimizer(0.001).minimize(loss)
        self.saver = tf.train.Saver()

    def update(self, state, action, reward, next_state, done):
        reward_vecs = self.sess.run(self.y_hat, {self.x: next_state})
        q_vals = reward + gamma*np.amax(reward_vecs, 1)*(1-done)
        self.sess.run(self.train, {self.x: state, self.q_val: q_vals, self.actions: action})

    def action(self, state):
        reward_vec = self.sess.run(self.y_hat, {self.x: state})
        return np.argmax(reward_vec)



# Deep Q-learning Agent
class DQNAgent:
    def __init__(self, state_width, state_height, action_size):
        self.state_width = state_width
        self.state_height = state_height
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.epsilon = 1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()
        self.episodes = 3000
        self.training_result = []
        self.batch_size = 32

    def _build_model(self):
        session = tf.Session()
        model = ICNet(session, self.state_height, self.state_width, self.action_size)
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
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
        actions = np.eye(self.action_size)[actions]
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        self.model.update(states, actions, rewards, next_states, dones)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.saver.restore(self.model.sess, name)

    def save(self, name):
        self.model.saver.save(self.model.sess, name)

    def train(self):
        env = gym.make('Breakout-v0')
        init = tf.global_variables_initializer()
        self.model.sess.run(init)
        epis = 0
        f = open("breakout_dqn_performance", "a")
        state_processor = StateProcessor()
        total_t = 0
        while True:
            while True:
                try:
                    env.reset()
                    state = env.render(mode='rgb_array')
                except RuntimeWarning:
                    print("RuntimeWarning caught: retrying")
                    continue
                except RuntimeError:
                    print("RuntimeError caught: retrying")
                    continue
                else:
                    break

            state = env.reset()
            state = state_processor.process(self.model.sess, state)
            state = np.stack([state] * 4, axis=2)
            performance_score = 0
            done = False
            t = 0
            while not done:
                # env.render()
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                next_state = state_processor.process(self.model.sess, next_state)
                next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
                performance_score += reward
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                num_batch = self.batch_size
                if len(self.memory) < self.batch_size:
                    num_batch = len(self.memory)
                agent.replay(num_batch)
                print("\rStep {} ({}) @ Episode {}/{}".format(
                    t, total_t, epis, float("inf")), end="")
                t += 1
                total_t += 1
            sys.stdout.flush()
            out_str = str(performance_score) + " "
            f.write(out_str)
            f.flush()
            print("episode: {}/{}, score: {}"
                          .format(epis, float("inf"), performance_score))
            epis += 1
        agent.model.save_model("breakoutgpu.h5")

env = gym.make('Breakout-v0')
print("obs space ", env.observation_space)
action_size = 4
agent = DQNAgent(160, 210, 4)

def train_agent():
    agent.train()
    agent.save(savefile)

def load_agent():
    agent.load(savefile)


if __name__ == "__main__":
    train_agent()
