import tensorflow as tf
from collections import deque
import numpy as np
import random
import gym
from math import log

record = open("performance", "w")
savefile = "./savefile.h5"

# POLE-SPECIFIC
max_time = 500

gamma = 0.9
class ICNet:
    def __init__(self, sess, input_size, action_num):
        self.sess = sess
        self.x = tf.placeholder("float32", [None, input_size])
        W1 = tf.get_variable('W1', [input_size, 20], initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable('b1', [20], initializer=tf.contrib.layers.xavier_initializer())
        l1 = tf.nn.relu(tf.matmul(self.x, W1)+b1)
        W2 = tf.get_variable('W2', [20, action_num], initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable('b2', [action_num], initializer=tf.contrib.layers.xavier_initializer())
        self.y_hat = tf.matmul(l1, W2)+b2

        #self.y = tf.placeholder("float", [None, action_num])
        self.q_val = tf.placeholder("float32", [None]) #Proper q-vals as calculated by the bellman equation
        self.actions = tf.placeholder("float32", [None, action_num]) #Actions stored as one-hot vectors
        self.q_val_hat = tf.reduce_sum(tf.multiply(self.y_hat, self.actions), axis=1) #The q-vals for the actions selected in game
        #loss = tf.reduce_sum(tf.square(self.q_val - q_val_hat))
        self.loss = tf.losses.mean_squared_error(self.q_val, self.q_val_hat)
        self.optimizer = tf.train.AdamOptimizer(0.001)
        self.train = self.optimizer.minimize(self.loss)
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
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.epsilon = 0.2  # exploration rate
        self.epsilon_min = 0.
        self.epsilon_decay = 0.995
        self.model = self._build_model()
        self.episodes = 3000
        self.training_result = []

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        session = tf.Session()
        model = ICNet(session, self.state_size, self.action_size)
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.action(state)
        return act_values  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for tup in minibatch:
            states.append(tup[0][0])
            actions.append(tup[1])
            rewards.append(tup[2])
            next_states.append(tup[3][0])
            dones.append(tup[4])
        states = np.array(states)
        actions = np.eye(self.action_size)[actions]
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        self.model.update(states, actions, rewards, next_states, dones)

    def load(self, name):
        self.model.saver.restore(self.model.sess, name)

    def save(self, name):
        self.model.saver.save(self.model.sess, name)

    def collect_rollout_data(self, n_timesteps=500, render=False):
        env = gym.make('CartPole-v1')
        done = True
        timesteps = 0
        n_eps = 0
        while timesteps < n_timesteps:
            if render: env.render()
            if done:
                state = env.reset()
                state = np.reshape(state, [1, 4])
                n_eps += 1
            action = self.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 4])
            self.remember(state, action, reward, next_state, done)
            state = next_state
            timesteps += 1
        # finish current rollout
        while not done:
            if render: env.render()
            action = self.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 4])
            self.remember(state, action, reward, next_state, done)
            state = next_state
            timesteps += 1
        print('average timesteps/episode:', timesteps/n_eps)
        return timesteps/n_eps

    # Should be def train(self, agent_action)
    def train(self):
        init = tf.global_variables_initializer()
        self.model.sess.run(init)
        # Iterate the game
        for e in range(self.episodes):
            self.training_result.append(self.collect_rollout_data())

            # optimize
            batch_size = 64
            for optim_step in range(4 * len(self.memory) // batch_size):
                agent.replay(batch_size)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        for e in self.training_result:
            record.write(str(e) + " ")

agent = DQNAgent(4,2)

def train_agent():
    agent.train()
    agent.save(savefile)

def load_agent():
    agent.load(savefile)


if __name__ == "__main__":
    train_agent()
