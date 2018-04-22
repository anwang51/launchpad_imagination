import tensorflow as tf
from collections import deque
import numpy as np
import random
from math import log

record = open("performance", "w")
savefile = "./savefile.h5"

class DQN_Model:

    def __init__(self, sess, input_size, action_size, max_time=float('inf'), gamma=0.9):
        self.sess = sess
        self.max_time = max_time
        self.gamma = gamma
        self.x = tf.placeholder("float32", [None, input_size])
        W1 = tf.Variable(tf.random_uniform([input_size, 20], 0, 1))
        b1 = tf.Variable(tf.random_uniform([20], 0, 1))
        l1 = tf.nn.relu(tf.matmul(self.x, W1)+b1)
        W2 = tf.Variable(tf.random_uniform([20, action_size], 0, 1))
        b2 = tf.Variable(tf.random_uniform([action_size], 0, 1))
        self.y_hat = tf.nn.relu(tf.matmul(l1, W2)+b2)
        self.q_val = tf.placeholder("float32", [None]) #Proper q-vals as calculated by the bellman equation
        self.actions = tf.placeholder("float32", [None, action_size]) #Actions stored as one-hot vectors
        q_val_hat = tf.reduce_sum(tf.multiply(self.y_hat, self.actions), 1) #The q-vals for the actions selected in game
        loss = tf.losses.mean_squared_error(self.q_val, q_val_hat)
        self.train = tf.train.AdamOptimizer(0.001).minimize(loss)
        self.saver = tf.train.Saver()

    def update(self, state, action, reward, next_state, done):
        reward_vecs = self.sess.run(self.y_hat, {self.x: next_state})
        q_vals = reward + self.gamma*np.amax(reward_vecs, 1)*(1-done)
        self.sess.run(self.train, {self.x: state, self.q_val: q_vals, self.actions: action})

    def action(self, state):
        reward_vec = self.sess.run(self.y_hat, {self.x: state})
        return np.argmax(reward_vec)



# Deep Q-learning Agent
class DQNAgent:

    def __init__(self, state_size, action_size, env):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.epsilon = 0.2  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()
        self.episodes = 3000
        self.training_result = []

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        session = tf.Session()
        model = DQN_Model(session, self.state_size, self.action_size)
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
        #print("actions_1", actions)
        actions = np.eye(self.action_size)[actions]
        #print("actions_2", actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        #print("states", states, "actions", actions, "rewards", rewards, "next_states", next_states)
        #print("rewards", rewards)
        self.model.update(states, actions, rewards, next_states, dones)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.saver.restore(self.model.sess, name)

    def save(self, name):
        self.model.saver.save(self.model.sess, name)

    # Should be def train(self, agent_action)
    def train(self):
        init = tf.global_variables_initializer()
        self.model.sess.run(init)
        # Iterate the game
        for e in range(self.episodes):
            # reset state in the beginning of each game
            state = self.env.reset()
            state = np.reshape(state, [1, 4])
            # time_t represents each frame of the game
            # Our goal is to keep the pole upright as long as possible until score of max_time
            # the more time_t the more score
            for time_t in range(self.max_time):
                # turn this on if you want to render
                self.env.render()
                # Decide action
                action = self.act(state)
                # Advance the game to the next frame based on the action.
                # Reward is 1 for every frame the pole survived
                next_state, reward, done, _ = self.env.step(action)
                if(done and time_t < 500):
                    reward = -1
                next_state = np.reshape(next_state, [1, 4])
                # Remember the previous state, action, reward, and done
                self.remember(state, action, reward, next_state, done)
                # make next_state the new current state for the next frame.
                state = next_state
                # done becomes True when the game ends
                # ex) The agent drops the pole
                if done:
                    # print the score and break out of the loop
                    print("episode: {}/{}, score: {}"
                          .format(e, self.episodes, time_t))
                    break
            # train the agent with the experience of the episode
            self.training_result.append(time_t)
            num_mem = len(self.memory)
            if(num_mem > 64):
                num_mem = 64
            for _ in range(100):
                agent.replay(num_mem)
        for e in self.training_result:
            record.write(str(e) + " ")