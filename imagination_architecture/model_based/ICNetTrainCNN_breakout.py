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
class ICNet:
    def __init__(self, sess, input_height, input_width, action_num):
        self.sess = sess
        #with tf.device("/gpu:0"):
        self.x = tf.placeholder("float32", [None, input_height, input_width, 3])
        layer1 = tf.image.resize_images(self.x, [32, 42])
        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(
        inputs=layer1,
        filters=16,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        c1height = math.ceil(32 / 2)
        c1width = math.ceil(42 / 2)

        # Convolutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        c2height = math.ceil(c1height / 2)
        c2width = math.ceil(c1width / 2)

        # Convolutional Layer #2 and Pooling Layer #2
        conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        c3height = math.ceil(c2height / 2)
        c3width = math.ceil(c2width / 2)
        # Dense Layer
        pool3_flat = tf.reshape(pool3, [-1, 2560])


        W1 = tf.Variable(tf.random_uniform([2560, action_num], 0, 1))
        b1 = tf.Variable(tf.random_uniform([action_num], 0, 1))

        self.y_hat = tf.nn.elu(tf.matmul(pool3_flat, W1)+b1)

        #self.y = tf.placeholder("float", [None, action_num])
        self.q_val = tf.placeholder("float32", [None]) #Proper q-vals as calculated by the bellman equation
        self.actions = tf.placeholder("float32", [None, action_num]) #Actions stored as one-hot vectors
        q_val_hat = tf.reduce_sum(tf.multiply(self.y_hat, self.actions), 1) #The q-vals for the actions selected in game
        #loss = tf.reduce_sum(tf.square(self.q_val - q_val_hat))
        loss = tf.losses.mean_squared_error(self.q_val, q_val_hat)
        self.train = tf.train.AdamOptimizer(0.001).minimize(loss)
        self.saver = tf.train.Saver()

    def update(self, state, action, reward, next_state, done):
        #print("next_state", next_state)
        #print("rewards", reward)
        reward_vecs = self.sess.run(self.y_hat, {self.x: next_state})
        #print("reward", reward)
        #print("reward_vec", reward_vecs.shape)

        #print("Reward Vec: ", reward_vecs)
        #print("update terms", np.amax(reward_vecs, 1))
        q_vals = reward + gamma*np.amax(reward_vecs, 1)*(1-done)


        #print("q_vals: ", q_vals)

        self.sess.run(self.train, {self.x: state, self.q_val: q_vals, self.actions: action})

    def action(self, state):
        reward_vec = self.sess.run(self.y_hat, {self.x: state})
        #print(reward_vec)
        #print(np.argmax(reward_vec))
        return np.argmax(reward_vec)



# Deep Q-learning Agent
class DQNAgent:
    def __init__(self, state_width, state_height, action_size):
        self.state_width = state_width
        self.state_height = state_height
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.epsilon = 1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()
        self.episodes = 3000
        self.training_result = []

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
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
        # print(batch_size)
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
        env = gym.make('Breakout-v0')
        #print("env stuff", env.observation_space, env.action_space)
        init = tf.global_variables_initializer()
        self.model.sess.run(init)
        epis = 0
        f = open("performance_timeseries", "a")
        # Iterate the game
        while True:
            # reset state in the beginning of each game
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

            #print("shape: ", np.shape(state))
            #print("shape0: ", np.shape(state[0]))
            state = env.render(mode='rgb_array')
            print(state.shape)
            #print("outside")
            #print("after reshape: ", state)
            # time_t represents each frame of the game
            # Our goal is to keep the pole upright as long as possible until score of max_time
            # the more time_t the more score
            performance_score = 0
            done = False
            while not done:
                # turn this on if you want to render
                # env.render()
                # Decide action
                action = agent.act(state)
                #print(np.shape(state))
                #test = np.array([[1,2,3]])
                #print(np.shape(test))
                # Advance the game to the next frame based on the action.
                # Reward is 1 for every frame the pole survived
                _, reward, done, _ = env.step(action)
                next_state = env.render(mode='rgb_array')
                performance_score += reward
                if done:
                    reward = -2
                # Remember the previous state, action, reward, and done
                agent.remember(state, action, reward, next_state, done)
                # make next_state the new current state for the next frame.
                state = next_state
                # done becomes True when the game ends
                # ex) The agent drops the pole
                if done:
                    # # print the score and break out of the loop
                    # print("episode: {}/{}, score: {}"
                    #       .format(e, episodes, reward))
                    break
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
        agent.model.save_model("tfmodel_weights.h5")

#     0: wall = [0, 0, 0]
#     1: floor = [243, 248, 238]
#     2: box_target = [254, 126, 125]
#     3: box_on_target = [254, 95, 56]
#     4: box = [142, 121, 56]
#     5: player = [160, 212, 56]
#     6: player_on_target = [219, 212, 56]


agent = DQNAgent(160, 210, 4)

def train_agent():
    agent.train()
    agent.save(savefile)

def load_agent():
    agent.load(savefile)


if __name__ == "__main__":
    train_agent()
