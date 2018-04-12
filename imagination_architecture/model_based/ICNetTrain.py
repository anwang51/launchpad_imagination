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
        W1 = tf.Variable(tf.random_uniform([input_size, 20], 0, 1))
        b1 = tf.Variable(tf.random_uniform([20], 0, 1))
        l1 = tf.nn.relu(tf.matmul(self.x, W1)+b1)
        W2 = tf.Variable(tf.random_uniform([20, action_num], 0, 1))
        b2 = tf.Variable(tf.random_uniform([action_num], 0, 1))
        self.y_hat = tf.nn.relu(tf.matmul(l1, W2)+b2)

        #self.y = tf.placeholder("float", [None, action_num])
        self.q_val = tf.placeholder("float32", [None]) #Proper q-vals as calculated by the bellman equation
        self.actions = tf.placeholder("float32", [None, action_num]) #Actions stored as one-hot vectors
        q_val_hat = tf.reduce_sum(tf.multiply(self.y_hat, self.actions), 1) #The q-vals for the actions selected in game
        #loss = tf.reduce_sum(tf.square(self.q_val - q_val_hat))
        loss = tf.losses.mean_squared_error(self.q_val, q_val_hat)
        self.train = tf.train.AdamOptimizer(0.001).minimize(loss)
        self.saver = tf.train.Saver()

    def update(self, state, action, reward, next_state):
        #print("next_state", next_state)
        reward_vecs = self.sess.run(self.y_hat, {self.x: next_state})
        #print("Reward Vec: ", reward_vecs)
        q_vals = reward + gamma*np.amax(reward_vecs, 1)
        #print("q_vals: ", q_vals)

        self.sess.run(self.train, {self.x: state, self.q_val: q_vals, self.actions: action})

    def action(self, state):
        reward_vec = self.sess.run(self.y_hat, {self.x: state})
        print(reward_vec)
        return np.argmax(reward_vec)



# Deep Q-learning Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1500)
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()
        self.episodes = 500
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
        for tup in minibatch:
            states.append(tup[0][0])
            actions.append(tup[1])
            rewards.append(tup[2])
            next_states.append(tup[3][0])
        states = np.array(states)
        actions = np.eye(self.action_size)[actions]
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        self.model.update(states, actions, rewards, next_states)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.saver.restore(self.model.sess, name)

    def save(self, name):
        self.model.saver.save(self.model.sess, name)

    # Should be def train(self, agent_action)
    def train(self):
        env = gym.make('CartPole-v1')
        print(env.observation_space, env.action_space)
        init = tf.global_variables_initializer()
        self.model.sess.run(init)
        # Iterate the game
        for e in range(self.episodes):
            # reset state in the beginning of each game
            state = env.reset()
            state = np.reshape(state, [1, 4])
            # time_t represents each frame of the game
            # Our goal is to keep the pole upright as long as possible until score of max_time
            # the more time_t the more score
            for time_t in range(max_time):
                # turn this on if you want to render
                env.render()
                # Decide action
                action = self.act(state)
                # Advance the game to the next frame based on the action.
                # Reward is 1 for every frame the pole survived
                next_state, reward, done, _ = env.step(action)
                
                # # POLE-SPECIFIC
                # if time_t == max_time - 1:
                #     reward = 150
                # elif done:
                #     reward = -5
                # else:
                #     reward = log(time_t + 1) / 10 + 1

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
            if num_mem > 5:
                num_mem = 5
            agent.replay(num_mem)
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