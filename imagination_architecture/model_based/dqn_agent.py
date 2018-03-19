import tensorflow as tf
from tensorflow import keras
from tensorflow import layers
from tensorflow import train
from collections import deque
import numpy as np
import random
import gym
from math import log
import matplotlib.pyplot as plt

record = open("performance", "w")
savefile = "./savefile.h5"

# POLE-SPECIFIC
max_time = 500


# Deep Q-learning Agent
class DQNAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.episodes = 2000
        self.training_result = []

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = keras.Sequential()
        model.add(keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(keras.layers.Dense(24, activation='relu'))
        model.add(keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
              target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    # Should be def train(self, agent_action)
    def train(self):
        env = gym.make('CartPole-v1')
        print(env.observation_space, env.action_space)
        # Iterate the game
        for e in range(self.episodes):
            # reset state in the beginning of each game
            state = env.reset()
            state = np.reshape(state, [1, 4])
            # time_t represents each frame of the game
            # Our goal is to keep the pole upright as long as possible until score of max_time
            # the more time_t the more score
            scores = []
            for time_t in range(max_time):
                # turn this on if you want to render
                env.render()
                # Decide action
                action = self.act(state)
                # Advance the game to the next frame based on the action.
                # Reward is 1 for every frame the pole survived
                next_state, reward, done, _ = env.step(action)
                
                # POLE-SPECIFIC
                if time_t == max_time - 1:
                    reward = 150
                elif done:
                    reward = -5
                else:
                    reward = log(time_t + 1) / 10 + 1

                next_state = np.reshape(next_state, [1, 4])
                # Remember the previous state, action, reward, and done
                self.remember(state, action, reward, next_state, done)
                # make next_state the new current state for the next frame.
                state = next_state
                # done becomes True when the game ends
                # ex) The agent drops the pole
                if done:
                    # print the score and break out of the loop
                    scores.append(time_t)
                    print("episode: {}/{}, score: {}"
                          .format(e, self.episodes, time_t))
                    break
            # train the agent with the experience of the episode
            self.training_result.append(time_t)
            num_mem = len(self.memory)
            if num_mem > 32:
                num_mem = 32
            agent.replay(num_mem)
        for e in self.training_result:
            record.write(str(e) + " ")

agent = DQNAgent(4,2)

def train_agent():
    agent.train()
    agent.save(savefile)
    plt.plot(score)

def load_agent():
    agent.load(savefile)


if __name__ == "__main__":
    train_agent()
