from __future__ import division
from time import sleep
from keras.activations import relu, linear
from keras.layers.advanced_activations import LeakyReLU
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# %matplotlib inline

from PIL import Image

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout, concatenate, Permute
from keras.layers import Input, Conv2D
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor

savefile = "./savefile.h7"

env = gym.make('Enduro-v0')

plt.imshow(env.render(mode='rgb_array'))

nb_actions = env.action_space.n
# print('Total number of Possible action is :', nb_actions)
frame_shape = (84, 84)
window_length = 4
input_shape = (window_length,) + frame_shape
# print('Input Shape is :', input_shape)

class EnduroCNN():
    def train_enduro(self):
        model = Sequential()
        model.add(Permute((2, 3, 1), input_shape=input_shape))
        model.add(Conv2D(32, (8, 8), strides=(4, 4)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (4, 4), strides=(2, 2)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1)))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(nb_actions))
        model.add(Activation('linear'))
        print(model.summary())

        memory = SequentialMemory(limit=1000000, window_length=window_length)
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05, nb_steps=1000000)
        dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory, processor=GameProcess(),
                       nb_steps_warmup=50000, gamma=.99, target_model_update=10000, train_interval=4, delta_clip=1.)
        dqn.compile(Adam(lr=.00025), metrics=['mae'])
        history = dqn.fit(env, nb_steps=2000000)
        dqn.load_weights('dqn_atari_Enduro.h5f')
        env.reset()

        dqn.test(env, nb_episodes=1, visualize=True)

        plt.imshow(env.render(mode='rgb_array'))

        env.close()

    def save(self, name):
        self.model.saver.save(self.model.sess, name)

    def load(self, name):
        self.model.saver.restore(self.model.sess, name)

class GameProcess(Processor):
    def process_observation(self, observation):
        img = Image.fromarray(observation)
        img = np.array(img.resize(frame_shape).convert('L'))
        return img.astype('uint8')

    def process_state_batch(self, batch):
        Processed_batch = batch.astype('float32') / 255.
        return Processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)


agent = EnduroCNN()
def train_agent():
    agent.train_enduro()
    agent.save(savefile)

def load_agent():
    agent.load(savefile)


if __name__ == "__main__":
    train_agent()