import tensorflow as tf
import gym
import gym_sokoban
import numpy as np
import random
import DQN
import EnvModelBatch
import ImaginationCore
import INet

class ImaginationAgent:
    def __init__(self, state_size, action_size):
        self.sess = tf.Session()
        self.state_size = state_size
        self.action_size = action_size
        self.episodes = int(1e100)
        #self.max_time = 500
        temp = DQN.DQNAgent(self.state_size, self.action_size)
        self.dqn = temp.model
        self.env_model = EnvModelBatch.EnvModel(self.state_size, self.action_size)
        #LSTM_input_size, num_paths, MF_input_size, output_size, path_length, sess
        #output of env model
        self.interpreter = INet.INet(15, 4, 5, 5, 4, self.sess)
        self.icore = ImaginationCore.ImaginationCore('TinyWorld-Sokoban-small-v0', self.state_size, self.action_size)

    def train(self):
        for e in range(self.episodes):
            if e % 100:
                self.save("./final.h5")

            while True:
                try:
                    state = env.reset()
                    state = compress(state)
                    prev_state = state

                except RuntimeWarning:
                    print("RuntimeWarning caught: retrying")
                    continue
                except RuntimeError:
                    print("RuntimeError caught: retrying")
                    continue
                else:
                    break
            done = False
            t = 0

            while not done and t < self.max_time:
                # MODEL FREE
                # store prediction
                # get the actual interpreter prediction and pick whatever action
                # update

                # MODEL BASED
                # perform the rollouts  
                # train the core's DQN in the same way
                # update LSTM

                # before updating anything, must act
                
                action = agent.act(state)
                dqn_result = dqn.act(state)
                action = random.randint(0,7)
                action_vec = one_hot([action], 8)

                next_state, reward, done, _ = env.step(action)
                next_state = compress(next_state)
                next_state_vec = np.reshape(one_hot(next_state, 7), (-1,))

                prev_state = state.copy()
                state = next_state.copy()
                t += 1


                x = np.reshape(np.hstack((mb_state,mb_action)), (-1, self.state_size + self.action_size))
                y = np.reshape(np.hstack((mb_next_state,mb_reward)), (-1, self.state_size*7 + 1))
                print(self.model.sess.run(self.model.loss, {self.model.x: x, self.model.y: y}))

                print("episode: {}, score: {}".format(e, reward))


iModel = ImaginationAgent()

if __name__ == "__main__":
    iModel.sess.run(init)
    iModel.train()
    








