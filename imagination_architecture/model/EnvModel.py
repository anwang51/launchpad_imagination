import tensorflow as tf
import gym
import gym_sokoban
import numpy as np
import random

state_size = 49
action_size = 8

def env_loss(y, y_hat):


class EnvModel:
    def __init__(self, sess, state_size, action_size):
        self.sess = sess
        self.input_size = state_size + action_size
        self.output_size = state_size*7 + 1
        self.x = tf.placeholder("float32", [None, input_size])
        W1 = tf.Variable(tf.random_uniform([input_size, 20], 0, 1))
        b1 = tf.Variable(tf.random_uniform([20], 0, 1))
        l1 = tf.nn.relu(tf.matmul(self.x, W1)+b1)
        W2 = tf.Variable(tf.random_uniform([20, self.output_size], 0, 1))
        b2 = tf.Variable(tf.random_uniform([self.output_size], 0, 1))
        self.y_hat = tf.nn.relu(tf.matmul(l1, W2)+b2)
        self.y = tf.placeholder("float32", [None, self.output_size]) 
        loss = tf.losses.mean_squared_error(self.y, self.y_hat)
        self.train = tf.train.AdamOptimizer(0.001).minimize(loss)
        self.saver = tf.train.Saver()

    def update(self, prev_state, action, next_state, reward):
        # print(np.shape(prev_state))
        # print(prev_state)
        x = np.reshape(np.append(prev_state,action), (1, self.input_size))
        y = np.reshape(np.append(next_state,reward), (1, self.output_size))
        self.sess.run(self.train, {self.x: x, self.y: y})

    def predict(self, prev_state, action):
        x = prev_state.append(action)
        reward_vec = self.sess.run(self.y_hat, {self.x: x})
        state = reward_vec[:len(reward_vec) - 1]
        reward = reward_vec[len(reward_vec):]
        return state, reward

class EnvironmentNN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self._build_model()
        self.episodes = 3000
        self.max_time = 500

    def _build_model(self):
        session = tf.Session()
        return EnvModel(session, self.state_size, self.action_size)

    def load(self, name):
        self.model.saver.restore(self.model.sess, name)

    def save(self, name):
        self.model.saver.save(self.model.sess, name)

    # Should be def train(self, agent_action)
    def train(self):
        while True:
            try:
                env = gym.make('TinyWorld-Sokoban-small-v0')
            except RuntimeWarning:
                print("RuntimeWarning caught: retrying")
                continue
            except RuntimeError:
                print("RuntimeError caught: retrying")
                continue
            else:
                break
        init = tf.global_variables_initializer()
        self.model.sess.run(init)

        for e in range(self.episodes):
            while True:
                try:
                    state = env.reset()
                    state = compress(state)
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
                action = random.randint(0,7)
                next_state, reward, done, _ = env.step(action)
                self.model.update(state, action, np.reshape(tf.one_hot(next_state, 7), (-1,)), reward)
                state = compress(next_state)
                t += 1

            print("episode: {}/{}, score: {}"
                          .format(e, self.episodes, reward))
        agent.model.save_model("tfmodel_weights.h5")


def compress(state):
    new_state = []
    for block in state:
        temp = []
        for arr in block:
            if arr[0] == 0:
                temp.append(0)
            elif arr[0] == 243:
                temp.append(1)
            elif arr[0] == 254:
                if arr[1] == 126:
                    temp.append(2)
                if arr[1] == 95:
                    temp.append(3)
            elif arr[0] == 142:
                temp.append(4)
            elif arr[0] == 160:
                temp.append(5) 
            elif arr[0] == 219:
                temp.append(6)   
        new_state.append(temp) 
    new_state = np.array(new_state)
    new_state = new_state.flatten()
    return new_state


nn = EnvironmentNN(state_size, action_size)

def load_agent():
    nn.load(savefile)

if __name__ == "__main__":
    nn.train()
    nn.save(savefile)








