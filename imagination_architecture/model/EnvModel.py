import tensorflow as tf
import gym
import gym_sokoban
import numpy as np

class EnvModel:
	def __init__(self, sess, input_size):
        self.sess = sess
        self.x = tf.placeholder("float32", [None, input_size])
        W1 = tf.Variable(tf.random_uniform([input_size, 20], 0, 1))
        b1 = tf.Variable(tf.random_uniform([20], 0, 1))
        l1 = tf.nn.relu(tf.matmul(self.x, W1)+b1)
        W2 = tf.Variable(tf.random_uniform([20, input_size], 0, 1))
        b2 = tf.Variable(tf.random_uniform([input_size], 0, 1))
        self.y_hat = tf.nn.relu(tf.matmul(l1, W2)+b2)
        self.y = tf.placeholder("float32", [None]) 
        loss = tf.losses.mean_squared_error(self.y, self.y_hat)
        self.train = tf.train.AdamOptimizer(0.001).minimize(loss)
        self.saver = tf.train.Saver()

    def update(self, prev_state, action, next_state, reward):
    	x = prev_state.append(action)
    	y = next_state.append(reward)
        self.sess.run(self.train, {self.x: x, self.y: y})

    def predict(self, prev_state, action):
    	x = prev_state.append(action)
        reward_vec = self.sess.run(self.y_hat, {self.x: x})
        state = reward_vec[:len(reward_vec) - 1]
        reward = reward_vec[len(reward_vec):]
        return state, reward

class EnvironmentNN:
	def __init__(self, state_size):
        self.input_size = input_size + 1
        self.model = self._build_model()
        self.episodes = 3000

    def _build_model(self):
        session = tf.Session()
        return EnvModel(session, self.input_size)

    def load(self, name):
        self.model.saver.restore(self.model.sess, name)

    def save(self, name):
        self.model.saver.save(self.model.sess, name)

    # Should be def train(self, agent_action)
    def train(self):
        env = gym.make('TinyWorld-Sokoban-small-v0')
        print("env stuff", env.observation_space, env.action_space)
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
            while not done:
                action = random.randint(0,7)
                next_state, reward, done = env.step(action)
                next_state = compress(next_state)
                self.model.update(state, action, next_state, reward)
                state = next_state

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
    new_state.flatten()
    return new_state

nn = EnvironmentNN(49)

def load_agent():
    nn.load(savefile)

if __name__ == "__main__":
    nn.train()
    nn.save(savefile)








