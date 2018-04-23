import tensorflow as tf
import gym
import gym_sokoban
import numpy as np
import random

state_size = 49
action_size = 8

def softmax(vec):
    return np.exp(x) / np.sum(np.exp(x), axis=0)    

def env_loss(y, y_hat):
    alpha = 1
    beta = 10
    reward = y[:,-1]
    reward_hat = y_hat[:,-1]
    reward_loss = tf.losses.mean_squared_error(reward, reward_hat)

    class_true = y[:,:-1]
    class_hat = y_hat[:,:-1]
    class_mat = tf.reshape(class_true, (-1, 49, 7))
    class_mat_hat = tf.reshape(class_hat, (-1, 49, 7))
    # y_hat_mat = [softmax(row) for row in y_hat_mat]
    cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=class_mat_hat, labels=class_mat)
    # cross_entropy_loss = tf.reduce_sum(cross_entropy_loss)
    cross_entropy_loss = tf.reduce_mean(cross_entropy_loss)
    combined_loss = tf.add(tf.scalar_mul(alpha, cross_entropy_loss), tf.scalar_mul(beta, reward_loss))
    return combined_loss

def one_hot(arr, dim):
    out = []
    for elem in arr:
        temp = [0]*dim
        temp[elem] = 1
        out.append(temp)
    return np.array(out)

class EnvModel:
    def __init__(self, sess, state_size, action_size):
        self.sess = sess
        self.input_size = state_size + action_size
        self.output_size = state_size*7 + 1
        self.x = tf.placeholder("float32", [None, self.input_size])
        W1 = tf.Variable(tf.random_uniform([self.input_size, 600], 0, 1))
        b1 = tf.Variable(tf.random_uniform([600], 0, 1))
        l1 = tf.nn.elu(tf.matmul(self.x, W1)+b1)

        W2 = tf.Variable(tf.random_uniform([600, 600], 0, 1))
        b2 = tf.Variable(tf.random_uniform([600], 0, 1))
        l2 = tf.nn.elu(tf.matmul(l1, W2)+b2)

        W3 = tf.Variable(tf.random_uniform([600, self.output_size], 0, 1))
        b3 = tf.Variable(tf.random_uniform([self.output_size], 0, 1))
        self.y_hat = tf.nn.elu(tf.matmul(l2, W3)+b3)

        self.y = tf.placeholder("float32", [None, self.output_size]) 
        # loss = tf.losses.mean_squared_error(self.y, self.y_hat)
        self.loss = env_loss(self.y, self.y_hat)
        self.train = tf.train.AdamOptimizer(0.001).minimize(self.loss)
        self.saver = tf.train.Saver()

    def update(self, prev_state, action, next_state, reward):
        # print(np.shape(prev_state))
        # print(prev_state)
        x = np.reshape(np.hstack((prev_state,action)), (-1, self.input_size))
        y = np.reshape(np.hstack((next_state,reward)), (-1, self.output_size))
        # print("x: ", x)
        # print("y: ", y)
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
        self.episodes = 1e100
        self.max_time = 500

    def _build_model(self):
        session = tf.Session()
        return EnvModel(session, self.state_size, self.action_size)

    def load(self, name):
        self.model.saver.restore(self.model.sess, name)

    def save(self, name):
        self.model.saver.save(self.model.sess, name)

    def verify(self):
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
            action = random.randint(0,7)
            action_vec = one_hot([action], 8)
            next_state, reward, done, _ = env.step(action)
            print("reward: ", reward)
            print("next state: ", compress(next_state))
            next_state = compress(next_state)
            next_state_vec = np.reshape(one_hot(next_state, 7), (-1,))
            #self.model.update(state, action_vec, np.reshape(one_hot(next_state, 7), (-1,)), reward)
            # print(self.sess.run())
            x = np.reshape(np.append(prev_state,action_vec), (1, self.state_size + self.action_size))
            y = np.reshape(np.append(next_state_vec,reward), (1, self.state_size*7 + 1))
            predicted_state, loss = self.model.sess.run([self.model.y_hat, self.model.loss], {self.model.x: x, self.model.y: y})
            predicted_state = predicted_state[0]
            print("predicted reward: ", predicted_state[-1])
            predicted_state = predicted_state[:-1]
            predicted_state = np.reshape(predicted_state, (-1, 7))
            predicted_state = np.argmax(predicted_state, axis = 1)
            print("predicted_state: ", predicted_state)
            print("predicted loss: ", loss)
            prev_state = state
            state = next_state
            t += 1

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
                    prev_state = state

                except RuntimeWarning:
                    print("RuntimeWarning caught: retrying")
                    continue
                except RuntimeError:
                    print("RuntimeError caught: retrying")
                    continue
                else:
                    break

            batch = []
            done = False
            t = 0
            while not done and t < self.max_time:
                action = random.randint(0,7)
                action_vec = one_hot([action], 8)
                next_state, reward, done, _ = env.step(action)
                next_state = compress(next_state)
                next_state_vec = np.reshape(one_hot(next_state, 7), (-1,))
                # reward = np.array(reward)

                # print("state ", state)
                # print("action vec ", action_vec)
                # print("next state ", next_state)
                # print("reward ", reward)

                batch.append(np.array([state, action_vec, next_state_vec, reward]))
                #self.model.update(state, action_vec, next_state_vec, reward)
                # print(self.sess.run())
                prev_state = state.copy()
                state = next_state.copy()
                t += 1

            if len(batch) >= 30:
                batch = np.array(batch)
                np.random.shuffle(batch)
                minibatch= np.array(batch[:30])

                mb_state = np.array(minibatch[:,0])
                mb_state = np.array([np.array(lst) for lst in mb_state])

                mb_action = np.array(minibatch[:,1])
                mb_action = np.array([np.array(lst[0]) for lst in mb_action])

                mb_next_state = np.array(minibatch[:,2])
                mb_next_state = np.array([np.array(lst) for lst in mb_next_state])

                mb_reward = np.array(minibatch[:,3])
                mb_reward = np.array([np.array([item]) for item in mb_reward])

                # print("update state ", mb_state)
                # print("update action vec ", mb_action)
                # print("update next state ", mb_next_state)
                # print("update reward ", mb_reward)

                # print("update state shape ", mb_state.shape)
                # print("update action vec shape ", mb_action.shape)
                # print("update next state shape ", mb_next_state.shape)
                # print("update reward shape ", mb_reward.shape)

                self.model.update(mb_state, mb_action, mb_next_state, mb_reward)

                x = np.reshape(np.hstack((mb_state,mb_action)), (-1, self.state_size + self.action_size))
                y = np.reshape(np.hstack((mb_next_state,mb_reward)), (-1, self.state_size*7 + 1))
                print(self.model.sess.run(self.model.loss, {self.model.x: x, self.model.y: y}))
                print("episode: {}/{}, score: {}"
                              .format(e, self.episodes, reward))
            else:
                pass

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
    nn.verify()
