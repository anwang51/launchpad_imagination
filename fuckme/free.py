import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import numpy as np
import random

# Deep Q-learning Agent
class ModelFree:
    def __init__(self, state_size, action_size):
        self.input_size = state_size
        self.output_size = action_size
        self.hidden_size = 24
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.build_model()

    def build_model(self):
        # Neural Net for Deep-Q learning Model
        self.f_1 = tf.nn.relu
        self.f_2 = tf.nn.relu

        self.x = tf.placeholder(tf.float32, [None, self.input_size])
        self.W_1 = tf.Variable(tf.random_normal([self.input_size, self.hidden_size]))
        self.b_1 = tf.Variable(tf.random_normal([self.hidden_size]))
        self.W_2 = tf.Variable(tf.random_normal([self.hidden_size, self.output_size]))
        self.b_2 = tf.Variable(tf.random_normal([self.output_size]))
        self.y_hat = self.f_2(tf.matmul(self.f_1(tf.matmul(self.x, self.W_1)+self.b_1), self.W_2)+self.b_2)

        self.y_ = tf.placeholder(tf.float32, [None, output_size])
        self.mse = tf.losses.mean_squared_error(self.y_, self.y_hat) 
        self.train_step = tf.train.AdamOptimizer(.001).minimize(mse)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.output_size)
        return self.f_2(tf.matmul(self.f_1(tf.matmul(self.x, self.W_1)+self.b_1), self.W_2)+self.b_2)
        #return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        sess.run(train_step, feed_dict={x: np.array(data[0]).reshape((3,1)).T, y_: np.array(data[1]).reshape((1,1)).T})
        ####HOW DO WE DO THIS PART LMAO###
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













