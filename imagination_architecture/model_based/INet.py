import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

class INet:
	def __init__(self, LSTM_input_size, num_paths, MF_input_size, output_size, path_length, sess):
		self.sess = sess

		#lstm_layer = rnn.BasicLSTMCell(LSTM_input_size,forget_bias=1)
		lstm_layer = rnn.core_rnn_cell.BasicLSTMCell(LSTM_input_size,forget_bias=1)
		#Batch_size, path_length, LSTM.input_size
		self._paths = tf.placeholder("float", [None, path_length, LSTM_input_size])
		unstacked = tf.unstack(self._paths, None, 1)
		outputs, states = rnn.static_rnn(lstm_layer, unstacked, dtype="float")

		input_matrix = outputs[-1]
		input_pieces = tf.split(input_matrix, num_paths, 0)
		self._MF_output = tf.placeholder("float", [None, MF_input_size])
		input_pieces.append(self._MF_output)
		x = tf.concat(input_pieces, 1)

		W1 = tf.get_variable('W1', [num_paths*LSTM_input_size+MF_input_size, 40], initializer=tf.contrib.layers.xavier_initializer())
		b1 = tf.get_variable('b1', [40], initializer=tf.contrib.layers.xavier_initializer())
		l1 = tf.nn.relu(tf.matmul(x, W1)+b1)
		W2 = tf.get_variable('W2', [40, output_size], initializer=tf.contrib.layers.xavier_initializer())
		b2 = tf.get_variable('b2', [output_size], initializer=tf.contrib.layers.xavier_initializer())
		self.output = tf.matmul(l1, W2)+b2


		self.q_val = tf.placeholder("float32", [None]) #Proper q-vals as calculated by the bellman equation
		self.actions = tf.placeholder("float32", [None, output_size]) #Actions stored as one-hot vectors
		self.q_val_hat = tf.reduce_sum(tf.multiply(self.output, self.actions), axis=1) #The q-vals for the actions selected in game
		#loss = tf.reduce_sum(tf.square(self.q_val - q_val_hat))
		self.loss = tf.losses.mean_squared_error(self.q_val, self.q_val_hat)
		self.optimizer = tf.train.AdamOptimizer(0.001)
		self.train = self.optimizer.minimize(self.loss)
		self.saver = tf.train.Saver()

		#def remember(state, action, reward, next_state, done):

	def act(self, paths, MF_output):
		reward_vec = self.sess.run(self.output, {self._paths: paths, self._MF_input: MF_input})
		return np.argmax(reward_vec)

	def update(self, paths, MF_output, action, reward, next_paths, next_MF_output, done):
		reward_vec = self.sess.run(self.output, {self._paths: next_paths, self._MF_input: next_MF_input})
		q_val = reward + self.gamma*np.amax(self.act(next_paths, next_MF_output))*(1-done)
		self.sess.run(self.train, {self._paths: paths, self._MF_output : MF_output, self.q_val: q_val, self.action: action})

if __name__ == "__main__":
	print("compiling")
	model = INet(15, 4, 5, 5, 4 tf.Session())