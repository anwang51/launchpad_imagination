import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

class INet:
	def __init__(LSTM_input_size, num_paths, MF_input_size, output_size, sess):
		self.sess = sess
		self.input_size = input_size
		lstm_layer=rnn.BasicLSTMCell(input_size,forget_bias=1)

		self._paths = (tf.placeholder("float", [None, None, LSTM_input_size]) for _ in range(num_paths)) #Batch_size, path_length, LSTM.input_size
		input_pieces = [rnn.static_rnn(lstm_layer,tf.unstack(path, None, 1),dtype="float")[-1] for path in self._paths]
		self._MF_output = tf.placeholder("float", [None, MF_input_size])
		input_pieces.add(self._MF_output)
		x = tf.concat(input_pieces, 1)
		W1 = tf.Variable("float", tf.random_uniform([None, num_paths*LSTM_input_size+MF_input_size, 20], -0.5, 0.5))
		b1 = tf.Variable("float", [None, 20])
		l1 = tf.nn.relu(tf.add(tf.matmul(x, W1), b1))
		W2 = tf.Variable("float", [None, 20, output_size])
		b2 = tf.Variable("float", [None, output_size])
		self.output = tf.nn.relu(tf.add(tf.matmul(l1, W2), b2))
		self._y = tf.placeholder("float", [None])
		loss = tf.sum(tf.square(self.output - _y))
		self.train = tf.train.AdamOptimizer().minimize(loss)

	def remember(state, action, reward, next_state, done):

	def act(paths, MF_output):
		return self.sess.run(self.output, {self._paths: paths, self._MF_input: MF_input})

	def update(paths, MF_output, action, reward, next_paths, next_MF_output):
		output = self.act(next_paths, next_MF_output)
		better_val = reward + self.gamma*np.max(self.act(next_paths, next_MF_output))
		self.sess.run(self.train, {self._paths: paths, self._MF_input})
