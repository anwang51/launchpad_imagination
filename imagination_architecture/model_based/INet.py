import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

class INet:
	def __init__(LSTM_input_size, num_paths, MF_input_size, output_size):
		self.input_size = input_size
		lstm_layer=rnn.BasicLSTMCell(input_size,forget_bias=1)

		self._paths = (tf.placeholder("float", [None, None, LSTM_input_size]) for _ in range(num_paths)) #Batch_size, path_length, LSTM.input_size
		input_pieces = [rnn.static_rnn(lstm_layer,tf.unstack(path, None, 1),dtype="float")[-1] for path in self._paths]
		self._MF_input = tf.placeholder("float", [None, MF_input_size])
		input_pieces.add(self._MF_input)
		x = tf.concat(input_pieces, 1)
		W1 = tf.Variable("float", [None, num_paths*LSTM_input_size+MF_input_size, 20])
		b1 = tf.Variable("float", [None, 20])
		l1 = tf.nn.relu(tf.add(tf.matmul(x, W1), b1))
		W2 = tf.Variable("float", [None, 20, output_size])
		b2 = tf.Variable("float", [None, output_size])
		self.output = tf.nn.relu(tf.add(tf.matmul(l1, W2), b2))
		self._QVal = tf.Variable("float", [])
		loss = tf.square(tf.max(self.output, 1) - _QVal)
		self.optimizer = tf.train.AdamOptimizer().minimize(loss)

	def remember(state, action, reward, next_state, done):

	def act(paths, MF_output):
		init_op = tf.initialize_all_variables()
		with tf.Session() as sess:
			sess.run(init_op)
			return sess.run(self.output, {self._paths: paths, self._MF_input})

	def update(paths, MF_output, action, reward, next_paths, next_MF_output):
		better_val = reward + self.gamma*np.max(self.act(next_paths, next_MF_output))
		init_op = tf.initialize_all_variables()
		with tf.Session() as sess:
			sess.run(init_op)
			sess.run(self.optimizer, {self._paths: paths, self._MF_input})

	def save():

	def load():
