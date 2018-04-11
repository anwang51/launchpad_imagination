import tensorflow as tf
import numpy as np
gamma = 0.9
class ICNet:
	def __init__(sess, input_size, action_num):
		self.sess = sess
		self.x = tf.placeholder("float", [None, input_size])
		W1 = tf.Variable("float", tf.random_uniform([input_size, 20], -0.5, 0.5))
		b1 = tf.Variable("float", tf.random_uniform([20], -0.5, 0.5))
		l1 = tf.nn.relu(tf.matmul(x, W1)+b1)
		W2 = tf.Variable("float", tf.random_uniform([20, action_num], -0.5, 0.5))
		b2 = tf.Variable("float", tf.random_uniform([action_num], -0.5, 0.5))
		self.y_hat = tf.nn.relu(tf.matmul(l1, W2)+b2)
		self.y = tf.placeholder("float", [None, action_num])
		self.q_val = tf.placeholder("float", [None])
		self.actions = tf.placeholder("float", [None, action_num])
		q_val_hat = tf.reduce_sum(tf.multiply(self.y_hat, self.actions), 1)
		loss = tf.reduce_sum(tf.square(self.q_val - q_val_hat))
		self.train = tf.train.AdamOptimizer().minimize(loss)

	def update(state, action, reward, next_state):
		reward_vecs = self.sess.run(self.y_hat, {self.x: next_state})
		q_vals = reward + gamma*np.amax(reward_vecs, 1)
		self.sess.run(self.train, {self.x: state, self.q_val: q_vals, self.actions: action})

	def action(state):
		reward_vec = self.sess.run(self.y_hat, {self.x: state})
		return np.argmax(reward_vec)






