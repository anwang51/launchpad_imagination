import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
class ICNet:
	def __init__(sess, input_size, action_num):
		self.x = tf.placeholder("float", [None, input_size])
		W1 = tf.Variable("float", [input_size, 20])
		b1 = tf.Variable("float", [20])
		l1 = tf.nn.relu(tf.matmul(x, W1)+b1)
		W2 = tf.Variable("float", [20, action_num])
		b2 = tf.Variable("float", [action_num])
		self.y_hat = tf.nn.relu(tf.matmul(l1, W2)+b2)
		self.labels = tf.placeholder("float", [None])
		self.actions = tf.placeholder("float", [None, action_num])
		q_vals = tf.reduce_sum(tf.multiply(self.y_hat, self.actions), 1)
		self.loss = tf.reduce_sum(tf.square(self.labels - q_vals))
		
