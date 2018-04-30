import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import random
import DQN
import ImaginationCore


class StateProcessor():
	"""
	Processes raw Atari images. Resizes it to 84x84 and converts it to grayscale.

	"""
	def __init__(self):
		# Builds the Tensorflow graph
		with tf.variable_scope("state_processor"):
			self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8) #
			self.output = tf.image.rgb_to_grayscale(self.input_state)
			self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
			self.output = tf.image.resize_images(
				self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
			self.output = tf.squeeze(self.output)

	def process(self, sess, state):
		"""
		Args:
			sess: A Tensorflow session object
			state: A [210, 160, 3] Atari RGB State
		Returns:
			A processed [84, 84, 1] state representing grayscale values.
		"""
		return sess.run(self.output, { self.input_state: state })

class INet:
	def __init__(self, LSTM_input_size, num_paths, MF_output_size, output_size, path_length):
		tf.reset_default_graph()
		#with tf.Graph.as_default():
		lstm_layer = rnn.core_rnn_cell.BasicLSTMCell(LSTM_input_size,forget_bias=1)
		#Batch_size, path_length, LSTM.input_size
		self._paths = tf.placeholder("float32", [None, num_paths, path_length, LSTM_input_size])
		paths_list = tf.unstack(self._paths, None, 1)
		#unstacked = [tf.unstack(path, None, 1) for path in paths_list]
		with tf.variable_scope("encoder", reuse=None):
			outputs, states = tf.nn.dynamic_rnn(lstm_layer, paths_list[0], dtype="float32")
		input_pieces = [outputs[-1]]
		with tf.variable_scope("encoder", reuse=True):
			for path in paths_list[1:]:
				outputs, states = tf.nn.dynamic_rnn(lstm_layer, path, dtype="float32")
				input_pieces.append(outputs[-1])
		self._MF_output = tf.placeholder("float32", [None, MF_output_size])
		input_pieces.append(self._MF_output)
		x = tf.concat(input_pieces, 1)

		W1 = tf.get_variable('W1', [num_paths*LSTM_input_size+MF_output_size, 40], initializer=tf.contrib.layers.xavier_initializer())
		b1 = tf.get_variable('b1', [40], initializer=tf.contrib.layers.xavier_initializer())
		l1 = tf.nn.relu(tf.matmul(x, W1)+b1)
		W2 = tf.get_variable('W2', [40, output_size], initializer=tf.contrib.layers.xavier_initializer())
		b2 = tf.get_variable('b2', [output_size], initializer=tf.contrib.layers.xavier_initializer())
		self.output = tf.matmul(l1, W2)+b2


		self.q_val = tf.placeholder("float32", [None]) #Proper q-vals as calculated by the bellman equation
		self.actions = tf.placeholder("float32", [None, output_size]) #Actions stored as one-hot vectors
		self.q_val_hat = tf.reduce_sum(tf.multiply(self.output, self.actions), axis=1)
		self.loss = tf.losses.mean_squared_error(self.q_val, self.q_val_hat)
		self.optimizer = tf.train.AdamOptimizer(0.001)
		self.train = self.optimizer.minimize(self.loss)

		#self.saver = tf.train.Saver(max_to_keep = 5, keep_checkpoint_every_n_hours =1)
		self.sess = tf.Session()
		
		print("on other side")
		self.dqn = DQN.DQNAgent(84, 84, num_paths, sess=self.sess)
		print("on other other side")
		self.sess.run(tf.global_variables_initializer())
		
		self.processor = StateProcessor()
		print("other to the third")


		print("other to the fourth")

	def act(self, paths, MF_output):
		reward_vec = self.sess.run(self.output, {self._paths: paths, self._MF_output: MF_output})
		return np.argmax(reward_vec)

	def update(self, states, action, reward, next_states, done):
		paths = states[0]
		MF_input = states[1]
		next_paths = next_states[0]
		next_MF_input = next_states[1]
		paths = format_paths(paths)
		next_paths = format_paths(next_paths)
		reward_vec = self.sess.run(self.output, {self._paths: next_paths, self._MF_input: next_MF_input})
		q_val = reward + self.gamma*np.amax(self.act(next_paths, next_MF_output))*(1-done)
		self.sess.run(self.train, {self._paths: paths, self._MF_output : MF_output, self.q_val: q_val, self.action: action})

	def format_paths(paths):
		inputs = []
		for batch in paths:
			rollouts = []
			for path in batch:
				state_list = []
				for tup in path:
					state = tup[0]
					#state = self.processor.process(self.sess, state)
					state = np.concatenate([np.flatten(state), tup[1]])
					state_list.append(state)
				rollouts.append(state_list)
			inputs.append(rollouts)
		return np.array(inputs)

	def restore_session(self):
		path = tf.train.get_checkpoint_state('./FinalCheckpoints/')
		if path is None:
			raise IOError('No checkpoint to restore in ' + './FinalCheckpoints/')
		else:
			self.saver.restore(self.sess, path.model_checkpoint_path)

	def train(self, input_width, input_height, action_size, restore_session = False):   
		if restore_session:
			self.restore_session()

		env = gym.make('Breakout-v0')
		e = 0
		while True:
			while True:
				try:
					state = env.reset()
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
				curr_cloned_state = env.clone_full_state()
				icore = ImaginationCore.ImaginationCore(dqn, curr_cloned_state, input_width, input_height, action_size, self.processor)
				rollouts = icore.rollout()

				curr_dqn_predict = dqn.action(state)
				lstm_out = self.act(rollouts, curr_dqn_predict)
		  
				next_state, reward, done, _ = env.step(lstm_out) 
				next_dqn_predict = dqn.action(next_state)

				self.memory.append([curr_cloned_state, curr_dqn_predict, lstm_out, reward, env.clone_full_state(), done])
				dqn.remember(state, lstm_out, reward, next_state, done)

				state = next_state

			e += 1
			num_mem = len(dqn.memory)
			if num_mem > 32:
				num_mem = 32
			dqn.replay(num_mem)
			self.replay(num_mem)

			print("episode: {}, score: {}".format(e, reward))
			if e % 1000 == 0:
				saver.save(self.model.sess, './FinalCheckpoints/'+'model')
				print('Model {} saved'.format(e))

	def replay(self, batch_size):
		minibatch = random.sample(self.memory, batch_size)
		states = []
		MF_outputs = []
		actions = []
		rewards = []
		next_states = []
		next_MF_outputs = []
		dones = []
		for tup in minibatch:
			cur_IC = ImaginationCore.ImaginationCore(self.dqn, tup[0], input_width, input_height, action_size, self.processor)
			states.append(cur_IC.rollout())
			MF_outputs.append(tup[1])
			actions.append(tup[2])
			rewards.append(tup[3])
			next_IC = ImaginationCore.ImaginationCore(self.dqn, tup[4], input_width, input_height, action_size, self.processor)
			next_states.append(next_IC.rollout())
			next_MF_outputs.append(tup[5])
			dones.append(tup[6])
		states = tup([states, np.array(MF_outputs)])
		next_states = tup([next_states, np.array(next_MF_outputs)])
		print("shape", states.shape)
		actions = np.eye(self.action_size)[actions]
		rewards = np.array(rewards)
		dones = np.array(dones)
		self.model.update(states, actions, rewards, next_states, dones)

 
