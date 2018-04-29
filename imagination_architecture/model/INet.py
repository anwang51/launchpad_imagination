import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import random
import DQN
import EnvModelBatch
import ImaginationCore

class INet:
	def __init__(self, LSTM_input_size, num_paths, MF_input_size, output_size, path_length):
		lstm_layer = rnn.core_rnn_cell.BasicLSTMCell(LSTM_input_size,forget_bias=1)
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
		self.q_val_hat = tf.reduce_sum(tf.multiply(self.output, self.actions), axis=1)
		self.loss = tf.losses.mean_squared_error(self.q_val, self.q_val_hat)
		self.optimizer = tf.train.AdamOptimizer(0.001)
		self.train = self.optimizer.minimize(self.loss)

		self.saver = tf.train.Saver(max_to_keep = 5, keep_checkpoint_every_n_hours =1)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

	def act(self, paths, MF_output):
		reward_vec = self.sess.run(self.output, {self._paths: paths, self._MF_input: MF_input})
		return np.argmax(reward_vec)

	def update(self, paths, MF_output, action, reward, next_paths, next_MF_output, done):
		reward_vec = self.sess.run(self.output, {self._paths: next_paths, self._MF_input: next_MF_input})
		q_val = reward + self.gamma*np.amax(self.act(next_paths, next_MF_output))*(1-done)
		self.sess.run(self.train, {self._paths: paths, self._MF_output : MF_output, self.q_val: q_val, self.action: action})

	def restore_session(self):
        path = tf.train.get_checkpoint_state('./FinalCheckpoints/')
        if path is None:
            raise IOError('No checkpoint to restore in ' + './FinalCheckpoints/')
        else:
            self.saver.restore(self.sess, path.model_checkpoint_path)
        


	def train(self, state_size, action_size, restore_session = False):
		dqn = DQN.DQNAgent(600, 400, action_size)		
		
		if restore_session:
            self.restore_session()

		env = gym.make('CartPole-v1')
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
	        e = 0
	        while not done:
            # MODEL FREE
            # store prediction
            # get the actual interpreter prediction and pick whatever action
            # update

            # MODEL BASED
            # perform the rollouts  
            # train the core's DQN in the same way
            # update LSTM
            dqn_predict = dqn.action(state)
            icore = ImaginationCore.ImaginationCore(dqn, env, state_size, action_size)
            
            # before updating anything, must act
            

            e += 1
            print("episode: {}, score: {}".format(e, reward))
	       	if epis % 1000 == 0:
	       		saver.save(self.model.sess, './FinalCheckpoints/'+'model')
	        	print('Model {} saved'.format(epis))

	def update(self, rollouts, model_free_output):
		pass

if __name__ == "__main__":
	model = INet("size of output of envmodel", "action size", "model free output", "action size", "rollouts")