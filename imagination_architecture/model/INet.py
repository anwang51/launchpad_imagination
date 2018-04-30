import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import random
import DQN
import EnvModelBatch
import ImaginationCore

class INet:
    def __init__(self, LSTM_input_size, num_paths, MF_input_size, output_size, path_length):
        tf.reset_default_graph()
        lstm_layer = rnn.core_rnn_cell.BasicLSTMCell(LSTM_input_size,forget_bias=1)
        self._paths = tf.placeholder("float", [None, path_length, LSTM_input_size])
        unstacked = tf.unstack(self._paths, None, 1)
        outputs, states = rnn.static_rnn(lstm_layer, unstacked, dtype="float")
        self.memory = []

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

        self.dqn = DQN.DQNAgent(input_width, input_height, action_size) 

    def act(self, paths, MF_output):
        reward_vec = self.sess.run(self.output, {self._paths: paths, self._MF_input: MF_input})
        return np.argmax(reward_vec)

    def update(self, paths, MF_output, action, reward, next_paths, next_MF_output, done):
        paths = format_paths(paths)
        next_paths = format_paths(next_paths)
        reward_vec = self.sess.run(self.output, {self._paths: next_paths, self._MF_input: next_MF_input})
        q_val = reward + self.gamma*np.amax(self.act(next_paths, next_MF_output))*(1-done)
        self.sess.run(self.train, {self._paths: paths, self._MF_output : MF_output, self.q_val: q_val, self.action: action})

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
                icore = ImaginationCore.ImaginationCore(dqn, curr_cloned_state, input_width, input_height, action_size)
                rollouts = icore.rollout()

                curr_dqn_predict = dqn.action(state)
                lstm_out = self.act(rollouts, curr_dqn_predict)
          
                next_state, reward, done, _ = env.step(lstm_out) 
                next_dqn_predict = dqn.action(next_state)

                self.memory.append([curr_cloned_state, curr_dqn_predict, lstm_out, reward, env.clone_full_state, done])
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
        actions = []
        rewards = []
        next_states = []
        dones = []
        for tup in minibatch:
            cur_IC = ImaginationCore.ImaginationCore(self.dqn, tup[0], input_width, input_height, action_size)
            states.append([cur_IC.rollout(), tup[1]])
            actions.append(tup[2])
            rewards.append(tup[3])
            next_IC = ImaginationCore.ImaginationCore(self.dqn, tup[4], input_width, input_height, action_size)
            states.append([next_IC.rollout(), tup[5]])
            dones.append(tup[6])
        states = np.array(states)
        print("shape", states.shape)
        actions = np.eye(self.action_size)[actions]
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        self.model.update(states, actions, rewards, next_states, dones)

 
