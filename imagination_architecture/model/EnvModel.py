import tensorflow as tf

class EnvModel:

	def __init__(self):
        self.sess = sess
        self.x = tf.placeholder("float32", [None, input_size])
        W1 = tf.Variable(tf.random_uniform([input_size, 20], 0, 1))
        b1 = tf.Variable(tf.random_uniform([20], 0, 1))
        l1 = tf.nn.relu(tf.matmul(self.x, W1)+b1)
        W2 = tf.Variable(tf.random_uniform([20, action_size], 0, 1))
        b2 = tf.Variable(tf.random_uniform([action_size], 0, 1))
        self.y_hat = tf.nn.relu(tf.matmul(l1, W2)+b2)
        self.y = tf.placeholder("float32", [None]) #Proper q-vals as calculated by the bellman equation
        loss = tf.losses.mean_squared_error(self.y, self.y_hat)
        self.train = tf.train.AdamOptimizer(0.001).minimize(loss)
        self.saver = tf.train.Saver()

    def update(self, prev_state, action, next_state, reward):
    	x = prev_state.append(action)
    	y = next_state.append(reward)
        self.sess.run(self.train, {self.x: x, self.y = y})

    def predict(self, prev_state, action):
    	x = prev_state.append(action)
        reward_vec = self.sess.run(self.y_hat, {self.x: x})
        state = reward_vec[:len(reward_vec) - 1]
        reward = reward_vec[len(reward_vec):]
        return state, reward

class EnvironmentNN:

	def __init__(self):
		self.model = self._build_model()

	def _build_model(self):
        session = tf.Session()

		return model