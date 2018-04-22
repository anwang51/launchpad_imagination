from DQN import DQNAgent
import gym
import gym_sokoban
import numpy as np
import tensorflow as tf

class ImaginationCore:

	def __init__(self, environment_string, state_size, action_size):
		# self.env = EnvironmentModel()
		self.env = gym.make(environment_string)
		self.state_size = state_size
		self.action_size = action_size
		self.start_state = self.env.reset()
		self.start_state = np.reshape(self.start_state, [1, self.state_size])
		self.actor = DQNAgent(self.state_size, self.action_size, self.env)
		init = tf.global_variables_initializer()
		self.actor.model.sess.run(init)

	def rollout_single(self, state):
		action = self.actor.act(state)
		next_state, reward, done, _ = self.env.step(action)
		next_state = np.reshape(next_state, [1, self.state_size])
		return next_state, reward

	def rollout(self, state, depth=3):
		rw = 0
		st = state
		out = [(st, rw)]
		while depth > 0:
			st, rw = self.rollout_single(st)
			print(st)
			print(np.shape(st))
			out.append((st, rw))
			depth -= 1
		return out

# Example:
# im_core = ImaginationCore('Sokoban-small-v0', 37632, 8)
# print(im_core.start_state)
# print(np.shape(im_core.start_state))
# print(im_core.rollout(im_core.start_state))
