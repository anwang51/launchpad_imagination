from DQN import DQNAgent
import gym
import gym_sokoban
import numpy as np
import tensorflow as tf

class ImaginationCore:

	def __init__(self, agent, env, input_height, input_width, action_size):
		# self.env = EnvironmentModel()
		self.env = copy.deepcopy(env)
		self.input_height = input_height
		self.input_width = input_width
		self.action_size = action_size
		self.start_state = self.env.state
		# self.start_state = np.reshape(self.start_state, [self.input_width, self.input_height])
		self.actor = agent # DQNAgent

	def rollout_single(self, action):
		next_state, reward, done, _ = self.env.step(action) # make sure env outputs pixels, otherwise we're fucked
		# next_state = np.reshape(next_state, [self.input_width, self.input_height])
		return [next_state, reward]

	def rollout(self, state, depth=5):
		result = []
		for i in range(action_size):
			temp_env = copy.deepcopy(self.env)
			temp_depth = depth - 1
			rollout_result = []
			next_state, reward = self.rollout_single(state, i)
			rollout_result.append([next_state, reward])
			while depth > 0:
				action = self.actor.act(next_state)
				next_state, reward = self.rollout_single(next_state, action)
				rollout_result.append([next_state, reward])
				temp_depth -= 1
			result.append(rollout_result)
		return np.array(result)

