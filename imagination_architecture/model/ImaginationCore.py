from DQN import DQNAgent
import gym
import gym_sokoban
import numpy as np
import tensorflow as tf

class ImaginationCore:

	def __init__(self, agent, env, state_size, action_size):
		# self.env = EnvironmentModel()
		self.env = copy.deepcopy(env)
		self.state_size = state_size
		self.action_size = action_size
		self.start_state = self.env.state
		self.start_state = np.reshape(self.start_state, [1, self.state_size])
		self.actor = agent

	def rollout_single(self, state, action):
		next_state, reward, done, _ = self.env.step(action) # make sure env outputs pixels, otherwise we're fucked
		next_state = np.reshape(next_state, [1, self.state_size])
		return [next_state, reward]

	def rollout(self, state, depth=5):
		result = []
		for i in range(action_size):
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

