from DQN import DQNAgent
import gym
import numpy as np
import tensorflow as tf

class ImaginationCore:
	def __init__(self, agent, cloned_state, action_size, processor):
		self.env = gym.make('Breakout-v0')
		self.env.reset()
		self.cloned_state = cloned_state
		self.action_size = action_size
		# self.start_state = np.reshape(self.start_state, [self.input_width, self.input_height])
		self.actor = agent # DQNAgent
		self.processor = processor

	def rollout_single(self, action):
		next_state, reward, done, _ = self.env.step(action) # make sure env outputs pixels, otherwise we're fucked
		# next_state = np.reshape(next_state, [self.input_width, self.input_height])
		return [next_state, reward]

	def rollout_four(self, action):
		#next_states = []
		reward_sum = 0
		for _ in range(4):
			next_state, reward, done, _ = self.env.step(action) # make sure env outputs pixels, otherwise we're fucked
			#next_states.append(processor.process(next_state))
			reward_sum += reward
		# next_state = np.reshape(next_state, [self.input_width, self.input_height])
		#next_state = np.concatenate(next_states, 2)
		return (self.processor.process_2(next_state), reward_sum)

	def rollout(self, depth=5):
		result = []
		for i in range(self.action_size):
			self.env.env.restore_full_state(self.cloned_state)
			temp_depth = depth - 1
			rollout_result = []
			next_state, reward = self.rollout_four(i)
			rollout_result.append([next_state, reward])
			while temp_depth > 0:
				full_state = []
				action = self.actor.act(next_state)
				next_state, reward = self.rollout_four(action)
				rollout_result.append([next_state, reward])
				temp_depth -= 1
			result.append(rollout_result)
		return np.array(result)

