class ImaginationCore:

	def __init__(self):
		self.env = EnvironmentModel()
		self.actor = DQN()

	def rollout_single(self, state):
		action = self.actor(state)
		next_state, reward = self.env(state, action)
		return next_state, reward

	def rollout(self, state, depth=3):
		rew = 0
		st = state
		out = [(st, rw)]
		while depth > 0:
			st, rew = self.rollout_single(st)
			out.append((st, rw))
			depth -= 1
		return out