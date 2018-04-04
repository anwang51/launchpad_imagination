from ShipsEnv import ShipsEnv
import DumbAI as dum
import SmartAI as sm
import numpy as np

env = ShipsEnv(True)
vec = env.game_vec
reward = 0
while reward == 0:
	player_ai = sm.act(vec)
	opponent_ai = dum.act(vec)
	reward, vec = env.step(player_ai, opponent_ai)
	print("reward", reward)
	print("vec", vec)
	input("hit enter to continue")