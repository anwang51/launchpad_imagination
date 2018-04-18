from ShipsEnv import ShipsEnv
import DumbAI as dum
from SmartAI import DQNAgent
import numpy as np
from keras.models import load_model

env = ShipsEnv(True)
vec = env.game_vec
reward = 0
games_count = 0
wins = 0
win_rate = None

agent = DQNAgent(70, 32)
# agent.model = load_model("good_dodge_model11264_0.12.hd5")

num_games = 100000
save_checkpoint = 1024
i = 0
while i < num_games:
	while reward == 0:
		vec_smart = np.array([vec])
		vec_dumb = vec
		player_ai = agent.act(vec_smart)
		player_act = agent.actions[player_ai]
		opponent_ai = dum.act(vec_dumb)
		reward, new_vec = env.step(player_act, opponent_ai)
		new_vec_smart = np.array([new_vec])
		raw_input("continue")
		agent.remember(vec_smart, player_ai, reward, new_vec_smart, reward!=0)
		vec = new_vec
	agent.replay(32)
	games_count += 1
	if reward == 1:
		wins += 1
	if games_count % 100 == 0 and games_count != 0:
		win_rate = float(wins)/games_count
		print "Iteration ", i, ": ", win_rate
		games_count = 0
		wins = 0
	i += 1
	if i % save_checkpoint == 0 and i != 0:
		agent.model.save("win_model"+str(i)+"_"+str(win_rate)+".hd5")
	env.reset()
	reward = 0
	vec = env.game_vec
print(win_rate)