import AllIsCircle as g
from math import acos, sqrt
import numpy as np

rad = 0
counter = 0
move_x = 500*np.random.normal(0, 1)
move_y = 500*np.random.normal(0, 1)

def act(vector):
	global rad, counter, move_x, move_y
	tempGame = g.game(vector)
	ship1 = tempGame.ship_0
	ship2 = tempGame.ship_1
	dif_x = float(ship1.x - ship2.x)
	dif_y = float(ship1.y - ship2.y)
	hyp = float(sqrt((ship1.x - ship2.x)**2 + (ship1.y - ship2.y)**2))
	aim = acos(dif_x/hyp)
	if dif_y < 0:
		aim = -aim
	if counter > 20:
		move_x = 500*np.random.normal(0, 1)
		move_y = 500*np.random.normal(0, 1)
		counter = 0
	counter += 1
	action = [move_x, move_y, aim, 1]
	return action