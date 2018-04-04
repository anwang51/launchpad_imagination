import AllIsCircle as g
from tkinter import *
from math import sqrt, sin, cos
import random

class ShipsEnv:

	def __init__(self, render):
		self.render = render
		self.game = g.start_game()
		self.game_vec = self.game.to_vec()
		#self.game_vec[0] = random.random()*500
		#self.game_vec[1] = random.random()*500
		#self.game_vec[5] = random.random()*500
		#self.game_vec[6] = random.random()*500
		self.master = Tk()
		self.w = Canvas(self.master, width=600, height=600)
		self.w.pack()

	def step(self, act1, act2):
		reward, self.game_vec, self.game = g.change_state(self.game_vec, act1, act2, self.w)
		if self.render:
			g.print_game(self.w, self.game)
		return reward, self.game_vec

	def reset(self):
		self.game = g.start_game()
		self.game_vec = self.game.to_vec()
<<<<<<< HEAD
		#self.game_vec[0] = random.random()*500
		#self.game_vec[1] = random.random()*500
		#self.game_vec[5] = random.random()*500
		#self.game_vec[6] = random.random()*500
		return self.game.to_vec()
=======
		self.master = Tk()
		self.w = Canvas(self.master, width=600, height=600)
		self.w.pack()
		return self.game_vec
>>>>>>> ebdb4896e846bc73f3ba2712b63cb71dcd327b5c
