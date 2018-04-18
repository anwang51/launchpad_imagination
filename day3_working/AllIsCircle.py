from tkinter import *
from math import sqrt, sin, cos
import threading
import time

#Global variables
bounds = 500
refresh_time = 40
default_vec = [0, 0, 0, 50, 0]
class ship:
	def __init__(self, bullets_list, x=0, y=0, gun_direction=0, health=50, refresh_clock=0):
		self.bullets_list = bullets_list
		self.x = x
		self.y = y
		self.size = 15
		self.gun_direction = gun_direction
		self.health = health
		self.refresh_clock = refresh_clock

	def truncate_to_bounds(self):
		if(self.x < 0):
			self.x = 0
		elif(self.x > bounds):
			self.x = bounds
		if(self.y < 0):
			self.y = 0
		elif(self.y > bounds):
			self.y = bounds

	def action(self, input):
		#Move on the screen
		length = sqrt(input[0]**2+input[1]**2)
		vx = 0
		vy = 0
		if(length > 0):
			vx = input[0]/length
			vy = input[1]/length
		self.truncate_to_bounds()
		if(vx+self.x < bounds and vx + self.x > 0):
			self.x += vx
		if(vy+self.y < bounds and vy + self.y > 0):
			self.y += vy
		#Rotate the gun to determined position
		self.gun_direction = input[2] % (2*3.14159)
		#Shoot
		if(input[3] > 0 and self.refresh_clock <= 0):
			vx = 5*cos(self.gun_direction)
			vy = 5*sin(self.gun_direction)
			self.bullets_list.append(bullet(self.x + vx + 25*cos(self.gun_direction), self.y + vy + 25*sin(self.gun_direction), vx, vy))
			self.refresh_clock = refresh_time
		self.refresh_clock -= 1
	def to_vec(self):
		return [self.x, self.y, self.gun_direction, self.health, self.refresh_clock]

class bullet:
	def __init__(self, x=0, y=0, vx=0, vy=0):
		self.x = x
		self.y = y
		self.vx = vx
		self.vy = vy
		self.size = 8
	def action(self):
		self.x += self.vx
		self.y += self.vy
	def in_bounds(self):
		return (self.x > 0 and self.x < bounds and self.y > 0 and self.y < bounds)

	def to_vec(self):
		return [self.x, self.y, self.vx, self.vy]

class game:
	def __init__(self, state_vec):
		self.bullets_list = []
		s = state_vec[0 : 5]
		self.ship_0 = ship(self.bullets_list, s[0], s[1], s[2], s[3], s[4])
		s = state_vec[5 : 10]
		self.ship_1 = ship(self.bullets_list, s[0], s[1], s[2], s[3], s[4])
		place = 10
		while(place < 70):
			s = state_vec[place: place + 4]
			self.bullets_list.append(bullet(s[0], s[1], s[2], s[3]))
			place += 4

	def dist(self, a, b):
		return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

	def is_collision(self, a, b):
		a_vec = [a.x, a.y]
		b_vec = [b.x, b.y]
		dist = self.dist(a_vec, b_vec)
		#if((dist < (a.size + b.size))):
		#	print("collision", dist, a_vec, b_vec)
		return (dist < (a.size + b.size))

	def action(self, input_0, input_1):
		#Have ships and bullets move
		self.ship_0.action(input_0)
		self.ship_1.action(input_1)
		for b in self.bullets_list:
			b.action()
		#Delete bullets that go out of bounds or hit ships
		#Lower ship health if it's in collision
		i = 0
		while (i < len(self.bullets_list)):
			b = self.bullets_list[i]
			#print(i, b.x, b.y)
			if(self.is_collision(self.ship_0, self.bullets_list[i])):
				self.ship_0.health -= 1
				del self.bullets_list[i]
			elif(self.is_collision(self.ship_1, b)):
				self.ship_1.health -= 1
				del self.bullets_list[i]
			elif(not self.bullets_list[i].in_bounds()):
				del self.bullets_list[i]
			else:
				i += 1
		#Get reward for ship_0
		if(self.ship_0.health <= 0):
			return -1
		elif(self.ship_1.health <= 0):
			return 1
		else:
			return 0

	def to_vec(self):
		ret = self.ship_0.to_vec() + self.ship_1.to_vec()
		for b in self.bullets_list:
			ret = ret + b.to_vec()
		padding = []
		if(len(self.bullets_list) < 15):
			padding = [0, 0, 0, 0]*(15 - len(self.bullets_list))
		return ret + padding


def start_game():
	state_vec = [-1]*70
	# state_vec[0] = state_vec[1] = 100
	state_vec[3] = 1
	# state_vec[5] = state_vec[6] = 400
	state_vec[0] = state_vec[1] = 400
	state_vec[5] = state_vec[6] = 100
	state_vec[8] = 1
	return game(state_vec)

def change_state(state_vec, action_0, action_1):
	new_game = game(state_vec)
	reward = new_game.action(action_0, action_1)
	return reward, new_game.to_vec(), new_game

def tkinter_test():
	root = tk.Tk()
	w = tk.Label(root, text="Hello, world!")
	w.pack()
	root.mainloop()

def print_game(w, game):
	w.delete("all")
	w.create_rectangle(50, 50, 550, 550, fill="blue")
	print_entity(game.ship_0, w)
	print_entity(game.ship_1, w)
	for b in game.bullets_list:
		print_entity(b, w)
	#mainloop()

def run_game(curr_game):
	master = Tk()
	w = Canvas(master, width=600, height=600)
	w.pack()
	frame(w, curr_game, master)

def frame(w, curr_game, master):
	w.delete("all")
	w.create_rectangle(50, 50, 550, 550, fill="blue")
	print_entity(curr_game.ship_0, w)
	print_entity(curr_game.ship_1, w)
	for b in curr_game.bullets_list:
		print_entity(b, w)

	acts = [0, 1, 3.14159/4, 1]
	curr_game.action(acts, acts)
	master.after(40, (lambda : frame(w, curr_game, master)))

def print_entity(entity, c):
	x = entity.x+50
	y = entity.y+50
	s = entity.size
	c.create_oval(x-s/2, y-s/2, x+s/2,y+s/2, fill="red")












