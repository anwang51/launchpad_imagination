import matplotlib.pyplot as plt 
import sys

path1 =  "dqn/performance"
path2 =  "double_dqn_agent/performance"

dampening = 0.9

f1 = open(path1, "r")
text1 = f1.read().split(" ")
dqn = []
running_average = 0
for en in text1:
	if len(en) != 0:
		running_average = dampening*running_average + int(en)
		dqn.append(running_average)
f2 = open(path2, "r")
text2 = f2.read().split(" ")
double_dqn = []
running_average = 0
for en in text2:
	if len(en) != 0:
		running_average = dampening*running_average + int(en)
		double_dqn.append(running_average)
plt.plot(dqn, "b")
plt.plot(double_dqn, "r")
plt.show()