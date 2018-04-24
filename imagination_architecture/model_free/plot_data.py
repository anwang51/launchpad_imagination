import matplotlib.pyplot as plt

f = open("performance_timeseries", "r")
s = f.read()
s = s.split(" ")
nums = []
for e in s:
	if e != "":
		nums.append(float(e))
plt.plot(nums)
plt.show()