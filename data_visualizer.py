import matplotlib.pyplot as plt

data = [93.19,212.46,289.24,270.53,242.04,272.43,258.84,211.85,299.85,269.31,248.72,242.91,243.87,247.76,305.85,234.8,330.31,301.19,246.47,281.35,324.78]

inp = """Iteration  99 :  0.03
Iteration  199 :  0.15
Iteration  299 :  0.13
Iteration  399 :  0.21
Iteration  499 :  0.31
Iteration  599 :  0.12
Iteration  699 :  0.22
Iteration  799 :  0.22
Iteration  899 :  0.24
Iteration  999 :  0.22
Iteration  1099 :  0.19
Iteration  1199 :  0.3
Iteration  1299 :  0.28
Iteration  1399 :  0.31
Iteration  1499 :  0.28
Iteration  1599 :  0.17
Iteration  1699 :  0.25
Iteration  1799 :  0.37
Iteration  1899 :  0.35
Iteration  1999 :  0.2
Iteration  2099 :  0.2
Iteration  2199 :  0.24
Iteration  2299 :  0.27
Iteration  2399 :  0.29
Iteration  2499 :  0.23
Iteration  2599 :  0.28
Iteration  2699 :  0.22
Iteration  2799 :  0.34
Iteration  2899 :  0.24
Iteration  2999 :  0.25
Iteration  3099 :  0.23
Iteration  3199 :  0.21
Iteration  3299 :  0.38
Iteration  3399 :  0.39
Iteration  3499 :  0.24
Iteration  3599 :  0.29
Iteration  3699 :  0.2
Iteration  3799 :  0.34
Iteration  3899 :  0.24
Iteration  3999 :  0.2
Iteration  4099 :  0.39
Iteration  4199 :  0.22
Iteration  4299 :  0.23
Iteration  4399 :  0.18
Iteration  4499 :  0.2
Iteration  4599 :  0.31
Iteration  4699 :  0.28
Iteration  4799 :  0.2
Iteration  4899 :  0.35
Iteration  4999 :  0.26"""

lines = inp.split("\n")
data = [float(line.split(" ")[5]) for line in lines]

plt.plot(data)
plt.show()