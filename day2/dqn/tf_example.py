import numpy as np

r_test = [i*.02 for i in range(int(1/.02))]
def generate_data():
    a = abs(int(np.random.normal(100, 30)))+1
    k = abs(int(np.random.normal(10, 2)))+1
    n = abs(int(np.random.normal(10, 3)))+1

    avg_profit = np.array([0]*len(r_test))
    for trial in range(50):
        profit_r = [0]*len(r_test)
        trial_k = [k]*len(r_test)
        for auction in range(a):
            price = max(np.random.uniform(size=n))
            for i in range(len(r_test)):
                if price >= r_test[i] and trial_k[i] > 0:
                    trial_k[i] -= 1
                    profit_r[i] += price
                    
            avg_profit = avg_profit + np.array(profit_r)
    return ((a,k,n), r_test[np.argmax(avg_profit)])
generate_data()

import tensorflow as tf

f_1 = tf.nn.relu
f_2 = tf.nn.sigmoid
input_size = 3
output_size = 1
hidden_size = 5
x = tf.placeholder(tf.float32, [None, input_size])
W_1 = tf.Variable(tf.random_normal([input_size, hidden_size]))
b_1 = tf.Variable(tf.random_normal([hidden_size]))
W_2 = tf.Variable(tf.random_normal([hidden_size, output_size]))
b_2 = tf.Variable(tf.random_normal([output_size]))
y_hat = f_2(tf.matmul(f_1(tf.matmul(x, W_1)+b_1), W_2)+b_2)

y_ = tf.placeholder(tf.float32, [None, output_size])

mse = tf.losses.mean_squared_error(y_, y_hat) 

train_step = tf.train.AdamOptimizer(.001).minimize(mse)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(5000):
    if _%100==0: print("training step: ", _)
    data = generate_data()
    sess.run(train_step, feed_dict={x: np.array(data[0]).reshape((3,1)).T, y_: np.array(data[1]).reshape((1,1)).T})

import math
def relu(x):
    return [max(0, i) for i in x]

def sigmoid(x):
    return [1 / (1 + math.exp(-i)) for i in x]

def matmul(M, x):
    result = []
    for i in range(len(M[0])):
        result.append(sum([M[j][i]*x[j] for j in range(len(x))]))
    return result

def elemwise_add(lst1, lst2):
    return [lst1[i] + lst2[i] for i in range(len(lst1))]

def calculateReserve(a, k, n):
    inpt = [a,k,n]
    W_1 = [[ 0.38604805,  0.45385805, -1.03400028,  0.03189703, -1.89355886], 
    [-0.36459139, -0.31411317, -0.73125869, -0.24245232, -0.32934415],
    [ 0.89335972,  1.01257086, -1.97831976, -1.38710451, 0.08754305]]
    W_2 = [[ 0.60520422],[-0.48648942],[ 0.30566561],[ 0.11410404],[-0.88456303]]
    b_1 = [ 0.65610039, -0.53640163, -1.26945722, -2.12892461, 1.90470695]
    b_2 = [ 0.21751852]
    first_layer = relu(elemwise_add(matmul(W_1, inpt), b_1))
    return sigmoid(elemwise_add(matmul(W_2,first_layer),b_2))