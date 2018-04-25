import tensorflow as tf
from collections import deque
import numpy as np
import random
import gym
from math import log

record = open("performance", "w")
savefile = "./savefile.h5"

# POLE-SPECIFIC
max_time = 500

gamma = 0.9
class ICNet:
    def __init__(self, sess, input_size, action_num):
        self.sess = sess
        self.x = tf.placeholder("float32", [None, input_size])
        W1 = tf.Variable(tf.random_uniform([input_size, 20], 0, 1))
        b1 = tf.Variable(tf.random_uniform([20], 0, 1))
        l1 = tf.nn.relu(tf.matmul(self.x, W1)+b1)
        W2 = tf.Variable(tf.random_uniform([20, action_num], 0, 1))
        b2 = tf.Variable(tf.random_uniform([action_num], 0, 1))
        self.y_hat = tf.nn.relu(tf.matmul(l1, W2)+b2)

        #self.y = tf.placeholder("float", [None, action_num])
        self.q_val = tf.placeholder("float32", [None]) #Proper q-vals as calculated by the bellman equation
        self.actions = tf.placeholder("float32", [None, action_num]) #Actions stored as one-hot vectors
        q_val_hat = tf.reduce_sum(tf.multiply(self.y_hat, self.actions), 1) #The q-vals for the actions selected in game
        #loss = tf.reduce_sum(tf.square(self.q_val - q_val_hat))
        loss = tf.losses.mean_squared_error(self.q_val, q_val_hat)
        self.train = tf.train.AdamOptimizer(0.001).minimize(loss)
        self.saver = tf.train.Saver()

    def update(self, state, action, reward, next_state, done):
        #print("next_state", next_state)
        #print("rewards", reward)
        reward_vecs = self.sess.run(self.y_hat, {self.x: next_state})
        #print("Reward Vec: ", reward_vecs)
        #print("update terms", np.amax(reward_vecs, 1))
        q_vals = reward + gamma*np.amax(reward_vecs, 1)*(1-done)


        #print("q_vals: ", q_vals)

        self.sess.run(self.train, {self.x: state, self.q_val: q_vals, self.actions: action})

    def action(self, state):
        reward_vec = self.sess.run(self.y_hat, {self.x: state})
        #print(reward_vec)
        #print(np.argmax(reward_vec))
        return np.argmax(reward_vec)



# Deep Q-learning Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.epsilon = .8  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()
        self.episodes = 3000
        self.training_result = []

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        session = tf.Session()
        model = ICNet(session, self.state_size, self.action_size)
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.action(state)
        return act_values  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for tup in minibatch:
            states.append(tup[0][0])
            actions.append(tup[1])
            rewards.append(tup[2])
            next_states.append(tup[3][0])
            dones.append(tup[4])
        states = np.array(states)
        #print("actions_1", actions)
        actions = np.eye(self.action_size)[actions]
        #print("actions_2", actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        #print("states", states, "actions", actions, "rewards", rewards, "next_states", next_states)
        #print("rewards", rewards)
        self.model.update(states, actions, rewards, next_states, dones)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.saver.restore(self.model.sess, name)

    def save(self, name):
        self.model.saver.save(self.model.sess, name)

    # Should be def train(self, agent_action)
    def train(self):
        while True:
            try:
                env = gym.make(env_name)
            except RuntimeWarning:
                print("RuntimeWarning caught: retrying")
                continue
            except RuntimeError:
                print("RuntimeError caught: retrying")
                continue
            else:
                break
        #env = gym.make('CartPole-v1')
        print("env stuff", env.observation_space, env.action_space)
        init = tf.global_variables_initializer()
        self.model.sess.run(init)
        # Iterate the game
        for e in range(self.episodes):
            # reset state in the beginning of each game
            while True:
                try:
                    state = env.reset()
                except RuntimeWarning:
                    print("RuntimeWarning caught: retrying")
                    continue
                except RuntimeError:
                    print("RuntimeError caught: retrying")
                    continue
                else:
                    break
            #state = env.reset()
            state = np.reshape(state, [1, self.state_size])
            # time_t represents each frame of the game
            # Our goal is to keep the pole upright as long as possible until score of max_time
            # the more time_t the more score
            for time_t in range(max_time):
                # turn this on if you want to render
                #env.render()s
                # Decide action
                action = self.act(state)
                # Advance the game to the next frame based on the action.
                # Reward is 1 for every frame the pole survived
                next_state, reward, done, _ = env.step(action)
                if(done and time_t < 499):
                    reward = -1
                # # POLE-SPECIFIC
                # if time_t == max_time - 1:
                #     reward = 150
                # elif done:
                #     reward = -5
                # else:
                #     reward = log(time_t + 1) / 10 + 1
                next_state = np.reshape(next_state, [1, self.state_size])
                # Remember the previous state, action, reward, and done
                self.remember(state, action, reward, next_state, done)
                # make next_state the new current state for the next frame.
                state = next_state
                # done becomes True when the game ends
                # ex) The agent drops the pole
                if done:
                    # print the score and break out of the loop
                    if e%10==0: print("episode: {}/{}, score: {}"
                          .format(e, self.episodes, time_t))
                    break
            # train the agent with the experience of the episode
            self.training_result.append(time_t)
            num_mem = len(self.memory)
            if(num_mem > 64):
                num_mem = 64
            for _ in range(100):
                agent.replay(num_mem)
        for e in self.training_result:
            record.write(str(e) + " ")
def compress(state):
    new_state = []
    for block in state:
        temp = []
        for arr in block:
            if arr[0] == 0:
                temp.append(0)
            elif arr[0] == 243:
                temp.append(1)
            elif arr[0] == 254:
                if arr[1] == 126:
                    temp.append(2)
                if arr[1] == 95:
                    temp.append(3)
            elif arr[0] == 142:
                temp.append(4)
            elif arr[0] == 160:
                temp.append(5) 
            elif arr[0] == 219:
                temp.append(6)   
        new_state.append(temp)  
    return np.array(new_state)

env_name = 'CartPole-v1'    
if env_name == 'TinyWorld-Sokoban-small-v0':
    agent = DQNAgent(49,8)
else:
    print('CartPole')
    agent = DQNAgent(4, 2)

def train_agent():
    agent.train()
    agent.save(savefile)

def load_agent():
    agent.load(savefile)


if __name__ == "__main__":
    train_agent()