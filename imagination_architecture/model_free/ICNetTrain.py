import tensorflow as tf
from collections import deque
import numpy as np
import random
import gym
import gym_sokoban
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
        W1 = tf.Variable(tf.random_uniform([input_size, 128], 0, 1))
        b1 = tf.Variable(tf.random_uniform([128], 0, 1))
        W2 = tf.Variable(tf.random_uniform([128, 128], 0, 1))
        b2 = tf.Variable(tf.random_uniform([128], 0, 1))
        W3 = tf.Variable(tf.random_uniform([128, action_num], 0, 1))
        b3 = tf.Variable(tf.random_uniform([action_num], 0, 1))
        l1 = tf.nn.elu(tf.matmul(self.x, W1)+b1)
        l2 = tf.nn.elu(tf.matmul(l1, W2)+b2)
        self.y_hat = tf.nn.elu(tf.matmul(l2, W3)+b3)

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
        self.epsilon = 0.2  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()
        init = tf.global_variables_initializer()
        self.model.sess.run(init)
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
                env = gym.make(which_env)
            except RuntimeWarning:
                print("RuntimeWarning caught: retrying")
                continue
            except RuntimeError:
                print("RuntimeError caught: retrying")
                continue
            else:
                break
        print("env stuff", env.observation_space, env.action_space)
        epis = 0
        f = open("performance_timeseries", "a")
        # Iterate the game
        while True:
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
            if which_env == 'TinyWorld-Sokoban-small-v0':
                state = compress(state)
            #print("shape: ", np.shape(state))
            #print("shape0: ", np.shape(state[0]))
            state = np.reshape(state, [1, self.state_size])
            #print("outside")
            #print("after reshape: ", state)
            # time_t represents each frame of the game
            # Our goal is to keep the pole upright as long as possible until score of max_time
            # the more time_t the more score
            performance_score = 0
            done = False
            while not done:
                # turn this on if you want to render
                # env.render()
                # Decide action
                action = agent.act(state)
                #print(np.shape(state))
                #test = np.array([[1,2,3]])
                #print(np.shape(test))
                # Advance the game to the next frame based on the action.
                # Reward is 1 for every frame the pole survived
                next_state, reward, done, _ = env.step(action)
                performance_score += reward
                if which_env == 'TinyWorld-Sokoban-small-v0':
                    next_state = compress(next_state)
                next_state = np.reshape(next_state, [1, self.state_size])
                # Remember the previous state, action, reward, and done
                agent.remember(state, action, reward, next_state, done)
                # make next_state the new current state for the next frame.
                state = next_state
                # done becomes True when the game ends
                # ex) The agent drops the pole
                if done:
                    # # print the score and break out of the loop
                    # print("episode: {}/{}, score: {}"
                    #       .format(e, episodes, reward))
                    break
            out_str = str(performance_score) + " "
            f.write(out_str)
            f.flush()
            print("episode: {}/{}, score: {}"
                          .format(epis, float("inf"), performance_score))
            # train the agent with the experience of the episode
            num_mem = len(agent.memory)
            if num_mem > 32:
                num_mem = 32
            agent.replay(num_mem)
            epis += 1
        agent.model.save_model("tfmodel_weights.h5")

#     0: wall = [0, 0, 0] 
#     1: floor = [243, 248, 238]
#     2: box_target = [254, 126, 125]
#     3: box_on_target = [254, 95, 56]
#     4: box = [142, 121, 56]
#     5: player = [160, 212, 56]
#     6: player_on_target = [219, 212, 56]

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

#Sokoban: 49, 8
#Cartpole: 4, 2
which_env = ['TinyWorld-Sokoban-small-v0', 'CartPole-v0'][1]
if which_env == 'TinyWorld-Sokoban-small-v0':
    agent = DQNAgent(49,8)
else:
    agent = DQNAgent(4, 2)

def train_agent():
    agent.train()
    agent.save(savefile)

def load_agent():
    agent.load(savefile)


if __name__ == "__main__":
    train_agent()