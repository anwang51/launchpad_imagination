# Deep Q-learning Agent
class ImaginationAgent:
    def __init__(self, state_size, action_size, num_paths):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.epsilon = 0.2  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()
        self.episodes = 3000
        self.training_result = []

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        session1 = tf.Session()
        session2 = tf.Session()
        ICNet = ICNet(session1, self.state_size, self.action_size)
        INet = INet(session2, self.state_size, )
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.action(state)
        return act_values  # returns action

    def format_paths(paths_list_list):
        next_list = zip(paths_list_list)
        ret = []
        for paths in next_list:
            ret.extend(paths)

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        paths_list_list = []
        MF_outputs = []
        actions = []
        rewards = []
        next_paths = []
        next_MF_outputs = []
        dones = []
        for tup in minibatch:
            paths = tup[0][0]
            paths_list_list.append(paths)
            MF_outputs.append(tup[0][1])
            actions.append(tup[1])
            rewards.append(tup[2])
            next_states.append(tup[3][0])
            next_MF_outputs.append(tup[3][1])
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
        env = gym.make('Sokoban-small-v0')
        print("env stuff", env.observation_space, env.action_space)
        init = tf.global_variables_initializer()
        self.model.sess.run(init)
        # Iterate the game
        for e in range(self.episodes):
            # reset state in the beginning of each game
            state = env.reset()
            state = np.reshape(state, [1, 37632])
            # time_t represents each frame of the game
            # Our goal is to keep the pole upright as long as possible until score of max_time
            # the more time_t the more score
            done = False
            while not done:
                # turn this on if you want to render
                # env.render()
                # Decide action
                action = agent.act(state)
                # Advance the game to the next frame based on the action.
                # Reward is 1 for every frame the pole survived
                next_state, reward, done, _ = env.step(action)
                # print("reward ", reward)
                next_state = np.reshape(next_state, [1, 37632])
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
            print("episode: {}/{}, score: {}"
                          .format(e, self.episodes, reward))
            # train the agent with the experience of the episode
            num_mem = len(agent.memory)
            if num_mem > 32:
                num_mem = 32
            agent.replay(num_mem)
        agent.model.save_model("tfmodel_weights.h5")
        

agent = DQNAgent(37632,8)

def train_agent():
    agent.train()
    agent.save(savefile)

def load_agent():
    agent.load(savefile)


if __name__ == "__main__":
    train_agent()