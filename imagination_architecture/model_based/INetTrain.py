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

def generate_rollouts(policy_net, env_model, state, action_size):
    '''
    policy_net:
        Input: state
        Output: action

    env_model:
        Input: state, action
        Output: next state, reward, etc.
    '''
    actions = np.eye(action_size)
    rollouts = []
    for action in actions:
        curr_state = env_model(state, action)
        rollout = [curr_state]
        for _ in range(0, 4):
            generated_action = policy_net(curr_state)
            curr_state = env_model(curr_state, generated_action)
            rollout.append(curr_state)
        rollouts.append(rollout)
    return rollouts


if __name__ == "__main__":
    train_agent()
