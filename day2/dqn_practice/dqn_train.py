import gym
import gym_sokoban
from dqn_agent import DQNAgent
import settings
import numpy as np

episodes = settings.episodes

record = open("performance", "w")
training_result = []

if __name__ == "__main__":
    # initialize gym environment and the agent
    env = gym.make('Sokoban-v0')
    print(env.observation_space, env.action_space)
    agent = DQNAgent(4,2)
    # Iterate the game
    for e in range(episodes):
        # reset state in the beginning of each game
        state = env.reset()
        state = np.reshape(state, [1, 4])
        # time_t represents each frame of the game
        # Our goal is to keep the pole upright as long as possible until score of 500
        # the more time_t the more score
        for time_t in range(500):
            # turn this on if you want to render
            env.render()
            # Decide action
            action = agent.act(state)
            # Advance the game to the next frame based on the action.
            # Reward is 1 for every frame the pole survived
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 4])
            # Remember the previous state, action, reward, and done
            agent.remember(state, action, reward, next_state, done)
            # make next_state the new current state for the next frame.
            state = next_state
            # done becomes True when the game ends
            # ex) The agent drops the pole
            if done:
                # print the score and break out of the loop
                print("episode: {}/{}, score: {}"
                      .format(e, episodes, time_t))
                break
        # train the agent with the experience of the episode
        training_result.append(time_t)
        num_mem = len(agent.memory)
        if num_mem > 32:
            num_mem = 32
        agent.replay(num_mem)
    for e in training_result:
        record.write(str(e) + " ")