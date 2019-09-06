import time
import gym
env = gym.make('MsPacman-v0')

num_episodes = 5
num_actions = 1000

for i in range(num_episodes):
    state = env.reset()
    totalReward = 0

    for _ in range(num_actions):
        env.render()

        # take a random action
        randomAction = env.action_space.sample()

        observation,reward,done,info = env.step(randomAction) 

        time.sleep(0.1)
        totalReward += reward

    print('Episode', i,', Total reward:', totalReward)

env.close()