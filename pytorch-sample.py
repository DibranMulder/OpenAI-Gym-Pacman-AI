import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple

from dqn import Net
from replaymemory import ReplayMemory

import time
import gym

env = gym.make('MsPacman-v0')

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Running on:',device)

action_names = ['NOOP', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT']
image_dimensions = 210*160*3
num_episodes = 50
target_episode_update = 5
action_threshold = 250
train_batch_size = 64

memory = ReplayMemory(10000)

policy_net = Net().to(device)
target_net = Net().to(device)
target_net.eval()

def optimize_model():
    if len(memory) < train_batch_size:
        return

    transitions = memory.sample(train_batch_size)
    print('Training on:',len(transitions))

def store_transition(state, action, reward, observation):
    memory.push(state, action, reward, observation)

def print_action(action):
    print('Action:', action_names[action])

for i in range(num_episodes):
    print('Episode: ',i)
    state = env.reset()

    observation = None
    action = None
    frames = 0

    for _ in range(1000):
        state = env.render()

        frames += 1

        if action_threshold > frames:
            continue

        print('Action threshold met',frames)

        rand = torch.randint(0, 10, (1,)).item()

        if observation is None or rand % 2 == 0:
            action = env.action_space.sample()
            print('Random action:',action)            
        else:
            # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
            input = T.ToTensor()(observation)
            # Adds dimension, its excepts a 4 dimensional array.
            out = target_net(input.unsqueeze(0))
            # print(out)
            netAction = out.max(1)[1].view(1, 1)
            action = netAction.item()
            print_action(action)
            print('NN action:',action)

        observation,reward,done,info = env.step(action)
        store_transition(state, action, reward, observation)

        state = observation

        optimize_model()
        print(len(memory))
        # print(done) # bool
        # print(info) # json with lives
        # time.sleep(0.1)

    # Update the target network, copying all weights and biases in DQN
    if i_episode % target_episode_update == 0:
        print('Updating DQN with optimized values')
        target_net.load_state_dict(policy_net.state_dict())

env.close()

