import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from dqn import Net

import time
import gym

env = gym.make('MsPacman-v0')

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Running on:',device)

action_names = ['NOOP', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT']
image_dimensions = 210*160*3
num_episodes = 5
action_threshold = 200

net = Net()

def print_action(action):
    print('Action:', action_names[action])

for i in range(num_episodes):
    print('Episode: ',i)
    state = env.reset()

    observation = None
    action = None
    frames = 0

    for _ in range(1000):
        env.render()
        frames += 1

        if action_threshold > frames:
            continue

        print('Action threshold met',frames)

        if observation is None:
            action = env.action_space.sample()
            print('Random action:',action)            
        else:
            # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
            input = T.ToTensor()(observation)
            # Adds dimension, its excepts a 4 dimensional array.
            out = net(input.unsqueeze(0))
            # print(out)
            netAction = out.max(1)[1].view(1, 1)
            action = netAction.item()
            print_action(action)
            print('NN action:',action)

        observation,reward,done,info = env.step(action)
                
        # print(done) # bool
        # print(info) # json with lives

        # time.sleep(0.1)

env.close()

