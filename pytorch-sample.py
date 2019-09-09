import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import gym

env = gym.make('MsPacman-v0')

# ['NOOP', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT']
image_dimensions = 210*160*3

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 3 input image channels, 32 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 32, 5)
        # an affine operation: y = Wx + b
        # I Would expect 32*32*32 = 32768 
        self.fc1 = nn.Linear(58016, 8192)  # 6*6 from image dimension
        self.fc2 = nn.Linear(8192, 2048)
        self.fc3 = nn.Linear(2048, 256)
        self.fc4 = nn.Linear(256, 9)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)

batch_tensor = torch.randn(*(10, 3, 256, 256))

img = torch.randn(1, 3, 210, 160)

out = net(img)
print(out)

num_episodes = 5
for i in range(num_episodes):
    state = env.reset()

    for _ in range(1000):
        env.render()

        randomAction = env.action_space.sample()
        observation,reward,done,info = env.step(randomAction)

        tt = torch.from_numpy(observation)
        # input = torch.squeeze(tt)

        # out = net(input)
        # print(out)

        time.sleep(1)

env.close()