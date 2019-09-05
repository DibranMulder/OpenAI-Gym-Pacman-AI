import gym
import numpy as np
import tensorflow as tf
import time
env = gym.make('MsPacman-v0')

num_episodes = 5
env_features = 210*160*3


# ['NOOP', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT']

#input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
inputs = tf.placeholder(shape=[None, 210, 160, 3], dtype=tf.float32)
filters = tf.Variable(tf.truncated_normal([5, 5, 3, 32], stddev=0.1))

firstLayer = tf.nn.conv2d(inputs, filters, padding='SAME')

# with tf.Session() as sess:

for i in range(num_episodes):
    state = env.reset()
    print(state)
    totalReward = 0

    for _ in range(1000):
        env.render()
        # time.sleep(0.1)
        # print(env.action_space)

        # print(env.unwrapped.get_action_meanings())
        # env.action_space.sample() random action
        observation,reward,done,info = env.step(0) # take a random action

        # flatten the array
        x = np.reshape(observation, env_features)
        print(len(x))
        # print(env.observation_space) # Box(210, 160, 3)
        # print(env.observation_space)
        # print(env.observation_space.low)
        
        # print(env.observation_space.high[0])
        # print (len(observation[0][0]))
        # for p in observation[10]:
            #print(p)
            #time.sleep(1)
        #    avg = avg_rgb_pixel(p)
            #print(avg)

        #print (observation[0][0])
        
        
        # x = np.reshape(env.observation_space, (-1, 210, 160, 3))
        print('--------')
        time.sleep(0.1)
        # totalReward += reward

    print('Episode', i,', Total reward:', totalReward)

env.close()


