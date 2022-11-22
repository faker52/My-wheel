import gym
import numpy as np
a = gym.spaces.Box(low =-1,high = 1,shape=(3,))
b = gym.spaces.Discrete(4)
print(b.n)
print(a.shape[1])
print(a.sample())
