# This is a sample Python script.
import gym
import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple
import torch.nn as nn
import torch.optim as optim
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

HIDDEN_SIZE = 128
BATCH_SIZE =16
Percentile = 70

EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])
Episode = namedtuple('Episode', field_names=['reward', 'steps'])
stept=0


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_action):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_action)
        )
    def forward(self, x):
        return self.net(x)



class Agent:
    def __init__(self, env):
        self.env_name = env
        self.env = gym.make(env)
        self.action_shape = self.env.action_space.n
        self.observation_shape = self.env.observation_space.shape[0]
        self.net = Net(self.observation_shape, HIDDEN_SIZE, self.action_shape)
        self.tra = []

    def play(self):
        episode_reward = 0
        steps = []
        obs = self.env.reset()
        while True:
            sm = nn.Softmax(dim=0)
            action_probs = sm(self.net(torch.FloatTensor(obs)))
            action_probs = action_probs.data.numpy()
            action = np.random.choice(len(action_probs), p=action_probs)
            next_obs, reward, is_done, _ = self.env.step(action)
            episode_reward += reward
            episode = EpisodeStep(observation=obs, action=action)
            steps.append(episode)
            if is_done:
                traj = Episode(reward=episode_reward, steps=steps)
                self.tra.append(traj)
                break
            obs = next_obs


if __name__ == '__main__':
    agent = Agent('CartPole-v0')
    while True:
        for i in range(BATCH_SIZE):
            agent.play()
        tra = agent.tra
        rewards = list(map(lambda s: s.reward, tra))
        reward_bound = np.percentile(rewards, Percentile)  # 一个阈值，占比percentile%的数小于这个阈值
        reward_mean = float(np.mean(rewards))


        if reward_mean > 199:
            break
        train_obs = []
        train_action = []
        for reward, step in tra:
            if reward < reward_bound:
                continue
            train_action.extend(map(lambda s: s.action, step))
            train_obs.extend(map(lambda s: s.observation, step))
        agent.tra = []
        objective = nn.CrossEntropyLoss()  ##交叉熵损失函数，自带softmax
        optimizer = optim.Adam(params=agent.net.parameters(), lr=0.01)
        optimizer.zero_grad()
        action_net = agent.net(torch.FloatTensor(train_obs))
        loss_v = objective(action_net, torch.LongTensor(train_action))
        loss_v.backward()
        optimizer.step()
        print("%d  reward_mean: %.3f,  reward_bound :%.3f  loss:%.2f" % (stept, reward_mean, reward_bound,loss_v.item()))
        stept = stept+1
print("reward_mean: %.3f,  reward_bound :%.3f  loss:%.2f" % (reward_mean, reward_bound, loss_v.item()))





