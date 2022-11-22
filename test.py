#!/usr/bin/env python3
import gym
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim


HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70
batch = []
stept = 0


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


def iterate_batches(env, net):
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    sm1 = nn.Softmax(dim=0)

    while True:
        act_probs_v = sm1(net(torch.FloatTensor(obs)))#tensor([[0.6065, 0.3935]], grad_fn=<SoftmaxBackward>) <class 'torch.Tensor'>
        act_probs = act_probs_v.data.numpy()#取出数据部分，转化为numpy
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, _ = env.step(action)
        episode_reward += reward
        step = EpisodeStep(observation=obs, action=action)
        episode_steps.append(step)
        if is_done:
            e = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(e)
            break
        obs = next_obs



if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    #env = gym.wrappers.Monitor(env, directory="mon", force=True)
    obs_size = env.observation_space.shape[0]##环境给出的观察值空间大小
    n_actions = env.action_space.n ##环境动作空间大小

    net = Net(obs_size, HIDDEN_SIZE, n_actions)##输入为环境的观察值（4），隐藏层为128，输出为动作空间（2）
    objective = nn.CrossEntropyLoss()##交叉熵损失函数，自带softmax
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)
    writer = SummaryWriter(comment="-cartpole")

    while True:
        for i in range(BATCH_SIZE):
            iterate_batches(env, net)

        rewards = list(map(lambda s: s.reward, batch))
        reward_bound = np.percentile(rewards, PERCENTILE)  # 一个阈值，占比percentile%的数小于这个阈值
        reward_mean = float(np.mean(rewards))

        if reward_mean > 199:
            break
        train_obs = []
        train_action = []
        for reward, step in batch:
            if reward < reward_bound:
                continue
            train_action.extend(map(lambda s: s.action, step))
            train_obs.extend(map(lambda s: s.observation, step))

        #obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        batch = []
        optimizer.zero_grad()#梯度置0
        action_scores_v = net(torch.FloatTensor(train_obs))
        loss_v = objective(action_scores_v, torch.LongTensor(train_action))
        loss_v.backward() #计算梯度
        optimizer.step()#更新参数
        print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (
            stept, loss_v.item(), reward_mean, reward_bound))

        writer.add_scalar("loss", loss_v.item(), stept)
        writer.add_scalar("reward_bound", reward_bound, stept)
        writer.add_scalar("reward_mean", reward_mean, stept)
        stept = stept + 1
    writer.close()
