import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
from core import repaly_memory

class Actor(nn.Module):

    def __init__(self, args, init=False):
        super(Actor, self).__init__()
        self.args = args
        l1 = args.ls; l2 = args.ls; l3 = l2

        # Construct Hidden Layer 1
        self.w_l1 = nn.Linear(args.state_dim, l1)
        if self.args.use_ln: self.lnorm1 = LayerNorm(l1)

        # Hidden Layer 2
        self.w_l2 = nn.Linear(l1, l2)
        if self.args.use_ln: self.lnorm2 = LayerNorm(l2)

        # Out
        self.w_out = nn.Linear(l3, args.action_dim)

        # Init
        if init:
            self.w_out.weight.data.mul_(0.1)
            self.w_out.bias.data.mul_(0.1)

        self.to(self.args.device)

    def forward(self, input):

        # Hidden Layer 1
        out = self.w_l1(input)
        out = self.lnorm1(out)
        out = out.tanh()

        # Hidden Layer 2
        out = self.w_l2(out)
        out = self.lnorm2(out)
        out = out.tanh()

        # Out
        out = (self.w_out(out)).tanh()
        return out

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.args.device)
        return self.forward(state).cpu().data.numpy().flatten()



class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.args = args

        l1 = 200; l2 = 300; l3 = l2
        # Construct input interface (Hidden Layer 1)
        self.w_state_l1 = nn.Linear(args.state_dim, l1)
        self.w_action_l1 = nn.Linear(args.action_dim, l1)

        # Hidden Layer 2
        self.w_l2 = nn.Linear(2 * l1, l2)
        self.lnorm2 = LayerNorm(l2)

        # Out
        self.w_out = nn.Linear(l3, 1)
        self.w_out.weight.data.mul_(0.1)
        self.w_out.bias.data.mul_(0.1)

        self.cuda()

    def forward(self, input, action):

        # Hidden Layer 1 (Input Interface)
        out_state = F.elu(self.w_state_l1(input))
        out_action = F.elu(self.w_action_l1(action))
        out = torch.cat((out_state, out_action), 1)

        # Hidden Layer 2
        out = self.w_l2(out)
        if self.args.use_ln: out = self.lnorm2(out)
        out = F.elu(out)

        # Output interface
        out = self.w_out(out)

        return out


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class DDPG(object):

    def __init__(self, args):
        self.args = args

        self.tau = args.tau
        self.actor = Actor(args)
        self.actor_target = Actor(args)
        self.critic = Critic(args)
        self.critic_target = Critic(args)

        self.actor_optim = Adam(self.actor.parameters(), lr=0.5e-4)
        self.critic_optim = Adam(self.critic.parameters(), lr=0.5e-3)

        self.gamma = args.gamma;
        self.loss = nn.MSELoss()

        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)


    def update_parameters(self, batch):
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = batch
        print(np.shape(state_batch), np.shape(action_batch), np.shape(next_state_batch), np.shape(reward_batch), np.shape(done_batch))
        # Load everything to GPU if not already
        self.actor_target.to(self.args.device)
        self.critic_target.to(self.args.device)
        self.critic.to(self.args.device)
        state_batch = state_batch.to(self.args.device)
        next_state_batch = next_state_batch.to(self.args.device)
        action_batch = action_batch.to(self.args.device)
        reward_batch = reward_batch.to(self.args.device)


        # Critic Update
        next_action_batch = self.actor_target.forward(next_state_batch)
        next_q = self.critic_target.forward(next_state_batch, next_action_batch)

        target_q = reward_batch + (self.gamma * next_q).detach()

        self.critic_optim.zero_grad()
        current_q = self.critic.forward(state_batch, action_batch)
        delta = (current_q - target_q).abs()
        dt = torch.mean(delta**2)
        dt.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10)
        self.critic_optim.step()

        # Actor Update
        self.actor_optim.zero_grad()

        policy_grad_loss = -(self.critic.forward(state_batch, self.actor.forward(state_batch))).mean()
        policy_loss = policy_grad_loss

        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
        self.actor_optim.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return policy_grad_loss.data.cpu().numpy(), delta.data.cpu().numpy()




class GeneticAgent:
    def __init__(self, args):

        self.args = args

        self.actor = Actor(args)
        self.actor_optim = Adam(self.actor.parameters(), lr=1e-3)

        self.buffer = repaly_memory.ReplayMemory(self.args.individual_bs)
        self.loss = nn.MSELoss()