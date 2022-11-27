import random
from collections import namedtuple
import torch
import numpy as np
Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):

#capacity = buffer_size
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.device = 'cuda'

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        reshaped_args = []
        for arg in args:
            reshaped_args.append(np.reshape(arg, (1, -1)))

        self.memory[self.position] = Transition(*reshaped_args)
        self.position = (self.position + 1) % self.capacity


    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))

        state = torch.FloatTensor(np.concatenate(batch.state)).to(self.device)
        action = torch.FloatTensor(np.concatenate(batch.action)).to(self.device)
        next_state = torch.FloatTensor(np.concatenate(batch.next_state)).to(self.device)
        # Load everything to GPU if not already
        done = torch.FloatTensor(np.concatenate(batch.done)).to(self.device)
        reward = torch.FloatTensor(np.concatenate(batch.reward)).to(self.device)

        return state, action, next_state, reward, done

    def __len__(self):
        return len(self.memory)