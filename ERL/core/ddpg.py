import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np


class Actor(nn.Module):

    def __init__(self, args):
        super(Actor, self).__init__()
        self.args = args
        l1 = 128; l2 = 128; l3 = l2

        self.net = nn.Sequential(
            nn.Linear(args.observation_dim, l1),
            nn.LayerNorm(l1),
            nn.Linear(hidden_size, n_action)
        )
