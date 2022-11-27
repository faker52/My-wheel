import random
from collections import namedtuple
import numpy as np
Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'done'))




def push(*args):
    """Saves a transition."""

    reshaped_args = []


    memory = []
    memory.append(Transition(*args))
    print(memory)


push(1,1,1,1,1)
