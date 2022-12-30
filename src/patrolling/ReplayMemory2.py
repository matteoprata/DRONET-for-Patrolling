

from collections import namedtuple, deque
import random

Transition = namedtuple('Transition', ("previous_states", "current_states", "actions", "rewards"))


class ReplayMemory(object):

    def __init__(self, capacity, random_state):
        self.memory = deque([], maxlen=capacity)
        self.random_state = random_state

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
