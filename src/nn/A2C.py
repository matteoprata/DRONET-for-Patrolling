
import torch
from pytorch_lightning import LightningModule
import torch.nn.functional as fn
import numpy as np


class A2C(torch.nn.Module):
    def __init__(self, in_shape, n_h1, n_h2, n_h3, out_shape):
        super(A2C, self).__init__()

        # NN Body
        self.l1 = torch.nn.Linear(in_shape, n_h1)
        self.l2 = torch.nn.Linear(n_h1, n_h2)
        self.l3 = torch.nn.Linear(n_h2, n_h3)

        self.head1_policy = torch.nn.Linear(n_h3, out_shape)
        self.head2_value = torch.nn.Linear(n_h3, 1)

        self.double()

    def forward(self, state):
        h1 = fn.relu(self.l1(state))
        h2 = fn.relu(self.l2(h1))
        h3 = self.l3(h2)

        return fn.softmax(self.head1_policy(h3), dim=-1), self.head2_value(h3)

# https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py