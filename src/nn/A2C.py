
import torch
from pytorch_lightning import LightningModule
import torch.nn.functional as fn
import numpy as np


class A2C(LightningModule):
    def __init__(self, in_shape, out_shape, n_h1=10, n_h2=8, n_h3=6, lr=0.001):
        super(A2C, self).__init__()
        self.n_actions = out_shape

        # NN Body
        self.layers = (torch.nn.Linear(in_shape, n_h1),
                       torch.nn.ReLU(),
                       torch.nn.Linear(n_h1, n_h2),
                       torch.nn.ReLU(),
                       torch.nn.Linear(n_h2, n_h3))
        self.nn_body = torch.nn.Sequential(*self.layers)

        # NN Actor head, the policy
        self.layers_actor = (torch.nn.ReLU(),
                             torch.nn.Linear(n_h3, out_shape))
        self.nn_head_actor = torch.nn.Sequential(*self.layers_actor)

        # NN Critic head, the value
        self.layers_critic = (torch.nn.ReLU(),
                              torch.nn.Linear(n_h3, 1))
        self.nn_head_critic = torch.nn.Sequential(*self.layers_critic)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        body = self.nn_body(state)
        return fn.softmax(self.nn_head_actor(body)), self.nn_head_critic(body)

    def forward_get_action(self, state):
        fwd = self.forward(state)
        actions = [i for i in range(self.n_actions)]
        return np.random.choice(actions, p=fwd[0])
