

import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, hidden_layers_neurons):
        super(DQN, self).__init__()

        hidden_layers_neurons = list(filter(lambda v: v != 0, hidden_layers_neurons))  # removes 0 entries
        assert len(hidden_layers_neurons) > 0, "ERROR: hidden_layers_neurons len was not > 0."

        # adding layers
        # layer 0
        self.layers = [nn.Linear(n_observations, hidden_layers_neurons[0])]

        # intermediate layers if any
        for ih in range(len(hidden_layers_neurons)-1):
            self.layers.append(nn.Linear(hidden_layers_neurons[ih], hidden_layers_neurons[ih+1]))

        # layer -1
        self.layers.append(nn.Linear(hidden_layers_neurons[-1], n_actions))
        self.layers = nn.ParameterList(self.layers)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = F.relu(layer(out))
        return out
