import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperparameters import *


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


# sometimes called Actor
class Policy_network(nn.Module):

    def __init__(self, state_size, action_size, filename, hidden_layer_sizes=[512,256]):
        super(Policy_network, self).__init__()
        torch.manual_seed(SEED)
        self.hidden_layer_1 = nn.Linear(state_size, hidden_layer_sizes[0])
        self.hidden_layer_2 = nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1])
        self.output = nn.Linear(hidden_layer_sizes[1], action_size)
        # load weights and biases if given
        self.reset_parameters()
        if LOAD:
            self.load_state_dict(torch.load("neural_networks/"+filename))
            print("Values loaded")

    def reset_parameters(self):
        self.hidden_layer_1.weight.data.uniform_(*hidden_init(self.hidden_layer_1))
        self.hidden_layer_2.weight.data.uniform_(*hidden_init(self.hidden_layer_2))
        self.output.weight.data.uniform_(-3e-3, 3e-3)


    def forward(self, state):
        x = F.relu(self.hidden_layer_1(state))
        x = F.relu(self.hidden_layer_2(x))
        return torch.tanh(self.output(x))


    def evaluate(self, state, requires_grad):
        # set evaluation mode
        self.eval()
        if requires_grad:
            action_values = self.forward(state)
        # for evaluation no grad bc its faster
        else:
            with torch.no_grad():
                # compute action values
                action_values = self.forward(state)
        # set training mode
        self.train()
        return action_values

    def save(self,filename):
        torch.save(self.state_dict(), "neural_networks/"+filename)

# sometimes called critic
class Q_network(nn.Module):

    def __init__(self, state_size, action_size,filename, hidden_layer_sizes=[512,256]):
        super(Q_network, self).__init__()
        torch.manual_seed(SEED)
        self.state_size=state_size
        self.action_size=action_size
        self.hidden_layer_1 = nn.Linear(state_size*2+action_size*2, hidden_layer_sizes[0])
        self.hidden_layer_2 = nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1])
        self.output = nn.Linear(hidden_layer_sizes[1], 1)
        self.reset_parameters()
        # load weights and biases if given
        if LOAD:
            self.load_state_dict(torch.load("neural_networks/"+filename))
            print("Values loaded")

    def reset_parameters(self):
        self.hidden_layer_1.weight.data.uniform_(*hidden_init(self.hidden_layer_1))
        self.hidden_layer_2.weight.data.uniform_(*hidden_init(self.hidden_layer_2))
        self.output.weight.data.uniform_(-3e-3, 3e-3)


    # input: tensor with states-> shape (batchsize,number agents*state dimension)
    #        tensor with action-> shape (batchsize,number agents*action dimension)
    def forward(self, states, actions):
        input =torch.cat((states,actions), dim=1)
        x = F.relu(self.hidden_layer_1(input))
        x = F.relu(self.hidden_layer_2(x))
        return self.output(x)

    def evaluate(self, states,actions, requires_grad):
        # set evaluation mode
        self.eval()
        if requires_grad:
            quality_values = self.forward(states, actions)
        # for evaluation no grad bc its faster
        else:
            with torch.no_grad():
                # compute action values
                quality_values = self.forward(states, actions)
        # set training mode
        self.train()
        return quality_values

    def save(self,filename):
        torch.save(self.state_dict(), "neural_networks/"+filename)
