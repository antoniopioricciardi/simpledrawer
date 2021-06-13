import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DuelingDDQNDoubleOut(nn.Module):
    def __init__(self, input_n, output_n, n_hidden=256, lr=1e-3, name='Duelingddqn', checkpoint_dir='models'):

        """
        :param lr:
        :param input_n:
        :param output_n: number of actions
        :param name: name of the network, for saving
        :param checkpoint_dir: directory in which to save the network
        """
        super(DuelingDDQNDoubleOut, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.fc1 = nn.Linear(input_n, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.act_value = nn.Linear(n_hidden, 1)  # find the value of a given set of states (therefore a single output for each element in the batch)
        self.act_advantage = nn.Linear(n_hidden, output_n)  #  advantage tells the advantage of each action at a given set of states
        self.pen_value = nn.Linear(n_hidden,
                                   1)  # find the value of a given set of states (therefore a single output for each element in the batch)
        self.pen_advantage = nn.Linear(n_hidden,
                                       2)  #  advantage tells the advantage of each action at a given set of states

        # self.fc1 = nn.Linear(input_n, 512)
        # self.value = nn.Linear(512, 1)  # find the value of a given set of states (therefore a single output for each element in the batch)
        # self.advantage = nn.Linear(512, output_n)  # advantage tells the advantage of each action at a given set of states

        #self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))  # action values
        act_value = self.act_value(x)
        act_advantages = self.act_advantage(x)
        pen_value = self.pen_value(x)
        pen_advantages = self.pen_advantage(x)
        return act_value, act_advantages, pen_value, pen_advantages

    def save_checkpoint(self):
        print('....saving model....')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        if self.device.type == 'cpu':
            self.load_state_dict(torch.load(self.checkpoint_file, map_location=torch.device('cpu')))
        else:
            self.load_state_dict(torch.load(self.checkpoint_file))
