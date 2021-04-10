import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DDQN(nn.Module):
    def __init__(self, input_n, output_n, n_hidden=256, lr=1e-3, name='dqn', checkpoint_dir='models'):
        """
        :param lr:
        :param input_n:
        :param output_n: number of actions
        :param name: name of the network, for saving
        :param checkpoint_dir: directory in which to save the network
        """
        super(DDQN, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.fc1 = nn.Linear(input_n, n_hidden)
        self.fc2 = nn.Linear(n_hidden, output_n)
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        # self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        #self.loss = nn.SmoothL1Loss()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        actions = self.fc2(x)  # action values

        return actions

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        if self.device.type == 'cpu':
            self.load_state_dict(torch.load(self.checkpoint_file, map_location=torch.device('cpu')))
        else:
            self.load_state_dict(torch.load(self.checkpoint_file))
