import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DuelingDDQN(nn.Module):
    def __init__(self, lr, input_n, output_n, name, checkpoint_dir):
        """

        :param lr:
        :param input_n:
        :param output_n: number of actions
        :param name: name of the network, for saving
        :param checkpoint_dir: directory in which to save the network
        """
        super(DuelingDDQN, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        # input_n[0] is the number of channels for the input images (4x1, 4 frames by one channel since we have grayscaled images)
        # 32 number of outgoing filters

        self.fc1 = nn.Linear(input_n, 512)
        self.value = nn.Linear(512, 1)  # find the value of a given set of states (therefore a single output for each element in the batch)
        self.advantage = nn.Linear(512, output_n)  # advantage tells the advantage of each action at a given set of states

        #self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        value = self.value(x)
        advantages = self.advantage(x)
        return value, advantages

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))