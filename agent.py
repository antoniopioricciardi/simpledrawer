import torch
import numpy as np

from ddqn import DDQN
from replaymemory import ReplayMemory

class Agent:
    def __init__(self, n_states, n_actions, n_hidden, lr, gamma, epsilon, epsilon_min, epsilon_dec, replace, mem_size,
                 batch_size, name, checkpoint_dir):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.epsilon_new_min = self.epsilon_min*0.1
        self.new_min = False

        self.replace_target_cnt = replace
        self.learn_step_counter = 0
        self.batch_size = batch_size

        self.memory = ReplayMemory(mem_size, n_states, n_actions)

        self.eval_Q = DDQN(n_states, n_actions, n_hidden, lr, name, checkpoint_dir)
        self.target_Q = DDQN(n_states, n_actions, n_hidden, lr, name, checkpoint_dir)

    def is_training(self, training=True):
        if training:
            self.eval_Q.train()
        else:
            self.eval_Q.eval()

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.n_actions)
        else:
            # input_dims is a batch, therefore we need to create a batch for every single observation
            state = torch.tensor([state], dtype=torch.float).to(self.eval_Q.device)
            actions = self.eval_Q.forward(state)
            action = torch.argmax(actions).item()
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def sample_memory(self):
        state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)
        # convert each memory element in a pytorch tensor and sent it to the device
        # (lowaercase .tensor() preserves the data type of the underlying numpy array
        states = torch.tensor(state).to(self.eval_Q.device)
        actions = torch.tensor(action).to(self.eval_Q.device)
        rewards = torch.tensor(reward).to(self.eval_Q.device)
        next_states = torch.tensor(next_state).to(self.eval_Q.device)
        dones = torch.tensor(done, dtype=torch.bool).to(self.eval_Q.device)

        return states, actions, rewards, next_states, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.target_Q.load_state_dict(self.eval_Q.state_dict())

    def learn(self):
        # if memory has not been filled yet, return
        if self.memory.mem_counter < self.batch_size:
            return
        self.eval_Q.optimizer.zero_grad()
        self.replace_target_network()

        states, actions, rewards, next_states, dones = self.sample_memory()
        indices = np.arange(self.batch_size)  # array of numbers in range [0, ..., batch_size]

        q_pred = self.eval_Q.forward(states)[indices, actions]
        q_next = self.target_Q.forward(next_states)
        q_eval = self.eval_Q.forward(next_states)

        next_actions = torch.argmax(q_eval, dim=1)

        q_next[dones] = 0.0
        target_next_q_pred = q_next[indices, next_actions]

        q_target = rewards + self.gamma * target_next_q_pred
        # q_target = rewards + self.gamma * q_next[indices, next_actions]

        loss = self.eval_Q.loss(q_target, q_pred).to(self.eval_Q.device)
        loss.backward()
        # clip_grad_norm_(self.eval_Q.parameters(), 10)
        self.eval_Q.optimizer.step()

        self.learn_step_counter += 1
        self.decrement_epsilon()

    def save_models(self):
        self.eval_Q.save_checkpoint()
        self.target_Q.save_checkpoint()

    def load_models(self):
        self.eval_Q.load_checkpoint()
        self.target_Q.load_checkpoint()

