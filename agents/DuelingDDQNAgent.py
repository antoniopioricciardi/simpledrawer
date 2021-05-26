import numpy as np
import torch
from nnets import DuelingDDQN
from replaymemory import ReplayMemory
from nnets.DuelingDDQN import DuelingDDQN


class DuelingDDQNAgent:
    # def __init__(self, input_dims, n_actions, memory_size, batch_size,
    #              lr, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_dec=5e-7,
    #              replace=1000, algo=None, env_name=None, checkpoint_dir='models/'):
    def __init__(self, n_states, n_actions, n_hidden, lr, gamma, epsilon, epsilon_min, epsilon_dec, replace,
                 mem_size,
                 batch_size, name, checkpoint_dir):
        self.n_states = n_states
        self.n_actions = n_actions

        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec

        self.batch_size = batch_size
        self.replace_target_cnt = replace
        self.checkpoint_dir = checkpoint_dir
        self.learn_step_counter = 0

        self.memory = ReplayMemory(mem_size, n_states, n_actions)

        #self.eval_DuelingDDQN = DuelingDDQN(self.lr, self.n_states, self.n_actions, self.env_name+'_'+self.algo+'_q_eval',
        #                       self.checkpoint_dir)
        # will be used to compute Q(s',a') - that is for the resulting states
        # We won't perform gradient descent/backprob on this net, only on the eval.
        #self.target_DuelingDDQN = DuelingDDQN(self.lr, self.n_states, self.n_actions, self.env_name + '_' + self.algo + '_q_target',
        #                         self.checkpoint_dir)

        self.eval_DuelingDDQN = DuelingDDQN(n_states, n_actions, n_hidden, lr, name + '_eval', checkpoint_dir)
        # will be used to compute Q(s',a') - that is for the resulting states
        # We won't perform gradient descent/backprob on this net, only on the eval.
        self.target_DuelingDDQN = DuelingDDQN(n_states, n_actions, n_hidden, lr, name + '_target', checkpoint_dir)

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min

    def is_training(self, training=True):
        if training:
            self.eval_DuelingDDQN.train()
            self.target_DuelingDDQN.train()
        else:
            self.eval_DuelingDDQN.eval()
            self.target_DuelingDDQN.eval()

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.n_actions)
        else:
            # input_dims is a batch, therefore we need to create a batch for every single observation
            state = torch.tensor([state], dtype=torch.float).to(self.eval_DuelingDDQN.device)
            _, advantages = self.eval_DuelingDDQN.forward(state)
            action = torch.argmax(advantages).item()
        return action

    def choose_action_debug(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.n_actions)
        else:
            # input_dims is a batch, therefore we need to create a batch for every single observation
            state = torch.tensor([state], dtype=torch.float).to(self.eval_DuelingDDQN.device)
            value, advantages = self.eval_DuelingDDQN.forward(state)
            action = torch.argmax(advantages).item()
        return action, (value, advantages)

    def choose_action_eval(self, state):
        # input_dims is a batch, therefore we need to create a batch for every single observation
        state = torch.tensor(state).to(self.eval_Q.device)
        actions = self.eval_Q.forward(state)
        action = torch.argmax(actions).item()
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def sample_memory(self):
        state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)
        # convert each memory element in a pytorch tensor and sent it to the device
        # (lowaercase .tensor() preserves the data type of the underlying numpy array
        states = torch.tensor(state).to(self.eval_DuelingDDQN.device)
        actions = torch.tensor(action).to(self.eval_DuelingDDQN.device)
        rewards = torch.tensor(reward).to(self.eval_DuelingDDQN.device)
        next_states = torch.tensor(next_state).to(self.eval_DuelingDDQN.device)
        dones = torch.tensor(done, dtype=torch.bool).to(self.eval_DuelingDDQN.device)

        return states, actions, rewards, next_states, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.target_DuelingDDQN.load_state_dict(self.eval_DuelingDDQN.state_dict())

    def learn(self):
        # if memory has not been filled yet, return
        if self.memory.mem_counter < self.batch_size:
            return

        self.eval_DuelingDDQN.optimizer.zero_grad()
        self.replace_target_network()

        states, actions, rewards, next_states, dones = self.sample_memory()
        indices = np.arange(self.batch_size)  # array of numbers in range [0, ..., batch_size]

        # here we combine the dueling deep q network and double deep q n.
        value_states, adv_states = self.eval_DuelingDDQN.forward(states)
        value_next, adv_next = self.eval_DuelingDDQN.forward(next_states)
        target_val, target_adv = self.target_DuelingDDQN.forward(next_states)

        q_pred = (value_states + (adv_states - torch.mean(adv_states, dim=1, keepdim=True)))[indices, actions]
        q_eval = value_next + (adv_next - torch.mean(adv_next, dim=1, keepdim=True))
        target_pred = (target_val + (target_adv - torch.mean(target_adv, dim=1, keepdim=True)))

        next_actions = torch.argmax(q_eval, dim=1)

        target_pred[dones] = 0

        q_target = rewards + self.gamma * target_pred[indices, next_actions.detach()]

        loss = self.eval_DuelingDDQN.loss(q_pred, q_target).to(self.eval_DuelingDDQN.device)
        loss.backward()
        self.eval_DuelingDDQN.optimizer.step()

        self.learn_step_counter += 1
        self.decrement_epsilon()

    def save_models(self):
        self.eval_DuelingDDQN.save_checkpoint()
        self.target_DuelingDDQN.save_checkpoint()

    def load_models(self):
        self.eval_DuelingDDQN.load_checkpoint()
        self.target_DuelingDDQN.load_checkpoint()
