import numpy as np
import torch
from replaymemory_multiout import ReplayMemory
from various_tests.dueling_double_out.DuelingDDQNDoubleOut import DuelingDDQNDoubleOut


class DuelingDDQNAgentDoubleOut:
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

        self.eval_DuelingDDQN = DuelingDDQNDoubleOut(n_states, n_actions, n_hidden, lr, name + '_eval', checkpoint_dir)
        # will be used to compute Q(s',a') - that is for the resulting states
        # We won't perform gradient descent/backprob on this net, only on the eval.
        self.target_DuelingDDQN = DuelingDDQNDoubleOut(n_states, n_actions, n_hidden, lr, name + '_target', checkpoint_dir)

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
            pen_state = np.random.choice(2)
        else:
            # input_dims is a batch, therefore we need to create a batch for every single observation
            state = torch.tensor([state], dtype=torch.float).to(self.eval_DuelingDDQN.device)
            # _, advantages = self.eval_DuelingDDQN.forward(state)
            _, act_advantages, _, pen_advantages = self.eval_DuelingDDQN.forward(state)
            action = torch.argmax(act_advantages).item()
            pen_state = torch.argmax(pen_advantages).item()
        return action, pen_state

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
        state = torch.tensor(state).to(self.eval_DuelingDDQN.device)

        _, act_advantages, _, pen_advantages = self.eval_DuelingDDQN.forward(state)
        action = torch.argmax(act_advantages).item()
        pen_state = torch.argmax(pen_advantages).item()

        return action, pen_state

    def store_transition(self, state, action, pen_state, reward, next_state, done):
        self.memory.store_transition(state, action, pen_state, reward, next_state, done)

    def sample_memory(self):
        state, action, pen_state, reward, next_state, done = self.memory.sample_buffer(self.batch_size)
        # convert each memory element in a pytorch tensor and sent it to the device
        # (lowaercase .tensor() preserves the data type of the underlying numpy array
        states = torch.tensor(state).to(self.eval_DuelingDDQN.device)
        actions = torch.tensor(action).to(self.eval_DuelingDDQN.device)
        pen_states = torch.tensor(pen_state).to(self.eval_DuelingDDQN.device)
        rewards = torch.tensor(reward).to(self.eval_DuelingDDQN.device)
        next_states = torch.tensor(next_state).to(self.eval_DuelingDDQN.device)
        dones = torch.tensor(done, dtype=torch.bool).to(self.eval_DuelingDDQN.device)

        return states, actions, pen_states, rewards, next_states, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.target_DuelingDDQN.load_state_dict(self.eval_DuelingDDQN.state_dict())

    def learn(self):
        # if memory has not been filled yet, return
        if self.memory.mem_counter < self.batch_size:
            return

        self.eval_DuelingDDQN.optimizer.zero_grad()
        self.replace_target_network()
        # TODO: wrap in a function
        states, actions, pen_states, rewards, next_states, dones = self.sample_memory()
        indices = np.arange(self.batch_size)  # array of numbers in range [0, ..., batch_size]
        # here we combine the dueling deep q network and double deep q n.
        act_value_states, act_adv_states, pen_value_states, pen_adv_states = self.eval_DuelingDDQN.forward(states)
        act_value_next, act_adv_next, pen_value_next, pen_adv_next = self.eval_DuelingDDQN.forward(next_states)
        act_target_val, act_target_adv, pen_target_val, pen_target_adv = self.target_DuelingDDQN.forward(next_states)

        act_q_pred = (act_value_states + (act_adv_states - torch.mean(act_adv_states, dim=1, keepdim=True)))[indices, actions]
        act_q_eval = act_value_next + (act_adv_next - torch.mean(act_adv_next, dim=1, keepdim=True))
        act_target_pred = (act_target_val + (act_target_adv - torch.mean(act_target_adv, dim=1, keepdim=True)))

        act_next_actions = torch.argmax(act_q_eval, dim=1)

        act_target_pred[dones] = 0

        act_q_target = rewards + self.gamma * act_target_pred[indices, act_next_actions.detach()]

        act_loss = self.eval_DuelingDDQN.loss(act_q_pred, act_q_target).to(self.eval_DuelingDDQN.device)

        pen_q_pred = (pen_value_states + (pen_adv_states - torch.mean(pen_adv_states, dim=1, keepdim=True)))[indices, pen_states]
        pen_q_eval = pen_value_next + (pen_adv_next - torch.mean(pen_adv_next, dim=1, keepdim=True))
        pen_target_pred = (pen_target_val + (pen_target_adv - torch.mean(pen_target_adv, dim=1, keepdim=True)))

        pen_next_actions = torch.argmax(pen_q_eval, dim=1)

        pen_target_pred[dones] = 0

        pen_q_target = rewards + self.gamma * pen_target_pred[indices, pen_next_actions.detach()]

        pen_loss = self.eval_DuelingDDQN.loss(pen_q_pred, pen_q_target).to(self.eval_DuelingDDQN.device)

        loss = act_loss + pen_loss
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
