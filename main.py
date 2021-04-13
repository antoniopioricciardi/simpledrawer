import os
import wandb
import pprint
import random
import numpy as np

from PIL import Image
from agent import Agent
from utils_plot import *
from environment import Environment

TRAINING = True
TESTING = True  # True if we want to skip training

tests_todo = ['ddqn_00_start_fixed_line', 'ddqn_random_start_fixed_line_pos', 'ddqn_00_start_random_line_pos', 'ddqn_random_start_random_line_pos']
TEST_N = 0  # 0 to 3 to choose the environment property from those in the list above

name = tests_todo[TEST_N]

if TEST_N == 0:
    env = Environment(random_starting_pos=False, random_horizontal_line=False)
elif TEST_N == 1:
    env = Environment(random_starting_pos=True, random_horizontal_line=False)
elif TEST_N == 2:
    env = Environment(random_starting_pos=False, random_horizontal_line=True)
elif TEST_N == 3:
    env = Environment(random_starting_pos=True, random_horizontal_line=True)


sweep_config = {
    'method': 'grid', #grid, random
    'metric': {
      'name': 'reward',  # 'loss',
      'goal':  'maximize'  # 'minimize'
    },
    'parameters': {
        # 'epochs': {
        #     'values': [2, 5, 10]
        # },
        'batch_size': {
            'values': [16, 32, 64]
        },
        'learning_rate': {
            'values': [1e-2, 1e-3]
        },
        'gamma': {
            'values': [0.99, 0.9, 0.6, 0.5, 0.1, 0.0]
        },
        'fc_layer_size': {
            'values': [32, 64, 128]
        }
        # 'fc_layer_size':{
        #     'values':[128,256,512]
        # },
        # 'optimizer': {
        #     'values': ['adam', 'sgd']
        # },
    }
}

sweep_id = wandb.sweep(sweep_config, project="simpledrawer-sweeps")


'''
filepath = os.path.join('shapes', 'line_1.png')
img = Image.open(filepath)
thresh = 200
fn = lambda x: 255 if x > thresh else 0
r = img.convert('L').point(fn, mode='1')
r.save('foo.png')
'''

# TODO: Un bel def train()
# Default values for hyper-parameters we're going to sweep over
config_defaults = {
    'epochs': 3000,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'gamma': 0.6,
    'optimizer': 'adam',
    'fc_layer_size': 128,
    'dropout': 0.5,
}

# Initialize a new wandb run
wandb.init(config=config_defaults)
# Config is a variable that holds and saves hyperparameters and inputs
config = wandb.config


def train():
    lr = config.learning_rate  # 0.001
    gamma = config.gamma  # 0.5
    epsilon = 0.7
    epsilon_min = 0.0
    epsilon_dec = 1e-5
    mem_size = 50000
    batch_size = config.batch_size  # 32
    checkpoint_dir = 'models'

    n_states = env.num_states
    n_actions = env.num_actions
    n_hidden = config.fc_layer_size  # 128

    n_epochs = config.epochs

    global name
    name += '_lr' + str(lr) + '_gamma' + str(gamma) + '_epsilon' + str(epsilon) + '_batch_size' + str(batch_size) + '_fc_size' + str(n_hidden)
    agent = Agent(n_states, n_actions, n_hidden, lr, gamma, epsilon, epsilon_min, epsilon_dec, replace, mem_size,
                  batch_size, name, checkpoint_dir)
    scores = []
    epsilon_history = []
    best_score = -1000
    best_win_pct = 0
    wins = 0
    for i in range(n_epochs):
        # if i==3000:
        #    env.random_starting = True
        done = False
        n_steps = 0
        score = 0
        state = env.reset()
        while not done:
            # print(agent.epsilon)
            if i % 200 == 0:
                env.render()
            n_steps += 1
            source, canvas, pointer = state
            state = np.append(source.reshape(-1), canvas.reshape(-1))
            state = np.append(state, pointer)
            action = agent.choose_action(state)
            # action = random.randint(0,4)
            state_, reward, done = env.step(action)
            source_, canvas_, pointer_ = state_
            if done:
                if np.array_equal(source_, canvas_):
                    wins += 1

            flat_shape_ = np.append(source.reshape(-1), canvas.reshape(-1))
            flat_shape_ = np.append(flat_shape_, pointer)

            agent.store_transition(state, action, reward, flat_shape_, int(done))
            agent.learn()

            state = state_

            score += reward
            score = round(score, 2)

        if i % 200 == 0:
            print(score)
        scores.append(score)
        epsilon_history.append(agent.epsilon)

        # if wins / 50 >= best_win_pct and agent.epsilon == 0:
            # best_win_pct = wins / 50

        if score >= best_score and agent.epsilon == 0:
            best_score = score
            agent.save_models()

        # if i > 900:
        #    print(score)
        if i % 50 == 0:
            print('############################\nepoch', i, '50 games avg SCORE:', np.mean(scores[-50:]),
                  'eps:', agent.epsilon, '50 games win pct', wins / 50, '\n##################\n')
            plot_scores(scores, epsilon_history, 50, 'plots/' + name + '.png')
            wandb.log({"50 games avg reward": np.mean(scores[-50:])})
            wandb.log({"50 games n wins": wins})
            wandb.log({"epsilon": agent.epsilon})
            wins = 0



replace = 1000
lr = config.learning_rate  # 0.001
gamma = config.gamma  # 0.5
epsilon = 0.0
epsilon_min = 0.0
epsilon_dec = 1e-5
mem_size = 50000
batch_size = config.batch_size # 32
checkpoint_dir = 'models'

n_states = env.num_states
n_actions = env.num_actions
n_hidden = config.fc_layer_size  # 128

n_epochs = config.epochs


def test():
    agent = Agent(n_states, n_actions, n_hidden, lr, gamma, epsilon, epsilon_min, epsilon_dec, replace, mem_size,
                  batch_size, name, checkpoint_dir)
    # keep track of wins
    starts_per_states = {i: 0 for i in range(n_states)}
    wins_per_states = {i: 0 for i in range(n_states)}
    losses_per_states = {i: 0 for i in range(n_states)}
    # TODO: Analizza con ogni starting point i risultati dell'agente
    '''TESTING'''
    scores = []
    agent.load_models()
    agent.is_training(training=False)
    n_epochs = 3000
    wins = 0
    for i in range(n_epochs):
        done = False
        n_steps = 0
        score = 0
        state = env.reset()
        starting_state = env.starting_pos
        starts_per_states[starting_state] += 1
        while not done:
            # print(agent.epsilon)
            if i % 200 == 0:
                env.render()
            n_steps += 1
            source, canvas, pointer = state
            state = np.append(source.reshape(-1), canvas.reshape(-1))
            state = np.append(state, pointer)
            action = agent.choose_action(state)
            # if i % 200 == 0:
            #    print(action)
            # action = random.randint(0,4)
            state_, reward, done = env.step(action)
            source_, canvas_, pointer_ = state_
            flat_shape_ = np.append(source.reshape(-1), canvas.reshape(-1))
            flat_shape_ = np.append(flat_shape_, pointer)

            state = state_

            score += reward
            score = round(score, 2)
        source_, canvas_, pointer_ = state_
        if np.array_equal(source_, canvas_):
            wins += 1
            wins_per_states[starting_state] += 1
        else:
            losses_per_states[starting_state] += 1

        # if i % 200 == 0:
        #    print(score)
        scores.append(score)

        # if i > 900:
        #    print(score)
        if i % 50 == 0:
            print('############################\nepoch', i, '50 games avg SCORE:', np.mean(scores[-50:]),
                  '50 games win pct',
                  wins / 50, '\n##################\n')
            plot_scores_testing(scores, 50, 'plots/' + name + '_eval.png')
            wins = 0

    print('Starts per states')
    print(starts_per_states)
    print('Wins per states')
    print(wins_per_states)
    print('#############')
    print('Losses per states')
    print(losses_per_states)

if __name__ == '__main__':
    # agent = Agent(n_states, n_actions, n_hidden, lr, gamma, epsilon, epsilon_min, epsilon_dec, replace, mem_size, batch_size, name, checkpoint_dir)
    if TRAINING:
        wandb.agent(sweep_id, train)

    if TESTING:
        test()
        # wandb.agent()