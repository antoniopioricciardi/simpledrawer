import os
import wandb
import pprint
import random
import numpy as np

from PIL import Image
from agent import Agent
from utils_plot import *
from environment import Environment
wandb.login()

TRAINING = True  # False if we want to skip training
TESTING = False  # True if we want to test

tests_todo = ['ddqn_00_start_fixed_line', 'ddqn_random_start_fixed_line_pos', 'ddqn_00_start_random_line_pos', 'ddqn_random_start_random_line_pos']
TEST_N = 0  # 0 to 3 to choose the environment property from those in the list above

global test_name
test_name = tests_todo[TEST_N]

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
            'values': [0.6]#, 0.9]
        },
        'fc_layer_size': {
            'values': [32, 64, 128, 256, 512]
        }
        # 'fc_layer_size':{
        #     'values':[128,256,512]
        # },
        # 'optimizer': {
        #     'values': ['adam', 'sgd']
        # },
    }
}

sweep_id = wandb.sweep(sweep_config, project="simpledrawer-" + test_name)


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


def train():
    global test_
    config_defaults = {
        # 'epochs': 3000,
        'max_steps': 200000,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'gamma': 0.6,
        'optimizer': 'adam',
        'fc_layer_size': 128,
    }

    # Initialize a new wandb run
    wandb.init(config=config_defaults)
    # Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config

    replace = 1000
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

    name = test_name + '_lr' + str(lr) + '_gamma' + str(gamma) + '_epsilon' + str(epsilon) + '_batch_size' + str(batch_size) + '_fc_size' + str(n_hidden)
    agent = Agent(n_states, n_actions, n_hidden, lr, gamma, epsilon, epsilon_min, epsilon_dec, replace, mem_size,
                  batch_size, name, checkpoint_dir)
    scores = []
    epsilon_history = []
    best_score = -1000
    best_win_pct = 0
    test_best_win_pct = 0
    test_win_pct = 0
    wins = 0
    n_test_games = 100
    max_steps = config.max_steps
    n_steps = 0
    game_n = 0
    while True:
        agent.eval_Q.train()
        if n_steps >= max_steps:
            break
        game_n += 1
        done = False
        score = 0
        state = env.reset()
        while not done:
            n_steps += 1
            # print(agent.epsilon)
            if game_n % 200 == 0:
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

        if game_n % 200 == 0:
            print(score)
        scores.append(score)
        epsilon_history.append(agent.epsilon)

        # if wins / 50 >= best_win_pct and agent.epsilon == 0:
        # best_win_pct = wins / 50

        # if score >= best_score and agent.epsilon == 0:
        #     best_score = score
        #     agent.save_models()

        # if i > 900:
        #    print(score)
        if game_n % 50 == 0:
            print('############################\ntraining recap after', n_steps, 'steps.\n', '50 games avg SCORE:', np.mean(scores[-50:]),
                  'eps:', agent.epsilon, '50 games win pct', wins / 50, '\n##################\n')
            plot_scores(scores, epsilon_history, 50, 'plots/' + name + '.png')
            wandb.log({"50 games avg reward": np.mean(scores[-50:])})
            wandb.log({"50 games n wins": wins/50*100})
            wandb.log({"epsilon": agent.epsilon})
            wins = 0

        '''########### TESTING ###########'''
        # TODO: un def test diverso da quello di sotto gia fatto
        # TODO: Rimuovere choose action per testing
        if game_n % 200 == 0:
            print('########### TESTING ###########')
            agent.eval_Q.eval()
            test_wins = 0
            test_score = 0
            test_scores = []
            for test_game_idx in range(n_test_games):
                while not done:
                    # print(agent.epsilon)
                    env.render()
                    n_steps += 1
                    source, canvas, pointer = state
                    state = np.append(source.reshape(-1), canvas.reshape(-1))
                    state = np.append(state, pointer)
                    action = agent.choose_action(state)
                    # action = random.randint(0,4)
                    state_, reward, done = env.step(action)
                    source_, canvas_, pointer_ = state_

                    state = state_

                    test_score += reward
                    test_score = round(score, 2)
                test_scores.append(test_score)
                if np.array_equal(source_, canvas_):
                    test_wins += 1

            test_win_pct = (test_wins/n_test_games) * 100
            if test_win_pct >= test_best_win_pct and agent.epsilon == 0:
                test_best_win_pct = test_win_pct
                agent.save_models()

            print('############################\ntest after', n_steps, 'iterations.\n', n_test_games, 'games avg SCORE:', np.mean(test_scores),
                      'win pct', (test_wins/n_test_games) * 100, '\n##################\n')
            wandb.log({str(n_test_games) + " test games win pct": test_wins/n_test_games*100})
            wandb.log({str(n_test_games) + " test games avg rewards": np.mean(test_scores)})
            plot_scores_testing(test_scores, n_test_games, 'plots/' + name + '_eval.png')


# for i in range(n_epochs):
#         # if i==3000:
#         #    env.random_starting = True
#         done = False
#         n_steps = 0
#         score = 0
#         state = env.reset()
#         while not done:
#             # print(agent.epsilon)
#             if i % 200 == 0:
#                 env.render()
#             n_steps += 1
#             source, canvas, pointer = state
#             state = np.append(source.reshape(-1), canvas.reshape(-1))
#             state = np.append(state, pointer)
#             action = agent.choose_action(state)
#             # action = random.randint(0,4)
#             state_, reward, done = env.step(action)
#             source_, canvas_, pointer_ = state_
#             if done:
#                 if np.array_equal(source_, canvas_):
#                     wins += 1
#
#             flat_shape_ = np.append(source.reshape(-1), canvas.reshape(-1))
#             flat_shape_ = np.append(flat_shape_, pointer)
#
#             agent.store_transition(state, action, reward, flat_shape_, int(done))
#             agent.learn()
#
#             state = state_
#
#             score += reward
#             score = round(score, 2)
#
#         if i % 200 == 0:
#             print(score)
#         scores.append(score)
#         epsilon_history.append(agent.epsilon)
#
#         # if wins / 50 >= best_win_pct and agent.epsilon == 0:
#             # best_win_pct = wins / 50
#
#         if score >= best_score and agent.epsilon == 0:
#             best_score = score
#             agent.save_models()
#
#         # if i > 900:
#         #    print(score)
#         if i % 50 == 0:
#             print('############################\nepoch', i, '50 games avg SCORE:', np.mean(scores[-50:]),
#                   'eps:', agent.epsilon, '50 games win pct', wins / 50, '\n##################\n')
#             plot_scores(scores, epsilon_history, 50, 'plots/' + name + '.png')
#             wandb.log({"50 games avg reward": np.mean(scores[-50:])})
#             wandb.log({"50 games n wins": wins})
#             wandb.log({"epsilon": agent.epsilon})
#             wins = 0

def test():
    replace = 1000
    lr = 0.001
    gamma = 0.9
    epsilon = 0.0
    epsilon_min = 0.0
    epsilon_dec = 1e-5
    mem_size = 50000
    batch_size = 32
    checkpoint_dir = 'models'

    n_states = env.num_states
    n_actions = env.num_actions
    n_hidden = 128

    name = 'ddqn_00_start_fixed_line_lr0.001_gamma0.9_epsilon0.7_batch_size32_fc_size128'

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
    n_epochs = 300
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
