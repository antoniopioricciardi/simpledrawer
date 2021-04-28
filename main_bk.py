import os
import torch
import wandb
import pprint
import random
import numpy as np

# from PIL import Image
from agent import Agent
from utils_plot import *
from environment import Environment
# os.environ["WANDB_API_KEY"] = "18473359ab7daf7626d168b95162c11133996567"
# os.environ["WANDB_MODE"] = "dryrun"
wandb.login()

TRAINING = True  # False if we want to skip training
TESTING = False  # True if we want to test

tests_todo = ['ddqn_00_start_fixed_line', 'ddqn_random_start_fixed_line_pos', 'ddqn_00_start_random_line_pos', 'ddqn_random_start_random_line_pos']
TEST_N = 1  # 0 to 3 to choose the environment property from those in the list above

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
            'values': [64]#[32, 64]
        },
        'learning_rate': {
            'values': [1e-3]
        },
        'gamma': {
            'values': [0.6] #[0.6, 0.9]
        },
        'fc_layer_size': {
            'values': [512]#[64,128,256, 512]
        },
        # 'optimizer': {
        #     'values': ['adam', 'sgd']
        # },
    }
}

sweep_id = wandb.sweep(sweep_config, project="simpledrawer_test-" + test_name)

'''
filepath = os.path.join('shapes', 'line_1.png')
img = Image.open(filepath)
thresh = 200
fn = lambda x: 255 if x > thresh else 0
r = img.convert('L').point(fn, mode='1')
r.save('foo.png')
'''

# Default values for hyper-parameters we're going to sweep over


#TODO: inserisci parametri di train, quelli esistenti qui portali in train_test funct.
def train(config, agent, name, n_train_games_to_avg, eval_games_freq, n_eval_games):

    # replace = config.replace
    # lr = config.learning_rate  # 0.001
    # gamma = config.gamma  # 0.5
    # epsilon = config.epsilon
    # epsilon_min = config.epsilon_min
    # epsilon_dec = config.epsilon_dec
    # mem_size = config.mem_size
    # batch_size = config.batch_size  # 32
    # checkpoint_dir = config.checkpoint_dir
    #
    # n_states = env.num_states
    # n_actions = env.num_actions
    # n_hidden = config.fc_layer_size  # 128

    # name = test_name + '_lr' + str(lr) + '_gamma' + str(gamma) + '_epsilon' + str(epsilon) + '_batch_size' + str(batch_size) + '_fc_size' + str(n_hidden)
    # agent = Agent(n_states, n_actions, n_hidden, lr, gamma, epsilon, epsilon_min, epsilon_dec, replace, mem_size, batch_size, name, checkpoint_dir)
    scores = []
    epsilon_history = []
    best_score = -1000
    best_win_pct = 0
    eval_best_win_n = 0
    test_win_pct = 0
    best_train_avg_score = 0
    wins = 0
    # eval_games_freq = config.eval_games_freq
    # n_eval_games = config.n_eval_games
    max_steps = config.max_steps
    n_steps = 0
    game_n = 0
    while True:
        agent.is_training()
        if n_steps >= max_steps:
            break
        game_n += 1
        done = False
        score = 0
        state = env.reset()
        while not done:
            n_steps += 1
            if game_n % 200 == 0:
                env.print_debug()
            if game_n % 1000 == 0:
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

            # TODO: Try not casting done to int
            agent.store_transition(state, action, reward, flat_shape_, int(done))
            agent.learn()

            state = state_

            score += reward
            score = round(score, 2)

        # Code below runs after each game
        if game_n % 200 == 0:
            print(score)
        scores.append(score)
        epsilon_history.append(agent.epsilon)
        if np.mean(scores[-n_train_games_to_avg:]) >= best_train_avg_score:
            best_train_avg_score = np.mean(scores[-n_train_games_to_avg:])
            agent.save_models()
        if game_n % n_train_games_to_avg == 0:
            print('############################\ntraining recap after', n_steps, 'steps and', game_n, 'games.\n', '50 games avg SCORE:', np.mean(scores[-n_train_games_to_avg:]),
                  'eps:', agent.epsilon, '50 games win pct', wins / n_train_games_to_avg, '\n##################\n')
            plot_scores(scores, epsilon_history, n_train_games_to_avg, 'plots/' + name + '.png')
            wandb.log({"50 games avg reward": np.mean(scores[-n_train_games_to_avg:])})
            wandb.log({"50 games n wins": wins/n_train_games_to_avg*100})
            wandb.log({"epsilon": agent.epsilon})
            wins = 0
        '''########### EVALUATION ###########'''
        # TODO: un def test diverso da quello di sotto gia fatto
        # TODO: Creare choose action per testing
        if game_n % eval_games_freq == 0:
            continue
            print('########### TESTING ###########')
            with torch.no_grad():
                agent.is_training(False)
                # agent.eval_Q.eval()
                eval_wins = 0
                eval_scores = []
                for test_game_idx in range(n_eval_games):
                    done = False
                    eval_score = 0
                    state = env.reset()
                    while not done:
                        # print(agent.epsilon)
                        #if test_game_idx % 10 == 0:
                        #    env.print_debug()
                        source, canvas, pointer = state
                        state = np.append(source.reshape(-1), canvas.reshape(-1))
                        state = np.append(state, pointer)
                        action = agent.choose_action(state)
                        # action = random.randint(0,4)
                        state_, reward, done = env.step(action)
                        source_, canvas_, pointer_ = state_

                        state = state_

                        eval_score += reward
                        eval_score = round(eval_score, 2)
                    eval_scores.append(eval_score)
                    if np.array_equal(source_, canvas_):
                        # TODO: pprint(source_, canvas_)
                        eval_wins += 1

                # test_win_pct = (eval_wins/n_eval_games) * 100
                if eval_wins >= eval_best_win_n and agent.epsilon == 0:
                    eval_best_win_n = eval_wins
                    # TODO: What do we prefer? An agent that achieves higher reward but does not draw 100% correct, or an agent that draws well but takes more time? Reward functions, however, could change.
                    agent.save_models()

                print('############################\nevaluation after', n_steps, 'iterations.\n', n_eval_games, 'games avg SCORE:', np.mean(eval_scores),
                          'win pct (%)', (eval_wins/n_eval_games) * 100, '\n##################\n')
                wandb.log({str(n_eval_games) + " eval games, win pct (%)": (eval_wins/n_eval_games)*100})
                wandb.log({str(n_eval_games) + " eval games, avg rewards": np.mean(eval_scores)})
                plot_scores_testing(eval_scores, n_eval_games, 'plots/' + name + '_eval.png')


def test(agent, name, n_test_games, n_test_games_to_avg):
    # n_test_games = config.n_test_games
    # n_test_games_to_avg = config.n_test_games_to_avg

    n_states = env.num_states

    # keep track of wins
    starts_per_states = {i: 0 for i in range(n_states)}
    wins_per_states = {i: 0 for i in range(n_states)}
    losses_per_states = {i: 0 for i in range(n_states)}
    '''########### TESTING ###########'''
    # TODO: un def test diverso da quello di sotto gia fatto
    # TODO: creare choose action per testing
    print('########### TESTING ###########')
    # agent.eval_Q.eval()
    test_wins = 0
    test_scores = []
    agent.load_models()
    with torch.no_grad():
        agent.is_training(training=False)
        for test_game_idx in range(n_test_games):
            done = False
            test_score = 0
            game_result = 'lost'
            state = env.reset()
            starting_state = env.starting_pos
            starts_per_states[starting_state] += 1
            while not done:
                env.render()
                # print(agent.epsilon)
                if test_game_idx % 50 == 0:
                    env.print_debug()
                source, canvas, pointer = state
                state = np.append(source.reshape(-1), canvas.reshape(-1))
                state = np.append(state, pointer)
                action = agent.choose_action(state)
                # action = random.randint(0,4)
                state_, reward, done = env.step(action)
                source_, canvas_, pointer_ = state_
                state = state_

                test_score += reward
                test_score = round(test_score, 2)
            test_scores.append(test_score)
            if np.array_equal(source_, canvas_):
                test_wins += 1
                game_result = 'won'
                wins_per_states[starting_state] += 1
            else:
                losses_per_states[starting_state] += 1
            print('############################\n game', test_game_idx, '\nscore:', test_scores[-1], '- game', game_result)

            # test_win_pct = (test_wins / n_test_games) * 100

            print('############################\n', test_game_idx, 'games tested.\n', n_test_games, 'games avg SCORE:',
                  np.mean(test_scores), '\n win pct (%):', (test_wins / (test_game_idx + 1)) * 100)

        wandb.log({str(n_test_games) + " test games, avg score": np.mean(test_scores[n_test_games_to_avg:])})
        wandb.log({str(n_test_games) + " test games, win pct": test_wins / n_test_games * 100})

        plot_scores_testing(test_scores, n_test_games_to_avg, 'plots/' + name + '_test.png')

        print('Starts per states')
        print(starts_per_states)
        print('Wins per states')
        print(wins_per_states)
        print('#############')
        print('Losses per states')
        print(losses_per_states)


def test_nosweeps(agent, name, n_test_games, n_test_games_to_avg):
    # n_test_games = config.n_test_games
    # n_test_games_to_avg = config.n_test_games_to_avg

    n_states_source = (env.length * 2) -1

    # keep track of wins
    starts_per_states = {i: 0 for i in range(n_states_source)}
    wins_per_states = {i: 0 for i in range(n_states_source)}
    losses_per_states = {i: 0 for i in range(n_states_source)}
    '''########### TESTING ###########'''
    # TODO: un def test diverso da quello di sotto gia fatto
    # TODO: creare choose action per testing
    print('########### TESTING ###########')
    # agent.eval_Q.eval()
    test_wins = 0
    test_scores = []
    agent.load_models()
    with torch.no_grad():
        agent.is_training(training=False)
        for test_game_idx in range(n_test_games):
            done = False
            test_score = 0
            game_result = 'lost'
            state = env.reset()
            starting_state = env.starting_pos
            starts_per_states[starting_state] += 1
            while not done:
                # print(agent.epsilon)
                if test_game_idx % 50 == 0:
                    env.print_debug()
                source, canvas, pointer = state
                state = np.append(source.reshape(-1), canvas.reshape(-1))
                state = np.append(state, pointer)
                action = agent.choose_action(state)
                # action = random.randint(0,4)
                state_, reward, done = env.step(action)
                source_, canvas_, pointer_ = state_
                state = state_

                test_score += reward
                test_score = round(test_score, 2)
            test_scores.append(test_score)
            if np.array_equal(source_, canvas_):
                test_wins += 1
                game_result = 'won'
                wins_per_states[starting_state] += 1
            else:
                losses_per_states[starting_state] += 1

            print('############################\n game', test_game_idx, '\nscore:', test_scores[-1], '- game', game_result)

            # test_win_pct = (test_wins / n_test_games) * 100

            print('############################\n', test_game_idx, 'games tested.\n', n_test_games, 'games avg SCORE:',
                  np.mean(test_scores), '\n win pct (%):', (test_wins / test_game_idx) * 100)


        plot_scores_testing(test_scores, n_test_games_to_avg, 'plots/' + name + '_testnosweep.png')

        print('Starts per states')
        print(starts_per_states)
        print('Wins per states')
        print(wins_per_states)
        print('#############')
        print('Losses per states')
        print(losses_per_states)


def train_test():
    config_defaults = {
        'replace': 1000,
        'learning_rate': 1e-3,
        'gamma': 0.6,
        'epsilon': 0.8,
        'epsilon_min': 0.0,
        'epsilon_dec': 1e-5,
        'mem_size': 50000,
        'batch_size': 64,
        'optimizer': 'adam',
        'fc_layer_size': 128,
        'max_steps': 400000#350000,
        #'n_eval_games': 100,
        #'eval_games_freq': 200,
        #'n_test_games': 1000,
        #'n_test_games_to_avg': 50,
    }

    """NON HYPERPARAMETERS"""
    training = False
    testing = True
    # TODO: move them out of here
    checkpoint_dir = 'models'
    n_train_games_to_avg = 50
    n_eval_games = 100
    eval_games_freq = 200
    n_test_games = 1000
    n_test_games_to_avg = 50

    # Initialize a new wandb run
    wandb.init(config=config_defaults)
    # Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config

    # replace = config.replace
    # lr = config.learning_rate  # 0.001
    # gamma = config.gamma  # 0.5
    # epsilon = 0.7
    # epsilon_min = 0.0
    # epsilon_dec = 1e-5
    # mem_size = 50000
    # batch_size = config.batch_size  # 32
    # checkpoint_dir = 'models'

    replace = config.replace
    lr = config.learning_rate  # 0.001
    gamma = config.gamma  # 0.5
    epsilon = config.epsilon
    epsilon_min = config.epsilon_min
    epsilon_dec = config.epsilon_dec
    mem_size = config.mem_size
    batch_size = config.batch_size  # 32
    #checkpoint_dir = config.checkpoint_dir

    n_states = env.num_states
    n_actions = env.num_actions
    n_hidden = config.fc_layer_size  # 128

    if not os.path.exists(os.path.join('models', test_name)):
        os.mkdir(os.path.join('models', test_name))
    if not os.path.exists(os.path.join('plots', test_name)):
        os.mkdir(os.path.join('plots', test_name))
    name = test_name + '/lr' + str(lr) + '_gamma' + str(gamma) + '_epsilon' + str(epsilon) + '_batch_size' + str(
        batch_size) + '_fc_size' + str(n_hidden)

    agent = Agent(n_states, n_actions, n_hidden, lr, gamma, epsilon, epsilon_min, epsilon_dec, replace, mem_size,
                  batch_size, name, checkpoint_dir)

    if training:
        train(config, agent, name, n_train_games_to_avg, eval_games_freq, n_eval_games)
    if testing:
        agent.epsilon = 0.2
        test(agent, name, n_test_games, n_test_games_to_avg)





from wandb_trainer import WandbTrainer

config_defaults = {
    'replace': 1000,
    'learning_rate': 1e-3,
    'gamma': 0.6,
    'epsilon': 0.8,
    'epsilon_min': 0.0,
    'epsilon_dec': 1e-5,
    'mem_size': 50000,
    'batch_size': 64,
    'optimizer': 'adam',
    'fc_layer_size': 128,
    'max_steps': 400000  # 350000,
    # 'n_eval_games': 100,
    # 'eval_games_freq': 200,
    # 'n_test_games': 1000,
    # 'n_test_games_to_avg': 50,
}

if __name__ == '__main__':
    wdb_trainer = WandbTrainer(config_defaults, sweep_config, sweeps_project_name="simpledrawer_test-", env=env, test_name=test_name, training=False, testing=True, games_to_avg=50)
    wdb_trainer.do_sweeps()
    # wandb.agent(sweep_id, train_test)

    # agent = Agent(n_states, n_actions, n_hidden, lr, gamma, epsilon, epsilon_min, epsilon_dec, replace, mem_size, batch_size, name, checkpoint_dir)
    #if TRAINING:
    #    wandb.agent(sweep_id, train)
    #if TESTING:
    #    test()
