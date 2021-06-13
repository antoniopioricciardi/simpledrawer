import os
import wandb
import argparse

from trainer import Trainer
from config.wandbKey import WANDB_KEY
from drawer_envs.simplegeometricshapesenv import *
from agents.agent import *
from agents.DuelingDDQNAgent import *
from agents.agent_ddqn_double_out import *
from various_tests.dueling_double_out.DuelingDDQNAgent_double_out import DuelingDDQNAgentDoubleOut

# TODO: Implement Pygame
# TODO: Environment in realtà è un "raccoglitore" di env. Con env.make('nomeenv') inizializziamo la simulazione scelta


# For offline metrics tracking
working_offline = False
parser = argparse.ArgumentParser()
parser.add_argument('-working_offline', type=bool, default=False, help='False if we do want to log wandb offline')
if working_offline:
    # if not existing, wandbKey.py file must be created, containing a single string variable named "WANDB_KEY"
    # with your wandb API key as a value
    os.environ["WANDB_API_KEY"] = WANDB_KEY  # YOUR_KEY_HERE
    os.environ["WANDB_MODE"] = "dryrun"

config_defaults = {
    'replace': 500,
    'learning_rate': 1e-3,
    'gamma': 0.6,
    'epsilon': 1,
    'epsilon_min': 0.0,
    'epsilon_dec': 1e-5, # 2.5e-6,#1e-5,
    'mem_size': 100000,
    'batch_size': 32, #64
    'optimizer': 'adam',
    'fc_layer_size': 1024,
    'max_steps': 1500000 #350000,
    # 'n_eval_games': 100,
    # 'eval_games_freq': 200,
    # 'n_test_games': 1000,
    # 'n_test_games_to_avg': 50,
}

if __name__ == '__main__':
    run = wandb.init(config=config_defaults)  # , project="prova")
    config = wandb.config
    side_length = 7
    max_steps = 50 #100
    n_train_games_to_avg = 50
    eval_games_freq = 200
    n_eval_games = 1
    sweeps_project_name = 'simpledrawerSEQUENTIALSHAPES-subtractcanvas-simultanousactions_' + str(side_length) + 'x' + str(side_length) + '_' +str(max_steps) + '_steps'
    tests_todo = ['duelingddqn_simplegeometricshapes']# ['ddqn_simplegeometricshapes']
    # TEST_N = 1  # 0 to 3 to choose the environment property from those in the list above
    test_name = tests_todo[0]
    print('#######################\nTraining/Testing env:', test_name, '\n#######################\n')

    name = test_name + '/lr' + str(config.learning_rate) + '_gamma' + str(config.gamma) + '_epsilon' + str(
        config.epsilon) + '_batch_size' + str(
        config.batch_size) + '_fc_size' + str(config.fc_layer_size)

    env = SimpleSequentialGeometricNonEpisodicShapeEnv(side_length, max_steps, random_starting_pos=False,
                                                       random_missing_pixel=False, subtract_canvas=True)
    wdb_trainer = Trainer(env, test_name, sweeps_project_name, n_train_games_to_avg, eval_games_freq, n_eval_games)

    # agent = AgentDoubleOut(env.num_states, env.num_actions, config.fc_layer_size, config.learning_rate, config.gamma,
    #                      config.epsilon, config.epsilon_min, config.epsilon_dec, config.replace,
    #                       config.mem_size, config.batch_size, name, wdb_trainer.models_path)
    agent = DuelingDDQNAgentDoubleOut(env.num_states, env.num_actions, config.fc_layer_size, config.learning_rate, config.gamma,
                           config.epsilon, config.epsilon_min, config.epsilon_dec, config.replace,
                           config.mem_size, config.batch_size, name, wdb_trainer.models_path)
    wdb_trainer.wandb_train(name, config, agent)
    # agent.epsilon=0.0
    # agent.load_models()
    n_test_games = 1
    # wdb_trainer.test(env, agent, n_test_games, name)
    #train(name, env, agent, wdb_trainer.plots_path, max_steps, n_train_games_to_avg, eval_games_freq, n_eval_games,
    #      using_wandb=True)





# OLDISSIMO
    # for TEST_N, test_name in enumerate(tests_todo):
    #     # test_name = tests_todo[TEST_N]
    #     print('#######################\nTraining/Testing env:', test_name, '\n#######################\n')
    #     # env = SimpleRandomGeometricNonEpisodicShapeEnv(side_length, max_steps, random_starting_pos=False)
    #     # env = SimpleRandomGeometricNonEpisodicShapeEnv(side_length, max_steps, random_starting_pos=False)
    #     env = SimpleSequentialGeometricNonEpisodicShapeEnv(side_length, max_steps, random_starting_pos=False, random_missing_pixel=False, subtract_canvas=True)
    #     # train_skip_wandb()
    #     # env = SimpleRandomGeometricShapeEnv(side_length, max_steps, random_starting_pos=False)
    #     # if TEST_N == 0:
    #     #     env = Environment(side_length, max_steps, random_starting_pos=False, random_horizontal_line=False)
    #     # elif TEST_N == 1:
    #     #     env = Environment(side_length, max_steps, random_starting_pos=True, random_horizontal_line=False)
    #     # elif TEST_N == 2:
    #     #     env = Environment(side_length, max_steps, random_starting_pos=False, random_horizontal_line=True)
    #     # elif TEST_N == 3:
    #     #     env = Environment(side_length, max_steps, random_starting_pos=True, random_horizontal_line=True)
    #     # elif TEST_N == 4:
    #     #     env = Environment(side_length, max_steps, random_starting_pos=False, random_horizontal_line=True, start_on_line=True)
    #     print('s0')
    #     wdb_trainer = WandbTrainer(config_defaults, sweeps_project_name=sweeps_project_name,
    #                                env=env, test_name=test_name, training=True, testing=False, games_to_avg=50)
    #     print('s1')
    #     wdb_trainer.train(config)

    # run.finish()
#
# wandb.login()
#
# sweep_config = {
#     'method': 'grid', #grid, random, bayesian
#     'metric': {
#       'name': 'reward',  # 'loss',
#       'goal':  'maximize'  # 'minimize'
#     },
#     'parameters': {
#         # 'epochs': {
#         #     'values': [2, 5, 10]
#         # },
#         'batch_size': {
#             #'values': [32, 64]
#             'values': [32]
#         },
#         'learning_rate': {
#             'values': [1e-3]
#         },
#         'gamma': {
#             # 'values': [0.6, 0.7, 0.9, 0.99]#[0.6, 0.7, 0.9]
#             'values': [0.6] #[0.6, 0.9]
#         },
#         'fc_layer_size': {
#             #'values': [64,128, 512]
#             'values': [1024]
#         },
#         # 'optimizer': {
#         #     'values': ['adam', 'sgd']
#         # },
#     }
# }
#
# '''
# filepath = os.path.join('shapes', 'line_1.png')
# img = Image.open(filepath)
# thresh = 200
# fn = lambda x: 255 if x > thresh else 0
# r = img.convert('L').point(fn, mode='1')
# r.save('foo.png')
# '''
#
# # Default values for hyper-parameters we're going to sweep over
#
# config_defaults = {
#     'replace': 500,
#     'learning_rate': 1e-3,
#     'gamma': 0.9,
#     'epsilon': 1,
#     'epsilon_min': 0.0,
#     'epsilon_dec': 1e-5, # 2.5e-6,#1e-5,
#     'mem_size': 100000,
#     'batch_size': 64,
#     'optimizer': 'adam',
#     'fc_layer_size': 128,
#     'max_steps': 1500000 #350000,
#     # 'n_eval_games': 100,
#     # 'eval_games_freq': 200,
#     # 'n_test_games': 1000,
#     # 'n_test_games_to_avg': 50,
# }
#
#
# def train_skip_wandb():
#     # env = SimpleRandomGeometricNonEpisodicShapeEnv(side_length, max_steps, random_starting_pos=False)
#     env = SimpleSequentialGeometricNonEpisodicShapeEnv(side_length, max_steps, random_starting_pos=False,
#                                                        random_missing_pixel=False, subtract_canvas=True)
#     replace = 1000
#     lr = 0.001
#     gamma = 0.6
#     epsilon = 1
#     epsilon_min = 0
#     epsilon_dec = 1e-5
#     # epsilon_dec = 2.5e-6  # from 1 to 0 in 400000 steps
#     mem_size = 1000000
#     batch_size = 32
#     # checkpoint_dir = config.checkpoint_dir
#
#     n_states = env.num_states
#     n_actions = env.num_actions - 2
#     n_hidden = 128
#     name = test_name + '/lr' + str(lr) + '_gamma' + str(gamma) + '_epsilon' + str(
#         epsilon) + '_batch_size' + str(batch_size) + '_fc_size' + str(n_hidden)
#     # agent = Agent(n_states, n_actions, n_hidden, lr, gamma, epsilon, epsilon_min, epsilon_dec, replace, mem_size,
#     # batch_size, name, 'models/')
#     # agent = Agent(n_states, n_actions, n_hidden, lr, gamma, epsilon, epsilon_min, epsilon_dec, replace, mem_size,
#     #               batch_size, name, 'models/')
#     # agent = DuelingDDQNAgent(n_states, n_actions, n_hidden, lr, gamma, epsilon, epsilon_min, epsilon_dec, replace, mem_size,
#     #              batch_size, name, 'models/')
#     agent = AgentDoubleOut(n_states, n_actions, n_hidden, lr, gamma, epsilon, epsilon_min, epsilon_dec, replace, mem_size,
#                  batch_size, name, 'models/')
#     train(name, env, agent, n_train_games_to_avg=50, eval_games_freq=1000, n_eval_games=50, plots_path='plots/', max_steps=50)


