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
from various_tests.stoppable_drawing.trainer_stoppable import StoppableTrainer
from various_tests.stoppable_drawing.drawer_envs.stoppable_simplegeometricshapesenv import StoppableSimpleSequentialGeometricNonEpisodicShapeEnv

from various_tests.cnn.cnn_trainer import CNNStoppableTrainer
from various_tests.cnn.drawer_envs.cnn_stoppable_simplegeometricshapesenv import CNNStoppableSimpleSequentialGeometricNonEpisodicShapeEnv
from various_tests.cnn.agents.cnn_agent_ddqn_double_out import CNNAgentDoubleOut


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
    'gamma': 0.9,
    'epsilon': 1,
    'epsilon_min': 0.0,
    'epsilon_dec': 1e-5, # 2.5e-6,#1e-5,
    'mem_size': 20000,  # 100000,
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
    training = False
    testing = True
    run = wandb.init(config=config_defaults)  # , project="prova")
    config = wandb.config
    side_length = 5
    max_steps = 50  # 100
    n_train_games_to_avg = 50
    eval_games_freq = 200
    n_eval_games = 1
    sweeps_project_name = 'simpledrawerSEQUENTIALSHAPES-subtractcanvas-simultaneousactions_' + str(side_length) + 'x' + str(side_length) + '_' +str(max_steps) + '_steps'
    tests_todo = ['duelingddqn_simplegeometricshapes']# ['ddqn_simplegeometricshapes']
    # TEST_N = 1  # 0 to 3 to choose the environment property from those in the list above
    test_name = tests_todo[0]
    print('#######################\nTraining/Testing env:', test_name, '\n#######################\n')

    name = test_name + '/lr' + str(config.learning_rate) + '_gamma' + str(config.gamma) + '_epsilon' + str(
        config.epsilon) + '_batch_size' + str(
        config.batch_size) + '_fc_size' + str(config.fc_layer_size)

    # env = SimpleSequentialGeometricNonEpisodicShapeEnv(side_length, max_steps, random_starting_pos=False,
    #                                                    random_missing_pixel=False, subtract_canvas=True)
    env = CNNStoppableSimpleSequentialGeometricNonEpisodicShapeEnv(side_length, max_steps, random_starting_pos=False, random_missing_pixel=False, subtract_canvas=True)
    # wdb_trainer = Trainer(env, test_name, sweeps_project_name, n_train_games_to_avg, eval_games_freq, n_eval_games)
    wdb_trainer = CNNStoppableTrainer(env, test_name, sweeps_project_name, n_train_games_to_avg, eval_games_freq, n_eval_games)

    agent = CNNAgentDoubleOut(env.obs_space, env.num_actions, config.fc_layer_size, config.learning_rate, config.gamma,
                         config.epsilon, config.epsilon_min, config.epsilon_dec, config.replace,
                          config.mem_size, config.batch_size, name, wdb_trainer.models_path)
    # agent = DuelingDDQNAgentDoubleOut(env.num_states, env.num_actions, config.fc_layer_size, config.learning_rate, config.gamma,
    #                        config.epsilon, config.epsilon_min, config.epsilon_dec, config.replace,
    #                        config.mem_size, config.batch_size, name, wdb_trainer.models_path)
    if training:
        wdb_trainer.wandb_train(name, config, agent)
    if testing:
        agent.epsilon=0.0
        agent.load_models()
        n_test_games = 1
        wdb_trainer.test(env, agent, n_test_games, name)

    #train(name, env, agent, wdb_trainer.plots_path, max_steps, n_train_games_to_avg, eval_games_freq, n_eval_games,
    #      using_wandb=True)
