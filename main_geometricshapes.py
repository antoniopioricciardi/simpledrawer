import wandb

# from drawer_envs.drawerenv_random_geometricshape_nonepisodic import SimpleRandomGeometricNonEpisodicShapeEnv
from wandb_sweeper import WandbTrainer
from drawer_envs.simplegeometricshapesenv import *
from agents.agent import *
from agents.DuelingDDQNAgent import *

# TODO: Implement Pygame
# TODO: Environment in realtà è un "raccoglitore" di env. Con env.make('nomeenv') inizializziamo la simulazione scelta


wandb.login()

sweep_config = {
    'method': 'grid', #grid, random, bayesian
    'metric': {
      'name': 'reward',  # 'loss',
      'goal':  'maximize'  # 'minimize'
    },
    'parameters': {
        # 'epochs': {
        #     'values': [2, 5, 10]
        # },
        'batch_size': {
            #'values': [32, 64]
            'values': [32]
        },
        'learning_rate': {
            'values': [1e-3]
        },
        'gamma': {
            'values': [0.6, 0.7, 0.9, 0.99]#[0.6, 0.7, 0.9]
            #'values': [0.6] #[0.6, 0.9]
        },
        'fc_layer_size': {
            #'values': [64,128, 512]
            'values': [1024]
        },
        # 'optimizer': {
        #     'values': ['adam', 'sgd']
        # },
    }
}

'''
filepath = os.path.join('shapes', 'line_1.png')
img = Image.open(filepath)
thresh = 200
fn = lambda x: 255 if x > thresh else 0
r = img.convert('L').point(fn, mode='1')
r.save('foo.png')
'''

# Default values for hyper-parameters we're going to sweep over

config_defaults = {
    'replace': 500,
    'learning_rate': 1e-3,
    'gamma': 0.9,
    'epsilon': 1,
    'epsilon_min': 0.0,
    'epsilon_dec': 1e-5, # 2.5e-6,#1e-5,
    'mem_size': 100000,
    'batch_size': 64,
    'optimizer': 'adam',
    'fc_layer_size': 128,
    'max_steps': 1600000 #350000,
    # 'n_eval_games': 100,
    # 'eval_games_freq': 200,
    # 'n_test_games': 1000,
    # 'n_test_games_to_avg': 50,
}


def train_skip_wandb():
    env = SimpleRandomGeometricNonEpisodicShapeEnv(side_length, max_steps, random_starting_pos=False)
    replace = 1000
    lr = 0.001
    gamma = 0.6
    epsilon = 1
    epsilon_min = 0
    epsilon_dec = 1e-5
    # epsilon_dec = 2.5e-6  # from 1 to 0 in 400000 steps
    mem_size = 1000000
    batch_size = 32
    # checkpoint_dir = config.checkpoint_dir

    n_states = env.num_states
    n_actions = env.num_actions
    n_hidden = 128
    name = test_name + '/lr' + str(lr) + '_gamma' + str(gamma) + '_epsilon' + str(
        epsilon) + '_batch_size' + str(batch_size) + '_fc_size' + str(n_hidden)
    # agent = Agent(n_states, n_actions, n_hidden, lr, gamma, epsilon, epsilon_min, epsilon_dec, replace, mem_size,
    # batch_size, name, 'models/')
    # agent = Agent(n_states, n_actions, n_hidden, lr, gamma, epsilon, epsilon_min, epsilon_dec, replace, mem_size,
    #               batch_size, name, 'models/')
    agent = DuelingDDQNAgent(n_states, n_actions, n_hidden, lr, gamma, epsilon, epsilon_min, epsilon_dec, replace, mem_size,
                  batch_size, name, 'models/')
    train(env, agent, name, n_train_games_to_avg=50, eval_games_freq=1000, n_eval_games=50)


from trainer import train
from agents.agent import Agent
if __name__ == '__main__':
    side_length = 7
    max_steps = 100
    sweeps_project_name = 'simpledrawerSEQUENTIALSHAPES_' + str(side_length) + 'x' + str(side_length) + '_' +str(max_steps) + '_steps'
    tests_todo = ['duelingddqn_simplegeometricshapes']# ['ddqn_simplegeometricshapes']
    # TEST_N = 1  # 0 to 3 to choose the environment property from those in the list above
    for TEST_N, test_name in enumerate(tests_todo):
        # test_name = tests_todo[TEST_N]
        print('#######################\nTraining/Testing env:', test_name, '\n#######################\n')
        # env = SimpleRandomGeometricNonEpisodicShapeEnv(side_length, max_steps, random_starting_pos=False)
        # env = SimpleRandomGeometricNonEpisodicShapeEnv(side_length, max_steps, random_starting_pos=False)
        env = SimpleSequentialGeometricNonEpisodicShapeEnv(side_length, max_steps, random_starting_pos=False, random_missing_pixel=False)
        # train_skip_wandb()
        # env = SimpleRandomGeometricShapeEnv(side_length, max_steps, random_starting_pos=False)
        # if TEST_N == 0:
        #     env = Environment(side_length, max_steps, random_starting_pos=False, random_horizontal_line=False)
        # elif TEST_N == 1:
        #     env = Environment(side_length, max_steps, random_starting_pos=True, random_horizontal_line=False)
        # elif TEST_N == 2:
        #     env = Environment(side_length, max_steps, random_starting_pos=False, random_horizontal_line=True)
        # elif TEST_N == 3:
        #     env = Environment(side_length, max_steps, random_starting_pos=True, random_horizontal_line=True)
        # elif TEST_N == 4:
        #     env = Environment(side_length, max_steps, random_starting_pos=False, random_horizontal_line=True, start_on_line=True)

        wdb_trainer = WandbTrainer(config_defaults, sweep_config, sweeps_project_name=sweeps_project_name,
                                   env=env, test_name=test_name, training=True, testing=False, games_to_avg=50)
        wdb_trainer.do_sweeps()
