import wandb

from environment import Environment
from wandb_trainer import WandbTrainer

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
            'values': [64]#[32, 64]
        },
        'learning_rate': {
            'values': [1e-3]
        },
        'gamma': {
            'values': [0.9] #[0.6, 0.9]
            #'values': [0.6, 0.9, 0.99]
        },
        'fc_layer_size': {
            'values': [512]#[64,128,256, 512]
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
    'replace': 1000,
    'learning_rate': 1e-3,
    'gamma': 0.6,
    'epsilon': 1,
    'epsilon_min': 0.0,
    'epsilon_dec': 1e-5,
    'mem_size': 50000,
    'batch_size': 64,
    'optimizer': 'adam',
    'fc_layer_size': 128,
    'max_steps': 500000 #350000,
    # 'n_eval_games': 100,
    # 'eval_games_freq': 200,
    # 'n_test_games': 1000,
    # 'n_test_games_to_avg': 50,
}

from environment_multishape import MultiShapeEnv
if __name__ == '__main__':
    side_length = 4
    max_steps = 60

    test_name = 'fixed_square'
    env = MultiShapeEnv(side_length, max_steps)
    wdb_trainer = WandbTrainer(config_defaults, sweep_config, sweeps_project_name="simpledrawer_4x4", env=env,
                               test_name=test_name, training=True, testing=True, games_to_avg=50)
    wdb_trainer.do_sweeps()

    # tests_todo = ['ddqn_00_start_fixed_line', 'ddqn_random_start_fixed_line_pos', 'ddqn_00_start_random_line_pos',
    #               'ddqn_random_start_random_line_pos', 'ddqn_start_on_line_random_line_pos']
    # # TEST_N = 1  # 0 to 3 to choose the environment property from those in the list above
    # for TEST_N, test_name in enumerate(tests_todo):
    #     if TEST_N != 4:
    #         continue
    #     # test_name = tests_todo[TEST_N]
    #     print('#######################\nTraining/Testing env:', test_name, '#######################\n')
    #     if TEST_N == 0:
    #         env = Environment(side_length, max_steps, random_starting_pos=False, random_horizontal_line=False)
    #     elif TEST_N == 1:
    #         env = Environment(side_length, max_steps, random_starting_pos=True, random_horizontal_line=False)
    #     elif TEST_N == 2:
    #         env = Environment(side_length, max_steps, random_starting_pos=False, random_horizontal_line=True)
    #     elif TEST_N == 3:
    #         env = Environment(side_length, max_steps, random_starting_pos=True, random_horizontal_line=True)
    #     elif TEST_N == 4:
    #         env = Environment(side_length, max_steps, random_starting_pos=False, random_horizontal_line=True, start_on_line=True)
    #
    #     wdb_trainer = WandbTrainer(config_defaults, sweep_config, sweeps_project_name="simpledrawer_10x10", env=env,
    #                                test_name=test_name, training=True, testing=True, games_to_avg=50)
    #     wdb_trainer.do_sweeps()