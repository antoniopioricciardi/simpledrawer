import wandb

from environment import Environment
from wandb_trainer import WandbTrainer

wandb.login()

TRAINING = True  # False if we want to skip training
TESTING = False  # True if we want to test

tests_todo = ['ddqn_00_start_fixed_line', 'ddqn_random_start_fixed_line_pos', 'ddqn_00_start_random_line_pos', 'ddqn_random_start_random_line_pos']
TEST_N = 1  # 0 to 3 to choose the environment property from those in the list above

global test_name
test_name = tests_todo[TEST_N]

side_length = 10
max_steps = 60
if TEST_N == 0:
    env = Environment(side_length, max_steps, random_starting_pos=False, random_horizontal_line=False)
elif TEST_N == 1:
    env = Environment(side_length, max_steps, random_starting_pos=True, random_horizontal_line=False)
elif TEST_N == 2:
    env = Environment(side_length, max_steps, random_starting_pos=False, random_horizontal_line=True)
elif TEST_N == 3:
    env = Environment(side_length, max_steps, random_starting_pos=True, random_horizontal_line=True)

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
    'max_steps': 1000000 #350000,
    # 'n_eval_games': 100,
    # 'eval_games_freq': 200,
    # 'n_test_games': 1000,
    # 'n_test_games_to_avg': 50,
}

if __name__ == '__main__':
    wdb_trainer = WandbTrainer(config_defaults, sweep_config, sweeps_project_name="simpledrawer_10x10-", env=env, test_name=test_name, training=True, testing=False, games_to_avg=50)
    wdb_trainer.do_sweeps()
    # wandb.agent(sweep_id, train_test)

    # agent = Agent(n_states, n_actions, n_hidden, lr, gamma, epsilon, epsilon_min, epsilon_dec, replace, mem_size, batch_size, name, checkpoint_dir)
    #if TRAINING:
    #    wandb.agent(sweep_id, train)
    #if TESTING:
    #    test()
