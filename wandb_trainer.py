import os
import torch
import wandb
import numpy as np

from agent import Agent
from utils_plot import plot_scores, plot_scores_testing


class WandbTrainer:
    def __init__(self, config_defaults, sweep_config, sweeps_project_name, env, test_name, training=False, testing=True,
                 games_to_avg=50):
        self.config_defaults = config_defaults
        self.env = env
        self.test_name = test_name

        self.training = training
        self.testing = testing

        self.sweep_id = wandb.sweep(sweep_config, project=sweeps_project_name + test_name) #project="simpledrawer_test-"

        if not os.path.exists(os.path.join('models', test_name)):
            os.mkdir(os.path.join('models', test_name))
        if not os.path.exists(os.path.join('plots', test_name)):
            os.mkdir(os.path.join('plots', test_name))

    def do_sweeps(self):
        wandb.agent(self.sweep_id, self.__train_test)

    def __train_test(self):
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

        """NON HYPERPARAMETERS"""
        #training = False
        #testing = True
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

        replace = config.replace
        lr = config.learning_rate  # 0.001
        gamma = config.gamma  # 0.5
        epsilon = config.epsilon
        epsilon_min = config.epsilon_min
        epsilon_dec = config.epsilon_dec
        mem_size = config.mem_size
        batch_size = config.batch_size  # 32
        # checkpoint_dir = config.checkpoint_dir

        n_states = self.env.num_states
        n_actions = self.env.num_actions
        n_hidden = config.fc_layer_size  # 128

        name = self.test_name + '/lr' + str(lr) + '_gamma' + str(gamma) + '_epsilon' + str(epsilon) + '_batch_size' + str(
            batch_size) + '_fc_size' + str(n_hidden)

        agent = Agent(n_states, n_actions, n_hidden, lr, gamma, epsilon, epsilon_min, epsilon_dec, replace, mem_size,
                      batch_size, name, checkpoint_dir)

        if self.training:
            self.train(config, agent, name, n_train_games_to_avg, eval_games_freq, n_eval_games)
        if self.testing:
            agent.epsilon = 0.2
            self.test(agent, name, n_test_games, n_test_games_to_avg)

    def train(self, config, agent, name, n_train_games_to_avg, eval_games_freq, n_eval_games):
        scores = []
        epsilon_history = []
        best_score = -1000
        best_win_pct = 0
        eval_best_win_n = 0
        test_win_pct = 0
        best_train_avg_score = -1000
        wins = 0
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
            state = self.env.reset()
            while not done:
                n_steps += 1
                if game_n % 200 == 0:
                    self.env.print_debug()
                if game_n % 1000 == 0:
                    self.env.render()
                n_steps += 1
                source, canvas, pointer = state
                state = np.append(source.reshape(-1), canvas.reshape(-1))
                state = np.append(state, pointer)
                action = agent.choose_action(state)
                # action = random.randint(0,4)
                state_, reward, done = self.env.step(action)
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
            # if np.mean(scores[-n_train_games_to_avg:]) >= best_train_avg_score:
            #     best_train_avg_score = np.mean(scores[-n_train_games_to_avg:])
            #     agent.save_models()
            if game_n % n_train_games_to_avg == 0:
                print('############################\ntraining recap after', n_steps, 'steps and', game_n,
                      'games.\n', '50 games avg SCORE:', np.mean(scores[-n_train_games_to_avg:]),
                      'eps:', agent.epsilon, '50 games win pct', wins / n_train_games_to_avg,
                      '\n##################\n')
                plot_scores(scores, epsilon_history, n_train_games_to_avg, 'plots/' + name + '.png')
                wandb.log({"50 games avg reward": np.mean(scores[-n_train_games_to_avg:])})
                wandb.log({"50 games n wins": wins / n_train_games_to_avg * 100})
                wandb.log({"epsilon": agent.epsilon})
                wins = 0
            '''########### EVALUATION ###########'''
            # TODO: un def test diverso da quello di sotto gia fatto
            # TODO: Creare choose action per testing
            if game_n % eval_games_freq == 0:
                with torch.no_grad():
                    agent.is_training(False)
                    # agent.eval_Q.eval()
                    eval_wins = 0
                    eval_scores = []
                    for test_game_idx in range(n_eval_games):
                        done = False
                        eval_score = 0
                        state = self.env.reset()
                        while not done:
                            # print(agent.epsilon)
                            # if test_game_idx % 10 == 0:
                            #    env.print_debug()
                            source, canvas, pointer = state
                            state = np.append(source.reshape(-1), canvas.reshape(-1))
                            state = np.append(state, pointer)
                            action = agent.choose_action(state)
                            # action = random.randint(0,4)
                            state_, reward, done = self.env.step(action)
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

                    print('############################\nevaluation after', n_steps, 'iterations.\n', n_eval_games,
                          'games avg SCORE:', np.mean(eval_scores),
                          'win pct (%)', (eval_wins / n_eval_games) * 100, '\n##################\n')
                    wandb.log({str(n_eval_games) + " eval games, win pct (%)": (eval_wins / n_eval_games) * 100})
                    wandb.log({str(n_eval_games) + " eval games, avg rewards": np.mean(eval_scores)})
                    plot_scores_testing(eval_scores, n_eval_games, 'plots/' + name + '_eval.png')

    def test(self, agent, name, n_test_games, n_test_games_to_avg):
        # n_test_games = config.n_test_games
        # n_test_games_to_avg = config.n_test_games_to_avg

        n_states = self.env.num_states

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
                state = self.env.reset()
                starting_state = self.env.starting_pos
                starts_per_states[starting_state] += 1
                while not done:
                    self.env.render()
                    # print(agent.epsilon)
                    if test_game_idx % 50 == 0:
                        self.env.print_debug()
                    source, canvas, pointer = state
                    state = np.append(source.reshape(-1), canvas.reshape(-1))
                    state = np.append(state, pointer)
                    action = agent.choose_action(state)
                    # action = random.randint(0,4)
                    state_, reward, done = self.env.step(action)
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
                print('############################\n game', test_game_idx, '\nscore:', test_scores[-1], '- game',
                      game_result)

                # test_win_pct = (test_wins / n_test_games) * 100

                print('############################\n', test_game_idx, 'games tested.\n', n_test_games,
                      'games avg SCORE:',
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


