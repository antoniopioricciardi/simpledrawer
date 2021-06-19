import os
import torch
import wandb
import numpy as np
import cv2

from PIL import Image
from agents.agent import Agent
from agents.DuelingDDQNAgent import DuelingDDQNAgent
from agents.agent_ddqn_double_out import AgentDoubleOut
from trainer_bk import train
from utils_plot import plot_scores, plot_scores_testing

class CNNStoppableTrainer:
    def __init__(self, env, test_name, sweeps_project_name, n_train_games_to_avg, eval_games_freq, n_eval_games):
        self.env = env
        self.test_name = test_name

        self.sweeps_project_name = sweeps_project_name
        self.test_name = test_name

        # self.plots_path = plots_path
        self.n_train_games_to_avg = n_train_games_to_avg
        self.eval_games_freq = eval_games_freq
        self.n_eval_games = n_eval_games

        self.eval_best_win_n = 0
        self.eval_best_score = -1000

        self.__create_paths()

    def __create_paths(self):
        if not os.path.exists('tests'):
            os.mkdir('tests')
        tests_sweepsproj_name = os.path.join('tests', self.sweeps_project_name)
        if not os.path.exists(tests_sweepsproj_name):
            os.mkdir(tests_sweepsproj_name)
        self.models_path = os.path.join(tests_sweepsproj_name, 'models')
        if not os.path.exists(self.models_path):
            os.mkdir(self.models_path)
        if not os.path.exists(os.path.join(self.models_path, self.test_name)):
            os.mkdir(os.path.join(self.models_path, self.test_name))

        self.plots_path = os.path.join(tests_sweepsproj_name, 'plots')
        if not os.path.exists(self.plots_path):
            os.mkdir(self.plots_path)
        if not os.path.exists(os.path.join(self.plots_path, self.test_name)):
            os.mkdir(os.path.join(self.plots_path, self.test_name))

    def wandb_train(self, name, config, agent):
        """NON HYPERPARAMETERS"""
        n_train_games_to_avg = 50
        n_eval_games = 10
        eval_games_freq = 200

        self.train(self.env, agent, config.max_steps, n_train_games_to_avg, eval_games_freq,
              n_eval_games, name, using_wandb=True)

    def test(self, env, agent, n_games, name, is_eval=False, using_wandb=False, do_render=True):
        self.eval(env, agent, n_games, name, is_eval=is_eval, using_wandb=using_wandb, do_render=do_render)

    def train(self, env, agent, max_steps, n_train_games_to_avg, eval_games_freq, n_eval_games, name, using_wandb=False):
        scores = []
        epsilon_history = []
        best_score = -1000
        best_win_pct = 0
        eval_best_win_n = 0
        test_win_pct = 0
        best_train_avg_score = -1000
        wins = 0
        max_steps = max_steps
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
            is_win = False
            while not done:
                n_steps += 1
                # TODO: shape_n not used, for now
                shape_n, image_state, pointer = state

                state = image_state
                action, pen_state = agent.choose_action(state)
                # action = random.randint(0,4)
                state_next, reward, done, is_win = env.step_simultaneous(action, pen_state)
                shape_n_next, image_state_next, pointer_next = state_next

                # TODO: Try to not cast done to int
                agent.store_transition(state, action, pen_state, reward, image_state_next, int(done))
                agent.learn()

                state = state_next

                score += reward
                score = round(score, 2)

            if is_win:
                wins += 1
            # Code below runs after each game
            if game_n % 200 == 0:
                print(score)
            scores.append(score)
            epsilon_history.append(agent.epsilon)
            # if np.mean(scores[-n_train_games_to_avg:]) >= best_train_avg_score:
            #     best_train_avg_score = np.mean(scores[-n_train_games_to_avg:])
            #     agent.save_models()
            if game_n % n_train_games_to_avg == 0:
                print('training recap after', n_steps, 'steps and', game_n,
                      'games.\n', '50 games avg SCORE:', np.mean(scores[-n_train_games_to_avg:]),
                      'eps:', agent.epsilon, '50 games win pct', wins / n_train_games_to_avg,
                      '\n')
                plot_scores(scores, epsilon_history, n_train_games_to_avg,
                            os.path.join(self.plots_path, name) + '.png')  # 'plots/' + name + '.png')
                if using_wandb:
                    wandb.log({"avg cumulative reward": np.mean(scores[-n_train_games_to_avg:])})
                    # wandb.log({"50 games avg reward": np.mean(scores[-n_train_games_to_avg:])})
                    wandb.log({"50 games pct wins": wins / n_train_games_to_avg * 100})
                    wandb.log({"epsilon": agent.epsilon})
                wins = 0
            '''########### EVALUATION ###########'''
            # TODO: un def test diverso da quello di sotto gia fatto
            # TODO: Creare choose action per testing
            if game_n % eval_games_freq == 0:
                self.eval(env, agent, n_eval_games, name, is_eval=True, using_wandb=using_wandb)

    def eval(self, env, agent, n_games, name, is_eval=False, using_wandb=False, do_render=False):
        if not is_eval:  # if it's not eval, it's testing
            agent.load_models()
        with torch.no_grad():
            step_count = 0
            is_win = False
            agent.is_training(False)
            eval_wins = 0
            eval_scores = []
            for test_game_idx in range(n_games):
                done = False
                eval_score = 0
                state = env.reset()
                while not done:
                    step_count += 1
                    if do_render:
                        self.env.render()
                    # print(agent.epsilon)
                    # if test_game_idx % 10 == 0:
                    #    env.print_debug()
                    shape_n, image_state, pointer = state

                    state = image_state
                    action, pen_state = agent.choose_action(state)
                    # action = random.randint(0,4)
                    state_next, reward, done, is_win = env.step_simultaneous(action, pen_state)
                    shape_n_next, image_state_next, pointer_next = state_next

                    # TODO: Try to not cast done to int
                    agent.store_transition(state, action, pen_state, reward, image_state_next, int(done))
                    agent.learn()

                    state = state_next

                    eval_score += reward
                    eval_score = round(eval_score, 2)

                eval_scores.append(eval_score)

                if is_win:
                    eval_wins += 1

            # test_win_pct = (eval_wins/n_eval_games) * 100
            if np.mean(eval_scores) >= self.eval_best_score and agent.epsilon == 0:
                self.eval_best_score = np.mean(eval_scores)
                if is_eval:
                    agent.save_models()

            # if eval_score >= self.eval_best_score and agent.epsilon == 0:
            #     self.eval_best_score = eval_score
            # if eval_wins >= self.eval_best_win_n and agent.epsilon == 0:
            #    self.eval_best_win_n = eval_wins
            #    # TODO: What do we prefer? An agent that achieves higher reward but does not draw 100% correct, or an agent that draws well but takes more time? Reward functions, however, could change.
                # if is_eval:
                #     agent.save_models()

            eval_or_test_name = 'eval' if is_eval else 'test'
            print('############################\n' + eval_or_test_name + '\n', n_games,
                  'games avg SCORE:', np.mean(eval_scores),
                  'win pct (%)', (eval_wins / n_games) * 100, '\n##################\n')
            if using_wandb:
                wandb.log({str(n_games) + " " + str(eval_or_test_name) + " games, win pct (%)": (eval_wins / n_games) * 100})
                wandb.log({str(n_games) + " " + str(eval_or_test_name) + " games, avg rewards": np.mean(eval_scores)})
            plot_scores_testing(eval_scores, n_games,
                                os.path.join(self.plots_path, name) + '_eval.png')  # 'plots/' + name + '_eval.png')
