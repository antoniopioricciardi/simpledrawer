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


def show_q_values(side_length: int, x, y, values):
    # moving the position by 1 is necessary because we will then need to show extra possible movements TODO: riscrivi
    x+=1
    y+=1
    # TODO: INEFFICIENT. Every time we recreate a new matrix instead of modifying the old one. To be fixed.
    # MOREOVER: We only change the source matrix when reset is called.
    # initialize the matrix that will contain qvalues
    # +2 because we need to be able to rappresent every movement even when the agent is at the matrix boundaries
    scores_matrix = np.zeros((side_length+2, side_length+2, 3), dtype=np.uint8)
    positions = [(0,1), (0,-1), (-1,0),(1,0),(0,0)]

    # scale values in [0,255]
    values += values.min()
    values = values/values.max() * 255
    print(values)
    print('sss')
    for i in range(5):
        y_offset = positions[i][0]
        x_offset = positions[i][1]
        scores_matrix[x+x_offset][y+y_offset] = [128, values[i], 0]

    scores_matrix_img = Image.fromarray(scores_matrix)
    scores_matrix_img = np.uint8(scores_matrix_img)

    height, width, ch = scores_matrix_img.shape
    new_width, new_height = width + 1, height + 2  # width + width//20, height + height//8

    # Crate a new canvas with new width and height.
    source_background = np.ones((new_height, new_width, ch), dtype=np.uint8) * 125

    # New replace the center of canvas with original image
    padding_top, padding_left = 1, 0  # 60, 10

    source_background[padding_top:padding_top + height, padding_left:padding_left + width] = scores_matrix_img

    text1 = "Source image"
    text2 = "Canvas"
    text_color_list = np.array([255, 0, 0])
    text_color = (int(text_color_list[0]), int(text_color_list[1]), int(text_color_list[2]))
    # img1 = cv2.putText(source_background.copy(), text1, (int(0.25 * width), 30), cv2.FONT_HERSHEY_COMPLEX, 1,
    #                   text_color)
    # img2 = cv2.putText(canvas_background.copy(), text2, (int(0.25 * width), 30), cv2.FONT_HERSHEY_COMPLEX, 1,
    #                   text_color)

    # shape[1] is the width, it seems it needs to go first when resizing.
    final = cv2.resize(source_background, (source_background.shape[1] * 30, source_background.shape[0] * 30), interpolation=cv2.INTER_NEAREST)
    # cv2.imwrite("./debug.png", final)
    cv2.imshow("pr2", final)  # this prevents code from running with wandb after first sweep
    cv2.waitKey(300)


class Trainer:
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
                shape_n, source, canvas, pointer = state
                # source, canvas, pointer = state
                # state = np.append(source.reshape(-1), canvas.reshape(-1))
                state = source.reshape(-1)
                state = np.append(state, pointer)
                state = np.array(state,
                                 dtype=np.float32)  # prevent automatic casting to float64 (don't know why that happens though...)
                action, pen_state = agent.choose_action(state)
                # action = random.randint(0,4)
                state_next, reward, done, is_win = env.step_simultaneous(action, pen_state)
                shape_n_next, source_next, canvas_next, pointer_next = state_next
                # source_next, canvas_next, pointer_next = state_next
                # if done:
                # if np.array_equal(source_next, canvas_next):
                # if reward == 100:
                #    print('win')
                #    wins += 1

                # flat_state_next = np.append(source_next.reshape(-1), canvas_next.reshape(-1))
                flat_state_next = source_next.reshape(-1)
                flat_state_next = np.append(flat_state_next, pointer_next)

                # TODO: Try to not cast done to int
                agent.store_transition(state, action, pen_state, reward, flat_state_next, int(done))
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
            is_win = False
            agent.is_training(False)
            eval_wins = 0
            eval_scores = []
            for test_game_idx in range(n_games):
                done = False
                eval_score = 0
                state = env.reset()
                while not done:
                    if do_render:
                        self.env.render()
                    # print(agent.epsilon)
                    # if test_game_idx % 10 == 0:
                    #    env.print_debug()
                    shape_n, source, canvas, pointer = state
                    # source, canvas, pointer = state
                    # state = np.append(source.reshape(-1), canvas.reshape(-1))
                    state = source.reshape(-1)
                    state = np.append(state, pointer)
                    state = np.array(state,
                                     dtype=np.float32)  # prevent automatic casting to float64 (don't know why that happened though...)

                    action, pen_state = agent.choose_action(state)
                    # action = random.randint(0,4)
                    state_next, reward, done, is_win = env.step_simultaneous(action, pen_state)
                    shape_n_next, source_next, canvas_next, pointer_next = state_next

                    state = state_next

                    eval_score += reward
                    eval_score = round(eval_score, 2)

                eval_scores.append(eval_score)

                if is_win:
                    eval_wins += 1
            # test_win_pct = (eval_wins/n_eval_games) * 100
            # if np.mean(eval_scores) >= best_eval_score:
            #    best_eval_score = np.mean(eval_scores)
            #    agent.save_models()

            # if eval_score >= self.eval_best_score and agent.epsilon == 0:
            #     self.eval_best_score = eval_score
            if eval_wins >= self.eval_best_win_n and agent.epsilon == 0:
                self.eval_best_win_n = eval_wins
                # TODO: What do we prefer? An agent that achieves higher reward but does not draw 100% correct, or an agent that draws well but takes more time? Reward functions, however, could change.
                if is_eval:
                    agent.save_models()

            eval_or_test_name = 'eval' if is_eval else 'test'
            print('############################\n' + eval_or_test_name + '\n', n_games,
                  'games avg SCORE:', np.mean(eval_scores),
                  'win pct (%)', (eval_wins / n_games) * 100, '\n##################\n')
            if using_wandb:
                wandb.log({str(n_games) + " " + str(eval_or_test_name) + " games, win pct (%)": (eval_wins / n_games) * 100})
                wandb.log({str(n_games) + " " + str(eval_or_test_name) + " games, avg rewards": np.mean(eval_scores)})
            plot_scores_testing(eval_scores, n_games,
                                os.path.join(self.plots_path, name) + '_eval.png')  # 'plots/' + name + '_eval.png')


# TODO: Code below can be deleted
class WandbTrainer:
    def __init__(self, config_defaults, sweeps_project_name, env, test_name, training=False, testing=True,
                 games_to_avg=50):
        # self.config_defaults = config_defaults
        self.env = env
        self.test_name = test_name

        self.training = training
        self.testing = testing

        # self.sweep_id = wandb.sweep(sweep_config, project=sweeps_project_name + '-' + test_name) #project="simpledrawer_test-"

        if not os.path.exists('tests'):
            os.mkdir('tests')
        tests_sweepsproj_name = os.path.join('tests', sweeps_project_name)
        if not os.path.exists(tests_sweepsproj_name):
            os.mkdir(tests_sweepsproj_name)
        self.models_path = os.path.join(tests_sweepsproj_name, 'models')
        if not os.path.exists(self.models_path):
            os.mkdir(self.models_path)
        if not os.path.exists(os.path.join(self.models_path, test_name)):
            os.mkdir(os.path.join(self.models_path, test_name))

        self.plots_path = os.path.join(tests_sweepsproj_name, 'plots')
        if not os.path.exists(self.plots_path):
            os.mkdir(self.plots_path)
        if not os.path.exists(os.path.join(self.plots_path, test_name)):
            os.mkdir(os.path.join(self.plots_path, test_name))

    def train(self, config):
        """NON HYPERPARAMETERS"""
        # training = False
        # testing = True
        # TODO: move them out of here
        checkpoint_dir = self.models_path  # 'models'
        n_train_games_to_avg = 50
        n_eval_games = 10
        eval_games_freq = 200
        n_test_games = 1
        n_test_games_to_avg = 1

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

        name = self.test_name + '/lr' + str(lr) + '_gamma' + str(gamma) + '_epsilon' + str(
            epsilon) + '_batch_size' + str(
            batch_size) + '_fc_size' + str(n_hidden)

        # agent = Agent(n_states, n_actions, n_hidden, lr, gamma, epsilon, epsilon_min, epsilon_dec, replace, mem_size,
        #               batch_size, name, checkpoint_dir)
        # agent = DuelingDDQNAgent(n_states, n_actions, n_hidden, lr, gamma, epsilon, epsilon_min, epsilon_dec, replace,
        #                          mem_size, batch_size, name, checkpoint_dir)

        agent = AgentDoubleOut(n_states, n_actions, n_hidden, lr, gamma, epsilon, epsilon_min, epsilon_dec, replace,
                               mem_size,
                               batch_size, name, checkpoint_dir)
        print('s2')
        train(name, self.env, agent, self.plots_path, config.max_steps, n_train_games_to_avg, eval_games_freq,
              n_eval_games, using_wandb=True)
        print('s3')

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
                print('testssss')
                done = False
                test_score = 0
                game_result = 'lost'
                state = self.env.reset()
                starting_state = self.env.starting_pos
                starts_per_states[starting_state] += 1
                while not done:
                    self.env.render()
                    # print(agent.epsilon)
                    #if test_game_idx % 50 == 0:
                    #    self.env.print_debug()
                    shape_n, source, canvas, pointer = state
                    # source, canvas, pointer = state
                    state = np.append(source.reshape(-1), canvas.reshape(-1))
                    state = np.append(state, pointer)
                    state = np.array(state, dtype=np.float32)  # prevent automatic casting to float64 (don't know why that happened though...)
                    # action = agent.choose_action(state)
                    action, pen_state = agent.choose_action(state)

                    # action = random.randint(0,4)
                    state_next, reward, done, is_win = self.env.step_simultaneous(action, pen_state)
                    print(reward)
                    shape_n_next, source_next, canvas_next, pointer_next = state_next
                    # source_next, canvas_next, pointer_next = state_next
                    state = state_next

                    test_score += reward
                    test_score = round(test_score, 2)
                test_scores.append(test_score)
                if np.array_equal(source_next, canvas_next):
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

            wandb.log({str(n_test_games) + " test games, avg score": np.mean(test_scores[n_test_games_to_avg-1:])})
            wandb.log({str(n_test_games) + " test games, win pct": test_wins / n_test_games * 100})

            plot_scores_testing(test_scores, n_test_games_to_avg, os.path.join(self.plots_path, name) + '_test.png')  # 'plots/' + name + '_test.png')

            print('Starts per states')
            print(starts_per_states)
            print('Wins per states')
            print(wins_per_states)
            print('#############')
            print('Losses per states')
            print(losses_per_states)

    def test_working(self, agent, name, n_test_games, n_test_games_to_avg):
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
                print('testssss')
                done = False
                test_score = 0
                game_result = 'lost'
                state = self.env.reset()
                starting_state = self.env.starting_pos
                starts_per_states[starting_state] += 1
                while not done:
                    self.env.render()
                    # print(agent.epsilon)
                    #if test_game_idx % 50 == 0:
                    #    self.env.print_debug()
                    shape_n, source, canvas, pointer = state
                    # source, canvas, pointer = state
                    state = np.append(source.reshape(-1), canvas.reshape(-1))
                    state = np.append(state, pointer)
                    state = np.array(state, dtype=np.float32)  # prevent automatic casting to float64 (don't know why that happened though...)
                    # action = agent.choose_action(state)
                    action, act_scores = agent.choose_action_debug(state)
                    act_scores = act_scores[1] # need to take advantages if working with DuelingDDQN
                    show_q_values(source.shape[0],pointer[0], pointer[1], act_scores.detach().cpu().numpy()[0])
                    # action = random.randint(0,4)
                    state_next, reward, done, is_win = self.env.step(action)
                    print(action, act_scores, reward)
                    shape_n_next, source_next, canvas_next, pointer_next = state_next
                    # source_next, canvas_next, pointer_next = state_next
                    state = state_next

                    test_score += reward
                    test_score = round(test_score, 2)
                test_scores.append(test_score)
                if np.array_equal(source_next, canvas_next):
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

            wandb.log({str(n_test_games) + " test games, avg score": np.mean(test_scores[n_test_games_to_avg-1:])})
            wandb.log({str(n_test_games) + " test games, win pct": test_wins / n_test_games * 100})

            plot_scores_testing(test_scores, n_test_games_to_avg, os.path.join(self.plots_path, name) + '_test.png')  # 'plots/' + name + '_test.png')

            print('Starts per states')
            print(starts_per_states)
            print('Wins per states')
            print(wins_per_states)
            print('#############')
            print('Losses per states')
            print(losses_per_states)

