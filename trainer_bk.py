import os
import torch
import wandb
import numpy as np

from utils_plot import plot_scores, plot_scores_testing


def __create_paths(sweeps_project_name, test_name):
    if not os.path.exists('tests'):
        os.mkdir('tests')
    tests_sweepsproj_name = os.path.join('tests', sweeps_project_name)
    if not os.path.exists(tests_sweepsproj_name):
        os.mkdir(tests_sweepsproj_name)
    models_path = os.path.join(tests_sweepsproj_name, 'models')
    if not os.path.exists(models_path):
        os.mkdir(models_path)
    if not os.path.exists(os.path.join(models_path, test_name)):
        os.mkdir(os.path.join(models_path, test_name))

    plots_path = os.path.join(tests_sweepsproj_name, 'plots')
    if not os.path.exists(plots_path):
        os.mkdir(plots_path)
    if not os.path.exists(os.path.join(plots_path, test_name)):
        os.mkdir(os.path.join(plots_path, test_name))
    return models_path, plots_path

def train(name, env, agent, plots_path, max_steps, n_train_games_to_avg, eval_games_freq, n_eval_games, using_wandb=False):
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
            state = np.append(source.reshape(-1), canvas.reshape(-1))
            state = np.append(state, pointer)
            state = np.array(state,
                             dtype=np.float32)  # prevent automatic casting to float64 (don't know why that happened though...)
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

            flat_state_next = np.append(source_next.reshape(-1), canvas_next.reshape(-1))
            flat_state_next = np.append(flat_state_next, pointer_next)

            # TODO: Try not casting done to int
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
            print('############################\ntraining recap after', n_steps, 'steps and', game_n,
                  'games.\n', '50 games avg SCORE:', np.mean(scores[-n_train_games_to_avg:]),
                  'eps:', agent.epsilon, '50 games win pct', wins / n_train_games_to_avg,
                  '\n##################\n')
            plot_scores(scores, epsilon_history, n_train_games_to_avg,
                        os.path.join(plots_path, name) + '.png')  # 'plots/' + name + '.png')
            if using_wandb:
                wandb.log({"50 games avg reward": np.mean(scores[-n_train_games_to_avg:])})
                wandb.log({"50 games n wins": wins / n_train_games_to_avg * 100})
                wandb.log({"epsilon": agent.epsilon})
            wins = 0
        '''########### EVALUATION ###########'''
        # TODO: un def test diverso da quello di sotto gia fatto
        # TODO: Creare choose action per testing
        if game_n % eval_games_freq == 0:
            with torch.no_grad():
                is_win = False
                agent.is_training(False)
                best_eval_score = -100
                # agent.eval_Q.eval()
                eval_wins = 0
                eval_scores = []
                for test_game_idx in range(n_eval_games):
                    done = False
                    eval_score = 0
                    state = env.reset()
                    while not done:
                        # print(agent.epsilon)
                        # if test_game_idx % 10 == 0:
                        #    env.print_debug()
                        shape_n, source, canvas, pointer = state
                        # source, canvas, pointer = state
                        state = np.append(source.reshape(-1), canvas.reshape(-1))
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
                if eval_wins >= eval_best_win_n and agent.epsilon == 0:
                    eval_best_win_n = eval_wins
                    # TODO: What do we prefer? An agent that achieves higher reward but does not draw 100% correct, or an agent that draws well but takes more time? Reward functions, however, could change.
                    agent.save_models()

                print('############################\nevaluation after', n_steps, 'iterations.\n', n_eval_games,
                      'games avg SCORE:', np.mean(eval_scores),
                      'win pct (%)', (eval_wins / n_eval_games) * 100, '\n##################\n')
                if using_wandb:
                    wandb.log({str(n_eval_games) + " eval games, win pct (%)": (eval_wins / n_eval_games) * 100})
                    wandb.log({str(n_eval_games) + " eval games, avg rewards": np.mean(eval_scores)})
                plot_scores_testing(eval_scores, n_eval_games,
                                    os.path.join(plots_path, name) + '_eval.png')  # 'plots/' + name + '_eval.png')


def train_working(name, env, agent, plots_path, max_steps, n_train_games_to_avg, eval_games_freq, n_eval_games, using_wandb=False):
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
            state = np.append(source.reshape(-1), canvas.reshape(-1))
            state = np.append(state, pointer)
            state = np.array(state,
                             dtype=np.float32)  # prevent automatic casting to float64 (don't know why that happened though...)
            action = agent.choose_action(state)
            # action = random.randint(0,4)
            state_next, reward, done, is_win = env.step(action)
            shape_n_next, source_next, canvas_next, pointer_next = state_next
            # source_next, canvas_next, pointer_next = state_next
            # if done:
            # if np.array_equal(source_next, canvas_next):
            # if reward == 100:
            #    print('win')
            #    wins += 1

            flat_state_next = np.append(source_next.reshape(-1), canvas_next.reshape(-1))
            flat_state_next = np.append(flat_state_next, pointer_next)

            # TODO: Try not casting done to int
            agent.store_transition(state, action, reward, flat_state_next, int(done))
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
            print('############################\ntraining recap after', n_steps, 'steps and', game_n,
                  'games.\n', '50 games avg SCORE:', np.mean(scores[-n_train_games_to_avg:]),
                  'eps:', agent.epsilon, '50 games win pct', wins / n_train_games_to_avg,
                  '\n##################\n')
            plot_scores(scores, epsilon_history, n_train_games_to_avg,
                        os.path.join(plots_path, name) + '.png')  # 'plots/' + name + '.png')
            if using_wandb:
                wandb.log({"50 games avg reward": np.mean(scores[-n_train_games_to_avg:])})
                wandb.log({"50 games n wins": wins / n_train_games_to_avg * 100})
                wandb.log({"epsilon": agent.epsilon})
            wins = 0
        '''########### EVALUATION ###########'''
        # TODO: un def test diverso da quello di sotto gia fatto
        # TODO: Creare choose action per testing
        if game_n % eval_games_freq == 0:
            with torch.no_grad():
                is_win = False
                agent.is_training(False)
                best_eval_score = -100
                # agent.eval_Q.eval()
                eval_wins = 0
                eval_scores = []
                for test_game_idx in range(n_eval_games):
                    done = False
                    eval_score = 0
                    state = env.reset()
                    while not done:
                        # print(agent.epsilon)
                        # if test_game_idx % 10 == 0:
                        #    env.print_debug()
                        shape_n, source, canvas, pointer = state
                        # source, canvas, pointer = state
                        state = np.append(source.reshape(-1), canvas.reshape(-1))
                        state = np.append(state, pointer)
                        state = np.array(state,
                                         dtype=np.float32)  # prevent automatic casting to float64 (don't know why that happened though...)

                        action = agent.choose_action(state)
                        # action = random.randint(0,4)
                        state_next, reward, done, is_win = env.step(action)
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
                if eval_wins >= eval_best_win_n and agent.epsilon == 0:
                    eval_best_win_n = eval_wins
                    # TODO: What do we prefer? An agent that achieves higher reward but does not draw 100% correct, or an agent that draws well but takes more time? Reward functions, however, could change.
                    agent.save_models()

                print('############################\nevaluation after', n_steps, 'iterations.\n', n_eval_games,
                      'games avg SCORE:', np.mean(eval_scores),
                      'win pct (%)', (eval_wins / n_eval_games) * 100, '\n##################\n')
                if using_wandb:
                    wandb.log({str(n_eval_games) + " eval games, win pct (%)": (eval_wins / n_eval_games) * 100})
                    wandb.log({str(n_eval_games) + " eval games, avg rewards": np.mean(eval_scores)})
                plot_scores_testing(eval_scores, n_eval_games,
                                    os.path.join(plots_path, name) + '_eval.png')  # 'plots/' + name + '_eval.png')

