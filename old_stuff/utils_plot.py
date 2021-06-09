import numpy as np
import matplotlib.pyplot as plt

def plot_scores(scores, epsilons, n_episodes_to_consider, figure_file):
    avg_scores = []
    x = []
    for t in range(len(scores)):
        if t < n_episodes_to_consider:
            avg_scores.append(np.mean(scores[0: t + 1]))
        else:
            avg_scores.append(np.mean(scores[t - n_episodes_to_consider: t]))
        x.append(t)

    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis="x", colors="C0")
    ax.tick_params(axis="y", colors="C0")

    # plt.plot(x, avg_scores)
    # plt.title('Average of the previous %d scores' %(n_episodes_to_consider))
    # plt.yticks(np.arange(round(min(scores)), round(max(scores)), 5))
    # plt.savefig(figure_file)

    ax2.scatter(x, avg_scores, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    plt.savefig(figure_file)
    plt.close()


def plot_scores_testing(scores, n_episodes_to_consider, figure_file):
    avg_scores = []
    x = []
    for t in range(len(scores)):
        if t < n_episodes_to_consider:
            avg_scores.append(np.mean(scores[0: t + 1]))
        else:
            avg_scores.append(np.mean(scores[t - n_episodes_to_consider: t]))
        x.append(t)
    plt.plot(x, avg_scores)
    plt.title('Average of the previous %d scores' %(n_episodes_to_consider))
    plt.savefig(figure_file)