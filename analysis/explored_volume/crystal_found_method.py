import os

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# adding parent directory to path, so we can access the utils easily
import sys
root_path = os.path.join(HERE_PATH, '..', '..')
sys.path.append(root_path)

analysis_path = os.path.join(HERE_PATH, '..')
sys.path.append(analysis_path)

import json
import random
import numpy as np

from utils.csv_helpers import read_data
from utils.plot_helpers import save_and_close_figure

from tools import N_INIT_POINTS
from tools import CRYSTAL_CLASS

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# design figure
fontsize = 22
linewidth = 3
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
matplotlib.rcParams.update({'font.size': fontsize})


def compute_n_crystal_found(filename, test_range=range(0, 101, 10)):

    X, y = read_data(filename)

    scores = []
    for t in test_range:
        ind = N_INIT_POINTS - 1 + t
        scores.append(np.sum(y[N_INIT_POINTS:ind] == CRYSTAL_CLASS))

    return test_range, scores


if __name__ == '__main__':

    import filetools
    PLOT_FOLDER = os.path.join(HERE_PATH, 'plot')
    filetools.ensure_dir(PLOT_FOLDER)

    from tools import FILENAMES

    XP_NAMES = ['uncertainty_0', 'uncertainty_1', 'human_0', 'human_1', 'random_0', 'random_1']
    LEGEND_NAMES = ['Algorithm - run 1', 'Algorithm - run 2', 'Human     - run 1', 'Human     - run 2', 'Random   - run 1', 'Random   - run 2']

    fig = plt.figure(figsize=(12, 8))
    for xp_name in XP_NAMES:
        filename = FILENAMES[xp_name]
        test_range, n_crystal = compute_n_crystal_found(filename)
        plt.plot(test_range, n_crystal, linewidth=linewidth)

    legend = plt.legend(LEGEND_NAMES, fontsize=fontsize, bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    plt.xlim([0, 100])
    plt.ylim([-1, 50])
    plt.xlabel('Number of experiments', fontsize=fontsize)
    plt.ylabel('Number of experiments with crystals', fontsize=fontsize)

    plot_filename = os.path.join(PLOT_FOLDER, 'n_crystal_found')
    save_and_close_figure(fig, plot_filename, exts=['.png'], legend=legend)
