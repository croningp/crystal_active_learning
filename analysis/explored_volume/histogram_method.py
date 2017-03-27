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

from scipy.spatial.distance import pdist

from utils.csv_helpers import read_data
from utils.plot_helpers import save_and_close_figure

from tools import N_INIT_POINTS
from tools import CRYSTAL_CLASS
from tools import FILENAMES
from tools import get_data_xp

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# design figure
fontsize = 22
linewidth = 3
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
matplotlib.rcParams.update({'font.size': fontsize})


def hist_from_X(X, bins):
    dist = pdist(X)
    hist, _ = np.histogram(dist, bins=bins)
    hist = hist / float(len(dist))
    x_plot = np.diff(bins)/2 + bins[:-1]
    return x_plot, hist


if __name__ == '__main__':

    import filetools
    PLOT_FOLDER = os.path.join(HERE_PATH, 'plot')
    filetools.ensure_dir(PLOT_FOLDER)

    radius_list = np.logspace(-2, 1, 30)

    XP_NAMES = ['uncertainty_0', 'uncertainty_1', 'human_0', 'human_1', 'random_0', 'random_1']
    LEGEND_NAMES = ['Algorithm - run 1', 'Algorithm - run 2', 'Human     - run 1', 'Human     - run 2', 'Random   - run 1', 'Random   - run 2']

    fig = plt.figure(figsize=(12, 8))
    for xp_name in XP_NAMES:
        X, y = get_data_xp(xp_name)
        points = X[y == CRYSTAL_CLASS, :]
        x_plot, hist = hist_from_X(points, bins = radius_list)
        plt.errorbar(x_plot, hist, linewidth=linewidth)

    plt.ylim([-0.01, 0.2])
    legend = plt.legend(LEGEND_NAMES, fontsize=fontsize, bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    plt.xlabel('Euclidean distance between experimental points', fontsize=fontsize)
    plt.ylabel('Histogram of crystals within radius of each others', fontsize=fontsize)

    plot_filename = os.path.join(PLOT_FOLDER, 'hist_dist_beween_crystals')
    save_and_close_figure(fig, plot_filename, exts=['.png'], legend=legend)
