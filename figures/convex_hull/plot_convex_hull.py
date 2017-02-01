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

from analysis.tools import N_INIT_POINTS
from analysis.tools import CRYSTAL_CLASS

from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay


import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# design figure
fontsize = 26
matplotlib.rc('xtick', labelsize=22)
matplotlib.rc('ytick', labelsize=22)
matplotlib.rcParams.update({'font.size': fontsize})
linewidth = 3
markersize = 10

def compute_volume_convex_hull(filename, test_range=range(0, 101, 10)):

    X, y = read_data(filename)

    scores = []
    for t in test_range:
        ind = N_INIT_POINTS - 1 + t

        X_train = X[0:ind]
        y_train = y[0:ind]

        points = X_train[y_train == CRYSTAL_CLASS, :]
        hull = ConvexHull(points)
        scores.append(hull.volume)

    return test_range, scores


if __name__ == '__main__':

    from analysis.tools import FILENAMES

    RESULTS = {}
    for xp_name, filename in FILENAMES.items():
        test_range, volumes = compute_volume_convex_hull(filename)
        RESULTS[xp_name] = volumes

    ##
    TEST_RANGE = test_range

    AVG_RESULTS = {}
    AVG_RESULTS['HUMAN'] = np.mean(np.vstack((RESULTS['human_0'], RESULTS['human_1'])), axis=0)
    AVG_RESULTS['ALGORITHM'] = np.mean(np.vstack((RESULTS['uncertainty_0'], RESULTS['uncertainty_1'])), axis=0)
    AVG_RESULTS['RANDOM'] = np.mean(np.vstack((RESULTS['random_0'], RESULTS['random_1'])), axis=0)

    ##
    fig = plt.figure(figsize=(12, 8))

    plt.plot(TEST_RANGE, AVG_RESULTS['ALGORITHM'], 'r', linestyle='-', linewidth=linewidth, marker='o', markersize=markersize)
    plt.plot(TEST_RANGE, AVG_RESULTS['HUMAN'], 'g', linestyle='-', linewidth=linewidth, marker='s', markersize=markersize)
    plt.plot(TEST_RANGE, AVG_RESULTS['RANDOM'], 'b', linestyle='-', linewidth=linewidth, marker='D', markersize=markersize)

    plt.title('Exploration of Crystalization Space', fontsize=fontsize)
    plt.legend(['Algorithm', 'Human', 'Random'], fontsize=fontsize, loc=2)
    plt.xlim([-1, 101])

    plt.xlabel('Number of Experiments', fontsize=fontsize)
    plt.ylabel('Explored Crystalization Space - AU', fontsize=fontsize)

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    #
    import filetools
    PLOT_FOLDER = os.path.join(HERE_PATH, 'plot')
    filetools.ensure_dir(PLOT_FOLDER)

    plot_filename = os.path.join(PLOT_FOLDER, 'volume_convex_hull')
    save_and_close_figure(fig, plot_filename)
