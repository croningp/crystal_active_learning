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
from tools import FILENAMES
from tools import get_data_xp

from sklearn.neighbors import BallTree

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# design figure
fontsize = 22
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
matplotlib.rcParams.update({'font.size': fontsize})


def average_n_neighbors(tree, X, radius):
    n_neighbors = tree.query_radius(X, r=radius, count_only=True)
    n_neighbors = n_neighbors / float(X.shape[0])
    return n_neighbors.mean(), n_neighbors.std()


def map_n_neighbors(X, radius_list):
    tree = BallTree(X)
    means = []
    stds = []
    for radius in radius_list:
        mean, std = average_n_neighbors(tree, X, radius)
        means.append(mean)
        stds.append(std)

    return means, stds


def compute_volume_neighbors(filename, radius, test_range=range(0, 101, 10)):

    X, y = read_data(filename)

    scores = []
    for t in test_range:
        ind = N_INIT_POINTS - 1 + t

        X_train = X[0:ind]
        y_train = y[0:ind]

        points = X_train[y_train == CRYSTAL_CLASS, :]
        tree = BallTree(points)
        score, _ = average_n_neighbors(tree, points, radius)
        scores.append(score)

    return test_range, scores


if __name__ == '__main__':

    import filetools
    PLOT_FOLDER = os.path.join(HERE_PATH, 'plot')
    filetools.ensure_dir(PLOT_FOLDER)

    radius_list = np.logspace(-2, 1, 100)

    all_means = []

    fig = plt.figure(figsize=(12, 8))
    for xp_name, _ in FILENAMES.items():
        X, y = get_data_xp(xp_name)
        points = X[y == CRYSTAL_CLASS, :]
        means, _ = map_n_neighbors(points, radius_list)
        all_means.append(means)
        plt.errorbar(radius_list, means)

    plt.ylim([-0.05, 1.05])
    plt.legend(FILENAMES.keys(), fontsize=fontsize, loc=4)
    plt.xlabel('Radius', fontsize=fontsize)
    plt.ylabel('Average ratio of crystals within \n radius of other crystals', fontsize=fontsize)

    plot_filename = os.path.join(PLOT_FOLDER, 'volume_neighbors_radius')
    save_and_close_figure(fig, plot_filename, exts=['.png'])

    ##
    std_n_coverage = np.std(all_means, axis=0)
    selected_ind = np.argmax(std_n_coverage)
    selected_radius =  radius_list[selected_ind]

    fig = plt.figure(figsize=(12, 8))
    plt.plot(radius_list, std_n_coverage)
    plt.scatter(selected_radius, std_n_coverage[selected_ind], c='r')
    plt.xlabel('Radius', fontsize=fontsize)
    plt.ylabel('Standard deviation of neighbors \n coverage between experiments', fontsize=fontsize)

    plot_filename = os.path.join(PLOT_FOLDER, 'volume_neighbors_calibration')
    save_and_close_figure(fig, plot_filename, exts=['.png'])

    ##
    fig = plt.figure(figsize=(12, 8))
    for xp_name, filename in FILENAMES.items():
        test_range, volumes = compute_volume_neighbors(filename, selected_radius)
        plt.plot(test_range, volumes)

    plt.legend(FILENAMES.keys(), fontsize=fontsize, loc=3)
    plt.xlim([0, 100])
    plt.ylim([0, 1])
    plt.xlabel('Number of experiments', fontsize=fontsize)
    plt.ylabel('Average ratio of crystals found within \n given distance of other crystals', fontsize=fontsize)

    plot_filename = os.path.join(PLOT_FOLDER, 'volume_neighbors')
    save_and_close_figure(fig, plot_filename, exts=['.png'])
