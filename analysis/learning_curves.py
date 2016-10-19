import os

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# adding parent directory to path, so we can access the utils easily
import sys
root_path = os.path.join(HERE_PATH, '..')
sys.path.append(root_path)

import json
import random
import numpy as np

from utils.csv_helpers import read_data
from utils.classifier import train_classifier


import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from utils.plot_helpers import save_and_close_figure


# design figure
fontsize = 22
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
matplotlib.rcParams.update({'font.size': fontsize})

N_INIT_POINTS = 89

def compute_learning_curve(filename, (X_test, y_test), test_range=range(0, 101, 10)):

    X, y = read_data(filename)

    scores = []
    for t in test_range:
        ind = N_INIT_POINTS - 1 + t

        X_train = X[0:ind]
        y_train = y[0:ind]

        clf = train_classifier(X_train, y_train)
        prediction_accuracy = clf.score(X_test, y_test)
        scores.append(prediction_accuracy)

    return test_range, scores


def get_init_data(filename):
    X, y = read_data(filename)
    return X[:N_INIT_POINTS, :], y[:N_INIT_POINTS]


def get_new_data(filename):
    X, y = read_data(filename)
    return X[N_INIT_POINTS:, :], y[N_INIT_POINTS:]


if __name__ == '__main__':

    random_filename = os.path.join(root_path, 'real_experiments', 'random', '0', 'data.csv')

    uncertainty_filename = os.path.join(root_path, 'real_experiments', 'uncertainty', '0', '0010', 'data.csv')

    human_filename = os.path.join(root_path, 'real_experiments', 'human', '0', 'data.csv')


    X_test, y_test = read_data(random_filename)
    X, y = get_new_data(uncertainty_filename)
    X_test = np.vstack((X_test, X))
    y_test = np.hstack((y_test, y))
    X, y = get_new_data(human_filename)
    X_test = np.vstack((X_test, X))
    y_test = np.hstack((y_test, y))

    test_range, r_scores = compute_learning_curve(random_filename, (X_test, y_test))
    test_range, u_scores = compute_learning_curve(uncertainty_filename, (X_test, y_test))
    test_range, h_scores = compute_learning_curve(human_filename, (X_test, y_test))


    fig = plt.figure(figsize=(12, 8))
    plt.plot(test_range, r_scores)
    plt.plot(test_range, u_scores)
    plt.plot(test_range, h_scores)
    plt.legend(['Random', 'Uncertainty', 'Human'], fontsize=fontsize, loc=4)
    plt.xlim([0, 100])
    plt.ylim([0.5, 1])
    plt.xlabel('Number of experiments', fontsize=fontsize)
    plt.ylabel('Prediction accuracy', fontsize=fontsize)

    plot_filename = os.path.join(HERE_PATH, 'plot', 'learning_curve')
    save_and_close_figure(fig, plot_filename, exts=['.png'])
