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


if __name__ == '__main__':

    filename = os.path.join(root_path, 'real_experiments', 'uncertainty', '0', '0010', 'data.csv')

    N_INIT_POINTS = 89
    test_range = range(0, 101, 10)

    X, y = read_data(filename)

    all_X = X
    all_y = y

    scores = []
    for t in test_range:
        ind = N_INIT_POINTS - 1 + t

        X_train = X[0:ind]
        y_train = y[0:ind]

        clf = train_classifier(X_train, y_train)
        prediction_accuracy = clf.score(all_X, all_y)
        scores.append(prediction_accuracy)


    fig = plt.figure(figsize=(12, 8))
    plt.plot(test_range, scores)
    plt.xlim([0, 100])
    plt.ylim([0.5, 1])
    plt.xlabel('Number of experiments', fontsize=fontsize)
    plt.ylabel('Prediction accuracy', fontsize=fontsize)

    plot_filename = os.path.join(HERE_PATH, 'plot', 'uncertainty')
    save_and_close_figure(fig, plot_filename, exts=['.png'])
