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
import multiprocessing

from sklearn import grid_search
from sklearn.metrics import accuracy_score

from utils.csv_helpers import read_data
from utils.plot_helpers import save_and_close_figure

from tools import N_INIT_POINTS
from tools import get_init_data
from tools import get_new_data

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# design figure
fontsize = 22
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
matplotlib.rcParams.update({'font.size': fontsize})


def train_classifier_method(X, y, method):
    blank_clf = method['blank_clf']
    param_grid = method['param_grid']

    n_cv = min(get_min_sample_per_class(y), 10)

    clf = grid_search.GridSearchCV(blank_clf, param_grid, n_jobs=multiprocessing.cpu_count(), cv=n_cv)
    clf.fit(X, y)
    return clf


def get_min_sample_per_class(y):
    min_sample_per_class = np.inf
    for class_number in np.unique(y):
        n_sample = np.sum(y == class_number)
        if n_sample < min_sample_per_class:
            min_sample_per_class = n_sample
    return min_sample_per_class


def proba_normalize(x):
    x = np.array(x, dtype=float)
    div = np.sum(x, 1)
    return x / div[:, None]


N_SAMPLING_MONTECARLO = 100000

def compute_learning_curve(filename, method, test_range=range(0, 101, 10)):

    X, y = read_data(filename)

    scores = []
    for t in test_range:
        ind = N_INIT_POINTS - 1 + t

        X_train = X[0:ind]
        y_train = y[0:ind]

        clf = train_classifier_method(X_train, y_train, method)

        X_test = np.random.rand(N_SAMPLING_MONTECARLO, 4)
        X_test = proba_normalize(X_test) * 15
        y_pred = clf.predict(X_test)

        scores.append(np.sum(y_pred)/float(N_SAMPLING_MONTECARLO))

    return test_range, scores


CLF_METHODS = {}


from sklearn.neighbors import KNeighborsClassifier
CLF_METHODS['KNN'] = {}
CLF_METHODS['KNN']['blank_clf'] = KNeighborsClassifier()
CLF_METHODS['KNN']['param_grid'] = {
    'n_neighbors': [1, 2, 3, 5, 10, 20],
    'weights': ['uniform', 'distance']
}

from sklearn.ensemble import RandomForestClassifier
CLF_METHODS['RandomForest'] = {}
CLF_METHODS['RandomForest']['blank_clf'] = RandomForestClassifier()
CLF_METHODS['RandomForest']['param_grid'] = {
    'n_estimators': [10, 20, 30, 50, 100, 200, 500]
}

from sklearn.svm import SVC
CLF_METHODS['SVM'] = {}
CLF_METHODS['SVM']['blank_clf'] = SVC()
CLF_METHODS['SVM']['param_grid'] = {
    'C': np.logspace(-5, 5, 21),
    'gamma': np.logspace(-5, 5, 21),
    'kernel': ['rbf'],
    'probability': [True]
}


if __name__ == '__main__':

    random_filename = os.path.join(root_path, 'real_experiments', 'random', '0', 'data.csv')

    uncertainty_filename = os.path.join(root_path, 'real_experiments', 'uncertainty', '0', '0010', 'data.csv')

    human_filename = os.path.join(root_path, 'real_experiments', 'human', '0', 'data.csv')


    for method_name, method in CLF_METHODS.items():

        test_range, r_scores = compute_learning_curve(random_filename, method)
        test_range, u_scores = compute_learning_curve(uncertainty_filename, method)
        test_range, h_scores = compute_learning_curve(human_filename, method)


        fig = plt.figure(figsize=(12, 8))
        plt.plot(test_range, r_scores)
        plt.plot(test_range, u_scores)
        plt.plot(test_range, h_scores)
        plt.title(method_name, fontsize=fontsize)
        plt.legend(['Random', 'Uncertainty', 'Human'], fontsize=fontsize, loc=1)
        plt.xlim([0, 100])
        # plt.ylim([0.5, 1])
        plt.xlabel('Number of experiments', fontsize=fontsize)
        plt.ylabel('Prediction accuracy', fontsize=fontsize)

        plot_filename = os.path.join(HERE_PATH, 'plot', 'volume_montecarlo_{}'.format(method_name))
        save_and_close_figure(fig, plot_filename, exts=['.png'])
