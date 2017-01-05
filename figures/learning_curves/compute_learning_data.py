import os

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# adding parent directory to path, so we can access the utils easily
import sys
root_path = os.path.join(HERE_PATH, '..', '..')
sys.path.append(root_path)

import pickle
import random
import numpy as np
import multiprocessing

from sklearn import grid_search
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from utils.csv_helpers import read_data
from utils.plot_helpers import save_and_close_figure

from analysis.tools import N_INIT_POINTS

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


def compute_learning_curve(filename, (X_test, y_test), blank_clf, test_range=range(0, 101, 10)):

    X, y = read_data(filename)

    scores = []
    confusions = []
    for t in test_range:
        ind = N_INIT_POINTS - 1 + t

        X_train = X[0:ind]
        y_train = y[0:ind]

        clf = clone(blank_clf)
        clf.fit(X_train, y_train)

        prediction_accuracy = clf.score(X_test, y_test)
        scores.append(prediction_accuracy)

        y_pred = clf.predict(X_test)
        confusions.append(confusion_matrix(y_test, y_pred))

    return test_range, scores, confusions


def confusion_to_class_accuracy(confusion_matrix):
    class_accuracy = []
    for iclass in range(confusion_matrix.shape[0]):
        pred = confusion_matrix[iclass, iclass] / float(np.sum(confusion_matrix[iclass, :]))
        class_accuracy.append(pred)
    return class_accuracy

def class_accuracy_through_time(confusions):
    class_accuracies = []
    for iclass in range(confusions[0].shape[0]):
        class_accuracies.append([])

    for confusion in confusions:
        class_accuracy = confusion_to_class_accuracy(confusion)
        for iclass, acc in enumerate(class_accuracy):
            class_accuracies[iclass].append(acc)

    return class_accuracies


CLF_METHODS = {}

from sklearn.svm import SVC
CLF_METHODS['SVM'] = {}
CLF_METHODS['SVM']['blank_clf'] = SVC()
CLF_METHODS['SVM']['param_grid'] = {
    'C': np.logspace(-3, 3, 13),
    'gamma': np.logspace(-3, 3, 13),
    'kernel': ['rbf'],
    'probability': [True]
}


def save_pickle(filename, data):
    with open(filename, 'w') as f:
        pickle.dump(data, f)


if __name__ == '__main__':

    # setting a seed for reproducibility
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    #
    from analysis.tools import FILENAMES
    from analysis.tools import get_all_data


    X_test, y_test = get_all_data()

    all_learning_info = {}

    for method_name, method in CLF_METHODS.items():
        print '### {}'.format(method_name)

        all_learning_info[method_name] = {}

        clf = train_classifier_method(X_test, y_test, method)
        method_blank_clf = clone(method['blank_clf'].set_params(**clf.best_params_))
        print 'clf params: {}'.format(clf.best_params_)
        ##

        RESULTS = {}
        for xp_name, filename in FILENAMES.items():
            test_range, scores, confusions = compute_learning_curve(filename, (X_test, y_test), method_blank_clf)
            class_acc = class_accuracy_through_time(confusions)

            RESULTS[xp_name] = {}
            RESULTS[xp_name]['test_range'] = test_range
            RESULTS[xp_name]['scores'] = scores
            RESULTS[xp_name]['confusions'] = confusions
            RESULTS[xp_name]['class_acc'] = class_acc
            RESULTS[xp_name]['unbiased_acc'] = np.mean(class_acc, 0)

            print '{}: {}'.format(xp_name, RESULTS[xp_name]['unbiased_acc'][-1])

        ##
        all_learning_info[method_name]['clf_params'] = clf.best_params_
        all_learning_info[method_name]['results'] = RESULTS

    ##
    info_filename = os.path.join(HERE_PATH, 'learning_data.pkl')
    save_pickle(info_filename, all_learning_info)
