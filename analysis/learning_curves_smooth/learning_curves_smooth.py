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

from tools import N_INIT_POINTS

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# design figure
fontsize = 22
linewidth = 3
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

from sklearn.ensemble import RandomForestClassifier
CLF_METHODS['RandomForest'] = {}
CLF_METHODS['RandomForest']['blank_clf'] = RandomForestClassifier()
CLF_METHODS['RandomForest']['param_grid'] = {
    'n_estimators': [50, 100, 200, 500]
}

from sklearn.ensemble import AdaBoostClassifier
CLF_METHODS['Adaboost'] = {}
CLF_METHODS['Adaboost']['blank_clf'] = AdaBoostClassifier()
CLF_METHODS['Adaboost']['param_grid'] = {
    'n_estimators': [50, 100, 200, 500]
}

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
    import filetools
    PLOT_FOLDER = os.path.join(HERE_PATH, 'plot')
    filetools.ensure_dir(PLOT_FOLDER)

    from tools import FILENAMES
    from tools import get_all_data

    XP_NAMES = ['uncertainty_0', 'uncertainty_1', 'human_0', 'human_1', 'random_0', 'random_1']
    LEGEND_NAMES = ['Algorithm - run 1', 'Algorithm - run 2', 'Human     - run 1', 'Human     - run 2', 'Random   - run 1', 'Random   - run 2']

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
        fig = plt.figure(figsize=(12, 8))
        for xp_name in XP_NAMES:
            result_dict = RESULTS[xp_name]
            plt.plot(result_dict['test_range'], result_dict['scores'], linewidth=linewidth)
        plt.title(method_name, fontsize=fontsize)
        legend = plt.legend(LEGEND_NAMES, fontsize=fontsize, bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
        plt.xlim([0, 100])
        plt.ylim([0.5, 1])
        plt.xlabel('Number of experiments', fontsize=fontsize)
        plt.ylabel('Prediction accuracy', fontsize=fontsize)
        plt.tight_layout()

        plot_filename = os.path.join(PLOT_FOLDER, 'learning_curve_{}'.format(method_name))
        save_and_close_figure(fig, plot_filename, exts=['.png'], legend=legend)

        ##
        class_names = ['NoCrystal', 'Crystal']
        for iclass, class_name in enumerate(class_names):

            fig = plt.figure(figsize=(12, 8))
            for xp_name in XP_NAMES:
                result_dict = RESULTS[xp_name]
                plt.plot(result_dict['test_range'], result_dict['class_acc'][iclass], linewidth=linewidth)
            plt.title('{} | {}'.format(class_name, method_name), fontsize=fontsize)
            legend = plt.legend(LEGEND_NAMES, fontsize=fontsize, bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
            plt.xlim([0, 100])
            plt.ylim([0.5, 1])
            plt.xlabel('Number of experiments', fontsize=fontsize)
            plt.ylabel('{} prediction accuracy'.format(class_name), fontsize=fontsize)

            plot_filename = os.path.join(PLOT_FOLDER, 'learning_curve_{}_{}'.format(method_name, class_name))
            save_and_close_figure(fig, plot_filename, exts=['.png'], legend=legend)

        ##
        fig = plt.figure(figsize=(12, 8))
        for xp_name in XP_NAMES:
            result_dict = RESULTS[xp_name]
            plt.plot(result_dict['test_range'], result_dict['unbiased_acc'], linewidth=linewidth)
        plt.title(method_name, fontsize=fontsize)
        legend = plt.legend(LEGEND_NAMES, fontsize=fontsize, bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
        plt.xlim([0, 100])
        plt.ylim([0.6, 0.9])
        plt.xlabel('Number of experiments', fontsize=fontsize)
        plt.ylabel('Prediction accuracy unbiased', fontsize=fontsize)

        plot_filename = os.path.join(PLOT_FOLDER, 'learning_curve_{}_unbiased'.format(method_name))
        save_and_close_figure(fig, plot_filename, exts=['.png'], legend=legend)

    ##
    info_filename = os.path.join(PLOT_FOLDER, 'info.pkl')
    save_pickle(info_filename, all_learning_info)
