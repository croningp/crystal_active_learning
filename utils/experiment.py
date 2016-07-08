import json

import numpy as np

from classifier import train_classifier
from classifier import count_support_vectors
from classifier import classification_accuracy
from uncertainty import generate_next_samples


def run_full_xp(n_iteration, n_selected_per_iteration, n_sampling, class_func, init_X):

    X = init_X

    all_info = []
    for _ in range(n_iteration):
        y = class_func(X)
        clf = train_classifier(X, y)
        all_X_selected, all_run_info = generate_next_samples(n_selected_per_iteration, clf, X.shape[1], n_sampling)
        #
        info = {}
        info['X'] = X
        info['y'] = y
        info['clf'] = clf
        info['all_X_selected'] = all_X_selected
        info['all_run_info'] = all_run_info
        all_info.append(info)
        #
        X = np.vstack((X, all_X_selected))

    return all_info


def run_random_xp(n_iteration, n_selected_per_iteration, n_sampling, class_func, init_X):

    X = init_X

    all_info = []
    for _ in range(n_iteration):
        y = class_func(X)
        clf = train_classifier(X, y)
        all_X_selected = np.random.rand(n_selected_per_iteration, X.shape[1])
        #
        info = {}
        info['X'] = X
        info['y'] = y
        info['clf'] = clf
        info['all_X_selected'] = all_X_selected
        all_info.append(info)
        #
        X = np.vstack((X, all_X_selected))

    return all_info


def evaluate_xp(all_info, X, class_func):

    n_support_vector = []
    accuracy_score = []

    for info in all_info:
        clf = info['clf']
        n_support_vector.append(count_support_vectors(clf))
        accuracy_score.append(classification_accuracy(X, class_func, clf))

    xp_eval = {}
    xp_eval['n_support_vector'] = n_support_vector
    xp_eval['accuracy_score'] = accuracy_score

    return xp_eval


def save_eval(filename, xp_eval):
    with open(filename, 'w') as f:
        json.dump(xp_eval, f)
