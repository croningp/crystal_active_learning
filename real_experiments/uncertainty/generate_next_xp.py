import os

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# adding parent directory to path, so we can access the utils easily
import sys
root_path = os.path.join(HERE_PATH, '..', '..')
sys.path.append(root_path)

import json
import random
import numpy as np

##
from utils.csv_helpers import read_data
from utils.csv_helpers import write_data

from utils.classifier import train_classifier

from utils.uncertainty import soft_max
from utils.uncertainty import compute_weights
from utils.uncertainty import probabilistic_choice
from utils.uncertainty import compute_best_temperature


def proba_normalize(x):
    x = np.array(x, dtype=float)
    if np.sum(x) == 0:
        x = np.ones(x.shape)
    return x / np.sum(x, dtype=float)


def proba_normalize_row(x):
    # not optimized at all
    x = np.array(x, dtype=float)
    for i in range(x.shape[0]):
        x[i, :] = proba_normalize(x[i, :])
    return x


def generate_next_samples(n_samples, clf, n_dim, n_sampling):

    all_run_info = []
    all_X_selected = []

    for i in range(n_samples):

        X_sampled = np.random.rand(n_sampling, n_dim)
        X_sampled = proba_normalize_row(X_sampled)

        X_proba = clf.best_estimator_.predict_proba(X_sampled)

        weights, entropy, repulsion = compute_weights(X_sampled, X_proba, clf, np.atleast_2d(all_X_selected))

        best_temperature = compute_best_temperature(weights)
        selection_proba = soft_max(weights, best_temperature)

        sampled_index = probabilistic_choice(selection_proba)
        X_selected = X_sampled[sampled_index, :]

        #
        run_info = {}
        run_info['X_sampled'] = X_sampled
        run_info['X_proba'] = X_proba
        run_info['weights'] = weights
        run_info['entropy'] = entropy
        run_info['repulsion'] = repulsion
        run_info['best_temperature'] = best_temperature
        run_info['selection_proba'] = selection_proba
        run_info['sampled_index'] = sampled_index
        run_info['X_selected'] = X_selected

        all_run_info.append(run_info)
        all_X_selected.append(X_selected)

    return np.array(all_X_selected), all_run_info


def save_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f)

N_SELECTED = 10
N_SAMPLING = 10000
TOTAL_VOLUME_IN_ML = 15.0
N_DECIMAL_EQUAL = 2

if __name__ == '__main__':

    import sys

    if len(sys.argv) != 2:
        print 'Please specify a root folder as argument'

    root_folder = os.path.join(HERE_PATH, sys.argv[1])

    import filetools

    folders = filetools.list_folders(root_folder)
    folders.sort()
    last_folder = folders[-1]
    last_folder_int = int(os.path.basename(last_folder))

    # seed
    random.seed(last_folder_int)
    np.random.seed(last_folder_int)

    # load data
    current_datafile = os.path.join(last_folder, 'data.csv')
    X, y = read_data(current_datafile)
    # check everything is fine
    np.testing.assert_array_almost_equal(np.sum(X, axis=1), TOTAL_VOLUME_IN_ML, decimal=N_DECIMAL_EQUAL)

    #
    X = proba_normalize_row(X)

    #
    clf = train_classifier(X, y)
    X_selected, all_run_info = generate_next_samples(N_SELECTED, clf, X.shape[1], N_SAMPLING)

    # save new csv
    next_folder_number = filetools.generate_n_digit_name(last_folder_int + 1)
    next_folder = os.path.join(root_folder, next_folder_number)

    if os.path.exists(next_folder):
        user_ok = False
        while not user_ok:
            rep = raw_input('{} already exist, you might erase stuff, continue [y, N]')
            if rep in ['y', 'Y']:
                user_ok = True

    filetools.ensure_dir(next_folder)

    # xout
    X_out = np.vstack((X, X_selected))
    X_out = TOTAL_VOLUME_IN_ML * X_out

    # check everything is fine
    np.testing.assert_array_almost_equal(np.sum(X_out, axis=1), TOTAL_VOLUME_IN_ML, decimal=N_DECIMAL_EQUAL)

    # save
    next_datafile = os.path.join(next_folder, 'data.csv')
    write_data(next_datafile, X_out, y)
