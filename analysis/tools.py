import os

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# adding parent directory to path, so we can access the utils easily
import sys
root_path = os.path.join(HERE_PATH, '..')
sys.path.append(root_path)

import numpy as np

from utils.csv_helpers import read_data


N_INIT_POINTS = 89
CRYSTAL_CLASS = 1
NO_CRYSTAL_CLASS = 0

def get_init_data(filename):
    X, y = read_data(filename)
    return X[:N_INIT_POINTS, :], y[:N_INIT_POINTS]


def get_new_data(filename):
    X, y = read_data(filename)
    return X[N_INIT_POINTS:, :], y[N_INIT_POINTS:]

##
FILENAMES = {}
FILENAMES['random_0'] = os.path.join(root_path, 'real_experiments', 'random', '0', 'data.csv')
FILENAMES['random_1'] = os.path.join(root_path, 'real_experiments', 'random', '1', 'data.csv')
FILENAMES['human_0'] = os.path.join(root_path, 'real_experiments', 'human', '0', 'data.csv')
FILENAMES['human_1'] = os.path.join(root_path, 'real_experiments', 'human', '1', 'data.csv')
FILENAMES['uncertainty_0'] = os.path.join(root_path, 'real_experiments', 'uncertainty', '0', '0010', 'data.csv')
FILENAMES['uncertainty_1'] = os.path.join(root_path, 'real_experiments', 'uncertainty', '1', '0010', 'data.csv')

def get_all_data():
    # we get the data common to all files
    X_test, y_test = get_init_data(FILENAMES['random_0'])
    for _, fname in FILENAMES.items():
        # we add the new data collected for each method
        X, y = get_new_data(fname)
        X_test = np.vstack((X_test, X))
        y_test = np.hstack((y_test, y))
    return X_test, y_test

def get_data_xp(xp_key):
    return read_data(FILENAMES[xp_key])
