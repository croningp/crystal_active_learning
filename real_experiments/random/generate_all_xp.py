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


N_GENERATED = 100
TOTAL_VOLUME_IN_ML = 15.0
N_DECIMAL_EQUAL = 2

if __name__ == '__main__':

    import sys
    import filetools

    if len(sys.argv) != 2:
        print 'Please specify a root folder as argument'

    # seed
    seed = int(sys.argv[1])
    random.seed(seed)
    np.random.seed(seed)

    #
    root_folder = os.path.join(HERE_PATH, sys.argv[1])
    filetools.ensure_dir(root_folder)

    # load data
    current_datafile = os.path.join(HERE_PATH, 'init_data.csv')
    X, y = read_data(current_datafile)
    # check everything is fine
    np.testing.assert_array_almost_equal(np.sum(X, axis=1), TOTAL_VOLUME_IN_ML, decimal=N_DECIMAL_EQUAL)

    #
    X_selected = np.random.rand(N_GENERATED, X.shape[1])
    X_selected = proba_normalize_row(X_selected)
    X_selected = TOTAL_VOLUME_IN_ML * X_selected

    # save new csv

    # xout
    X_out = np.vstack((X, X_selected))

    # check everything is fine
    np.testing.assert_array_almost_equal(np.sum(X_out, axis=1), TOTAL_VOLUME_IN_ML, decimal=N_DECIMAL_EQUAL)

    # save
    next_datafile = os.path.join(root_folder, 'data.csv')
    write_data(next_datafile, X_out, y)
