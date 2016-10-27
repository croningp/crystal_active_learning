import os

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# adding parent directory to path, so we can access the utils easily
import sys
root_path = os.path.join(HERE_PATH, '..')
sys.path.append(root_path)

from utils.csv_helpers import read_data


N_INIT_POINTS = 89

def get_init_data(filename):
    X, y = read_data(filename)
    return X[:N_INIT_POINTS, :], y[:N_INIT_POINTS]


def get_new_data(filename):
    X, y = read_data(filename)
    return X[N_INIT_POINTS:, :], y[N_INIT_POINTS:]
