import os

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# adding parent directory to path, so we can access the utils easily
import sys
root_path = os.path.join(HERE_PATH, '..', '..')
sys.path.append(root_path)

import pickle
import numpy as np

from utils.plot_helpers import save_and_close_figure

##
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# design figure
fontsize = 26
matplotlib.rc('xtick', labelsize=22)
matplotlib.rc('ytick', labelsize=22)
matplotlib.rcParams.update({'font.size': fontsize})
linewidth = 3
markersize = 10

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':

    import filetools

    learning_data_filename = os.path.join(HERE_PATH, 'learning_data.pkl')
    data = load_pickle(learning_data_filename)

    for method_name, method in data.items():

        RESULTS = data[method_name]['results']

        TEST_RANGE = RESULTS['random_0']['test_range']

        AVG_RESULTS = {}
        AVG_RESULTS['HUMAN'] = 100 * np.mean(np.vstack((RESULTS['human_0']['unbiased_acc'], RESULTS['human_1']['unbiased_acc'])), axis=0)
        AVG_RESULTS['ALGORITHM'] = 100 * np.mean(np.vstack((RESULTS['uncertainty_0']['unbiased_acc'], RESULTS['uncertainty_1']['unbiased_acc'])), axis=0)
        AVG_RESULTS['RANDOM'] = 100 * np.mean(np.vstack((RESULTS['random_0']['unbiased_acc'], RESULTS['random_1']['unbiased_acc'])), axis=0)

        ##
        fig = plt.figure(figsize=(12, 8))

        plt.plot(TEST_RANGE, AVG_RESULTS['ALGORITHM'], 'r', linestyle='-', linewidth=linewidth, marker='o', markersize=markersize)
        plt.plot(TEST_RANGE, AVG_RESULTS['HUMAN'], 'g', linestyle='-', linewidth=linewidth, marker='s', markersize=markersize)
        plt.plot(TEST_RANGE, AVG_RESULTS['RANDOM'], 'b', linestyle='-', linewidth=linewidth, marker='D', markersize=markersize)

        plt.title('Evolution of Crystalization Model Quality', fontsize=fontsize)
        plt.legend(['Algorithm', 'Human', 'Random'], fontsize=fontsize, loc=2)
        plt.xlim([-1, 101])

        if method_name == 'Adaboost':
            plt.ylim([60, 85])
            y_tick_pos = [60, 65, 70, 75, 80, 85]
        else:
            plt.ylim([65, 85])
            y_tick_pos = [65, 70, 75, 80, 85]
        plt.yticks(y_tick_pos, ['{}%'.format(i) for i in y_tick_pos])

        plt.xlabel('Number of Experiments', fontsize=fontsize)
        plt.ylabel('Prediction Accuracy - %', fontsize=fontsize)

        #
        PLOT_FOLDER = os.path.join(HERE_PATH, 'plot')
        filetools.ensure_dir(PLOT_FOLDER)

        plot_filename = os.path.join(PLOT_FOLDER, 'learning_curve_unbiased_{}'.format(method_name))
        save_and_close_figure(fig, plot_filename)
