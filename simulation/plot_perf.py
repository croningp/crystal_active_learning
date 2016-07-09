import os

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# adding parent directory to path, so we can access the utils easily
import sys
root_path = os.path.join(HERE_PATH, '..')
sys.path.append(root_path)

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import filetools

from utils.experiment import read_eval


if __name__ == '__main__':

    problem_names = ['circle', 'sinus']
    method_names = ['random', 'uncertainty', 'uncertainty_single']

    results = {}

    for problem_name in problem_names:
        results[problem_name] = {}
        for method_name in method_names:

            foldername = os.path.join(HERE_PATH, 'plot', problem_name, method_name)
            all_eval_file = filetools.list_files(foldername, ['xp_eval.json'])

            accuracy_score = []
            n_support_vector = []
            for eval_file in all_eval_file:
                xp_eval = read_eval(eval_file)
                accuracy_score.append(xp_eval['accuracy_score'])
                n_support_vector.append(xp_eval['n_support_vector'])

            result = {}
            result['accuracy_score'] = accuracy_score
            result['mean_accuracy_score'] = np.mean(accuracy_score, axis=0)
            result['std_accuracy_score'] = np.std(accuracy_score, axis=0)

            result['n_support_vector'] = n_support_vector
            result['mean_n_support_vector'] = np.mean(n_support_vector, axis=0)
            result['std_n_support_vector'] = np.std(n_support_vector, axis=0)

            results[problem_name][method_name] = result

    figures = []
    for problem_name in problem_names:
        fig = plt.figure()
        for method_name in method_names:

            data = results[problem_name][method_name]['accuracy_score']
            data = np.array(data)

            if method_name == 'uncertainty_single':
                data = data[:, 0:-1:10]

            sns.tsplot(data)
        plt.legend(method_names)
        figures.append(fig)
