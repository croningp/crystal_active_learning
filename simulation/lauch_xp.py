
import os

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# adding parent directory to path, so we can access the utils easily
import sys
root_path = os.path.join(HERE_PATH, '..')
sys.path.append(root_path)

#
import numpy as np

#
from utils.experiment import run_full_xp
from utils.experiment import run_random_xp
from utils.experiment import evaluate_xp
from utils.experiment import save_eval

from utils.plot_helpers import plot_full_xp_in_folder
from utils.class_function import in_circle
from utils.class_function import below_sinus_2D

from utils.classifier import generate_random_X


if __name__ == '__main__':

    N_START = 20
    N_ITERATION = 10
    N_SAMPLING = 1000
    N_SELECTED = 10
    N_TEST = 1000

    problems = {}
    problems['circle'] = lambda X: in_circle(X, [0.75, 0.5], 0.25)
    problems['sinus'] = lambda X: below_sinus_2D(X, 0.25, 10, 0.5)

    n_run = 50
    for problem_name, class_func in problems.items():
        for i in range(n_run):

            print '{}: {}/{}'.format(problem_name, i + 1, n_run)

            base_save_folder = os.path.join(HERE_PATH, 'plot', problem_name)

            np.random.seed(i)
            X = generate_random_X(N_START, class_func)

            X_test = np.random.rand(N_TEST, 2)

            # uncertainty
            save_folder = os.path.join(base_save_folder, 'uncertainty', str(i))
            eval_savefilename = os.path.join(save_folder, 'xp_eval.json')

            uncertainty_all_info = run_full_xp(N_ITERATION, N_SELECTED, N_SAMPLING, class_func, X)
            plot_full_xp_in_folder(save_folder, class_func, uncertainty_all_info, exts=['.png'])

            uncertainty_xp_eval = evaluate_xp(uncertainty_all_info, X_test, class_func)
            save_eval(eval_savefilename, uncertainty_xp_eval)

            # random
            save_folder = os.path.join(base_save_folder, 'uncertainty_single', str(i))
            eval_savefilename = os.path.join(save_folder, 'xp_eval.json')

            single_n_iteration = N_ITERATION * N_SELECTED
            uncertainty_single_all_info = run_full_xp(single_n_iteration, 1, N_SAMPLING, class_func, X)
            plot_full_xp_in_folder(save_folder, class_func, uncertainty_single_all_info, exts=['.png'])

            uncertainty_single_xp_eval = evaluate_xp(uncertainty_single_all_info, X_test, class_func)
            save_eval(eval_savefilename, uncertainty_single_xp_eval)

            # random
            save_folder = os.path.join(base_save_folder, 'random', str(i))
            eval_savefilename = os.path.join(save_folder, 'xp_eval.json')

            all_random_info = run_random_xp(N_ITERATION, N_SELECTED, N_SAMPLING, class_func, X)
            plot_full_xp_in_folder(save_folder, class_func, all_random_info, exts=['.png'])

            random_xp_eval = evaluate_xp(all_random_info, X_test, class_func)
            save_eval(eval_savefilename, random_xp_eval)
