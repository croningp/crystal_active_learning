import multiprocessing

import scipy
import numpy as np
from sklearn.svm import SVC
from sklearn import grid_search
from sklearn.metrics.pairwise import rbf_kernel

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


# The method is based on the notion that
# ln(a + b) = ln{exp[ln(a) - ln(b)] + 1} + ln(b).
def add_lns(a_ln, b_ln):
    return np.log(np.exp(a_ln - b_ln) + 1) + b_ln


def sum_log_array(log_array):
    log_array = np.atleast_2d(log_array)
    (n_lines, n_columns) = log_array.shape

    out = np.zeros(shape=(n_lines, ))
    for i in range(n_lines):
        if n_columns > 1:
            logSum = add_lns(log_array[i, 0], log_array[i, 1])
            for j in range(n_columns - 2):
                logSum = add_lns(logSum, log_array[i, j + 2])
        else:
            logSum = log_array[i, 1]
        out[i] = logSum
    return out


def normalize_log_array(log_array):
    init_shape = np.array(log_array).shape

    log_array = np.atleast_2d(log_array)
    (n_lines, n_columns) = log_array.shape

    logsum_array = sum_log_array(log_array)

    out = np.zeros(shape=(n_lines, n_columns))
    for i in range(n_lines):
        out[i, :] = np.exp(log_array[i, :] - logsum_array[i])

    return out.reshape(out, init_shape)


def compute_normalized_entropy(proba):
    proba = np.atleast_2d(proba)
    entropy = scipy.stats.entropy(proba.T)

    # even_proba = np.ones((1, proba.shape[1])) / proba.shape[1]
    # max_entropy = scipy.stats.entropy(even_proba.T)

    return entropy / np.max(entropy)


def soft_max(values, temp=1):
    x = np.array(values)
    e = np.exp(x / temp)
    soft_max_x = e / np.sum(e, axis=0)
    return soft_max_x


def probabilistic_choice(proba_distribution):
    return int(np.random.choice(range(len(proba_distribution)), 1, p=proba_distribution))


def train_classifier(X, y):
    param_grid = {
        'C': np.logspace(-5, 5, 21),
        'gamma': np.logspace(-5, 5, 21),
        'kernel': ['rbf'],
        'probability': [True]
    }

    clf = grid_search.GridSearchCV(SVC(), param_grid, n_jobs=multiprocessing.cpu_count())
    clf.fit(X, y)
    return clf


def compute_normalized_repulsion(X, X_repulse, gamma_repulse):
    X = np.atleast_2d(X)
    X_repulse = np.atleast_2d(X_repulse)

    (n_lines, _) = X.shape
    (n_repulse, n_dim) = X_repulse.shape

    out = np.zeros(shape=(n_lines, ))
    if n_dim > 0:
        for i in range(n_lines):
            for j in range(n_repulse):
                out[i] += rbf_kernel(X[i, :], X_repulse[j, :], gamma=gamma_repulse)
    return out / np.max(out)


def compute_weights(X_test, X_proba, X_repulse=None):
    entropy = compute_normalized_entropy(X_proba)

    if X_repulse is not None:
        gamma_repulse = clf.best_estimator_.gamma * 10.
        repulsion = compute_normalized_repulsion(X_test, X_repulse, gamma_repulse)
    else:
        repulsion = np.zeros(entropy.shape)

    weights = entropy * (1 - repulsion)

    return weights, entropy, repulsion


def compute_best_temperature(weights, t_candidate=np.logspace(-5, 1, 100), percentage_to_consider=0.05, target_proba_sum=0.95):

    errors = []
    for t in t_candidate:
        p = soft_max(weights, t)
        if np.any(np.isnan(p)):
            error = np.inf
        else:
            sorted_p = np.sort(p)
            n_ratio = int(percentage_to_consider * len(p))
            proba_accumulated = np.sum(sorted_p[-n_ratio:])
            error = np.abs(proba_accumulated - target_proba_sum)
        errors.append(error)
    return t_candidate[np.argmin(errors)]


def generate_next_samples(n_samples, clf, n_sampling):

    all_run_info = []
    all_X_selected = []

    for i in range(n_samples):

        X_sampled = np.random.rand(n_sampling, X.shape[1])

        X_proba = clf.best_estimator_.predict_proba(X_sampled)

        weights, entropy, repulsion = compute_weights(X_sampled, X_proba, all_X_selected)

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


def scatter_2D(ax, X, **args):
    X = np.atleast_2d(X)
    plt.scatter(X[:, 0], X[:, 1], **args)


def set_ax_lim(ax):
    delta = 0.05
    ax.set_aspect('equal')
    ax.set_xlim([-delta, 1 + delta])
    ax.set_ylim([-delta, 1 + delta])


def set_ax_text(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def plot_current_model(func, clf, X, y):
    class_palette = sns.diverging_palette(145, 280, s=100, l=50, n=7)
    c_class_map = mpl.colors.ListedColormap(class_palette)

    marker_size = 50

    n_row = 1
    n_column = 3

    grid_x, grid_y = np.mgrid[0:1:100j, 0:1:100j]
    # trick on coordinate for imshow to display as a scatter plot
    X_grid = np.vstack((grid_y.flatten(), 1 - grid_x.flatten())).T

    #
    fig = plt.figure(figsize=(12, 4))

    #
    y_grid = func(X_grid)
    y_img = y_grid.reshape(grid_x.shape)

    ax = plt.subplot(n_row, n_column, 1)
    plt.imshow(y_img, cmap=c_class_map, extent=(0, 1, 0, 1))
    set_ax_lim(ax)
    set_ax_text(ax, 'X1', 'X2', 'True Class')

    #
    ax = plt.subplot(n_row, n_column, 2)
    scatter_2D(ax, X, c=y, cmap=c_class_map, s=marker_size)
    set_ax_lim(ax)
    set_ax_text(ax, 'X1', 'X2', 'Training data')

    #
    y_grid = clf.best_estimator_.predict_proba(X_grid)[:, 1]
    y_img = y_grid.reshape(grid_x.shape)

    ax = plt.subplot(n_row, n_column, 3)
    plt.imshow(y_img, cmap=c_class_map, extent=(0, 1, 0, 1))
    set_ax_lim(ax)
    set_ax_text(ax, 'X1', 'X2', 'Predicted Class Proba')

    return fig


def plot_uncertainty_pipeline(X, y, run_info, X_sampled=None):

    class_palette = sns.diverging_palette(145, 280, s=100, l=50, n=7)
    c_class_map = mpl.colors.ListedColormap(class_palette)

    c_heat_map = plt.get_cmap('jet')

    light_red = np.array([1., 0.4, 0.4, 1.])
    dark_gray = np.array([0.25, 0.25, 0.25, 1.])

    marker_size = 50

    n_row = 3
    n_column = 3

    #
    fig = plt.figure(figsize=(12, 12))
    #
    ax = plt.subplot(n_row, n_column, 1)
    scatter_2D(ax, X, c=y, cmap=c_class_map, s=marker_size)
    if X_sampled is not None:
        scatter_2D(ax, X_sampled, c=dark_gray, s=marker_size, marker='s')
    set_ax_lim(ax)
    set_ax_text(ax, 'X1', 'X2', 'Training data')

    #
    ax = plt.subplot(n_row, n_column, 2)
    scatter_2D(ax, run_info['X_sampled'], c=run_info['entropy'], cmap=c_heat_map, s=marker_size)
    set_ax_lim(ax)
    set_ax_text(ax, 'X1', 'X2', 'Entropy')
    #
    ax = plt.subplot(n_row, n_column, 3)
    scatter_2D(ax, run_info['X_sampled'], c=run_info['repulsion'], cmap=c_heat_map, s=marker_size)
    set_ax_lim(ax)
    set_ax_text(ax, 'X1', 'X2', 'Repulsion')

    #
    ax = plt.subplot(n_row, n_column, 6)
    scatter_2D(ax, run_info['X_sampled'], c=run_info['weights'], cmap=c_heat_map, s=marker_size)
    set_ax_lim(ax)
    set_ax_text(ax, 'X1', 'X2', 'Weights')

    #
    ax = plt.subplot(n_row, n_column, 5)
    scatter_2D(ax, run_info['X_sampled'], c=run_info['selection_proba'], cmap=c_heat_map, s=marker_size)
    set_ax_lim(ax)
    set_ax_text(ax, 'X1', 'X2', 'Probabilities')
    #
    ax = plt.subplot(n_row, n_column, 4)
    if X_sampled is not None:
        scatter_2D(ax, X_sampled, c=dark_gray, s=marker_size, marker='s')
    scatter_2D(ax, run_info['X_selected'], c=light_red, s=marker_size, marker='s')
    set_ax_lim(ax)
    set_ax_text(ax, 'X1', 'X2', 'Selected experiment')

    #
    ax = plt.subplot(n_row, n_column, 9)
    ax.plot(run_info['weights'])
    ax.set_ylim([-0.05, 1.05])
    set_ax_text(ax, 'Sampled XP number', 'Weights', '')
    #
    ax = plt.subplot(n_row, n_column, 8)
    ax.plot(run_info['selection_proba'])
    set_ax_text(ax, 'Sampled XP number', 'Probabilities', '')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_ylim([-ylim[1] / 20., ylim[1]])
    #
    ax = plt.subplot(n_row, n_column, 7)
    ax.scatter(run_info['sampled_index'], run_info['selection_proba'][run_info['sampled_index']], c=light_red, s=2 * marker_size, marker=(5, 1))
    ax.plot(run_info['selection_proba'])
    set_ax_text(ax, 'Sampled XP number', 'Probabilities', 'Selected experiment')
    ax.set_xlim(xlim)
    ax.set_ylim([-ylim[1] / 20., ylim[1]])

    return fig


if __name__ == '__main__':

    def circle(X):
        X = np.atleast_2d(X)
        (n_lines, _) = X.shape

        out = np.zeros(shape=(n_lines, ))
        for i in range(n_lines):
            dist = np.sqrt((X[i, 0] - 0.75) ** 2 + (X[i, 1] - 0.5) ** 2)
            if dist < 0.25:
                out[i] = 1
        return out

    def line(X):
        X = np.atleast_2d(X)
        (n_lines, _) = X.shape

        out = np.zeros(shape=(n_lines, ))
        for i in range(n_lines):
            if X[i, 0] < 0.5:
                out[i] = 1
        return out

    N_SAMPLING = 1000
    N_SELECTED = 10

    np.random.seed(0)

    X = np.random.rand(20, 2)
    y = circle(X)
    clf = train_classifier(X, y)

    all_X_selected, all_run_info = generate_next_samples(N_SELECTED, clf, N_SAMPLING)

    plot_current_model(circle, clf, X, y)

    plot_uncertainty_pipeline(X, y, all_run_info[9], all_X_selected)

    plt.show()
