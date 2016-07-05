import multiprocessing

import scipy
import numpy as np
from sklearn.svm import SVC
from sklearn import grid_search
from sklearn.metrics.pairwise import rbf_kernel


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

    even_proba = np.ones((1, proba.shape[1])) / proba.shape[1]
    max_entropy = scipy.stats.entropy(even_proba.T)

    return entropy / max_entropy


def soft_max(values, temp=1):
    x = np.array(values)
    e = np.exp(x / temp)
    soft_max_x = e / np.sum(e, axis=0)
    return soft_max_x


def probabilistic_choice(proba_distribution):
    return int(np.random.choice(range(len(proba_distribution)), 1, p=proba_distribution))


def train_classifier(X, y):
    param_grid = {
        'C': np.logspace(-3, 2, 11),
        'gamma': np.logspace(-3, 2, 11),
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
    return out / float(n_repulse)


def compute_weights(clf, X_test, X_repulse=None):
    proba = clf.best_estimator_.predict_proba(X_test)
    entropy = compute_normalized_entropy(proba)

    if X_repulse is not None:
        gamma_repulse = clf.best_estimator_.gamma * 10.
        repulsion = compute_normalized_repulsion(X_test, X_repulse, gamma_repulse)
    else:
        repulsion = np.zeros(entropy.shape)

    weights = entropy * (1 - repulsion)

    return weights, entropy, repulsion


def compute_best_temperature(weights, t_candidate=np.logspace(-4, 1, 11), percentage_to_consider=0.1, target_proba_sum=0.9):

    errors = []
    for t in t_candidate:
        p = soft_max(weights, t)
        if np.any(np.isnan(p)):
            error = np.inf
        else:
            sorted_weights = np.sort(weights)
            n_above_mean = np.mean(p > np.mean(p))
            proba_accumulated =
            error = np.abs(proba_accumulated - target_proba_sum)
        errors.append(error)
    return t_candidate[np.argmin(errors)]


if __name__ == '__main__':

    from sklearn import datasets

    def circle(X):
        X = np.atleast_2d(X)
        (n_lines, _) = X.shape

        out = np.zeros(shape=(n_lines, ))
        for i in range(n_lines):
            dist = np.sqrt((X[i, 0] - 0.5) ** 2 + (X[i, 1] - 0.5) ** 2)
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

    iris = datasets.load_iris()
    X = np.random.rand(50, 2)
    y = line(X)
    clf = train_classifier(X, y)

    N_SAMPLING = 1000
    N_SELECTED = 5

    X_sampled = []
    for i in range(N_SELECTED):

        X_to_test = np.random.rand(N_SAMPLING, X.shape[1])

        weights, entropy, repulsion = compute_weights(clf, X_to_test, X_sampled)

        best_temperature = compute_best_temperature(weights)
        selection_proba = soft_max(weights, best_temperature)

        sampled_index = probabilistic_choice(selection_proba)

        # sampled_index = np.argmax(weights)
        X_sampled.append(X_to_test[sampled_index, :])
    X_sampled = np.array(X_sampled)

    #
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns

    # palette = sns.cubehelix_palette(20, start=.5, rot=-.75)
    # cmap = mpl.colors.ListedColormap(palette)
    cmap = plt.get_cmap('jet')

    plt.ion()

    ax = plt.subplot(2, 2, 1)

    # cmap = plt.get_cmap('Spectral')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, s=50)
    plt.scatter(X_sampled[:, 0], X_sampled[:, 1], c='g', s=50)

    plt.axis('equal')
    # plt.xlim([0, 1])
    plt.ylim([0, 1])

    #
    ax = plt.subplot(2, 2, 2)

    plt.scatter(X_to_test[:, 0], X_to_test[:, 1], c=weights, cmap=cmap, s=50)

    plt.axis('equal')
    # plt.xlim([0, 1])
    plt.ylim([0, 1])

    #
    ax = plt.subplot(2, 2, 3)

    plt.scatter(X_to_test[:, 0], X_to_test[:, 1], c=entropy, cmap=cmap, s=50)

    plt.axis('equal')
    # plt.xlim([0, 1])
    plt.ylim([0, 1])

    #
    ax = plt.subplot(2, 2, 4)

    plt.scatter(X_to_test[:, 0], X_to_test[:, 1], c=repulsion, cmap=cmap, s=50)

    plt.axis('equal')
    # plt.xlim([0, 1])
    plt.ylim([0, 1])

    #
    plt.draw()
