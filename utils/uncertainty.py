import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

from tools import soft_max
from tools import probabilistic_choice
from tools import compute_normalized_entropy


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


def compute_weights(X_test, X_proba, clf=None, X_repulse=None):
    entropy = compute_normalized_entropy(X_proba)

    repulsion = np.zeros(entropy.shape)
    if clf is not None:
        if X_repulse is not None and X_repulse.shape[1] > 0:
            gamma_repulse = clf.best_estimator_.gamma * 10.
            repulsion = compute_normalized_repulsion(X_test, X_repulse, gamma_repulse)

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


def generate_next_samples(n_samples, clf, n_dim, n_sampling):

    all_run_info = []
    all_X_selected = []

    for i in range(n_samples):

        X_sampled = np.random.rand(n_sampling, n_dim)

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
