import scipy
import numpy as np


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
