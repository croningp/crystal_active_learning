import multiprocessing

import numpy as np
from sklearn.svm import SVC
from sklearn import grid_search
from sklearn.metrics import accuracy_score


def train_classifier(X, y):
    param_grid = {
        'C': np.logspace(-5, 5, 21),
        'gamma': np.logspace(-5, 5, 21),
        'kernel': ['rbf'],
        'probability': [True]
    }

    n_cv = min(get_min_sample_per_class(y), 10)

    clf = grid_search.GridSearchCV(SVC(), param_grid, n_jobs=multiprocessing.cpu_count(), cv=n_cv)
    clf.fit(X, y)
    return clf


def count_support_vectors(clf):
    return len(clf.best_estimator_.support_)


def classification_accuracy(X, class_func, clf):
    y_true = class_func(X)
    y_pred = clf.predict(X)
    return accuracy_score(y_true, y_pred)


def generate_random_X(n_samples, class_func, n_dim=2, min_sample_per_class=5):
    X = np.random.rand(n_samples, n_dim)
    while not is_start_y_valid(X, class_func, min_sample_per_class):
        X = np.random.rand(n_samples, n_dim)
    return X


def get_min_sample_per_class(y):
    min_sample_per_class = np.inf
    for class_number in np.unique(y):
        n_sample = np.sum(y == class_number)
        if n_sample < min_sample_per_class:
            min_sample_per_class = n_sample
    return min_sample_per_class


def is_start_y_valid(X, class_func, min_sample_per_class=2):
    y = class_func(X)
    return get_min_sample_per_class(y) >= min_sample_per_class
