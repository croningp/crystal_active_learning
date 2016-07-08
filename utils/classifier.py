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

    clf = grid_search.GridSearchCV(SVC(), param_grid, n_jobs=multiprocessing.cpu_count())
    clf.fit(X, y)
    return clf


def count_support_vectors(clf):
    return len(clf.best_estimator_.support_)


def classification_accuracy(X, class_func, clf):
    y_true = class_func(X)
    y_pred = clf.predict(X)
    return accuracy_score(y_true, y_pred)
