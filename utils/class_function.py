import numpy as np


def true_false_to_class_int(true_false):
    return np.array(true_false, dtype='int')


def in_circle(X, center, radius):
    X = np.atleast_2d(X)
    inside = np.linalg.norm(X - center, axis=1) < radius
    return true_false_to_class_int(inside)


def below_sinus_2D(X, amp, freq, shift):
    X = np.atleast_2d(X)
    below = amp * np.sin(freq * X[:, 0]) + shift < X[:, 1]
    return true_false_to_class_int(below)


def left_sinus_2D(X, amp, freq, shift):
    X = np.atleast_2d(X)
    left = amp * np.sin(freq * X[:, 1]) + shift < X[:, 2]
    return true_false_to_class_int(left)
