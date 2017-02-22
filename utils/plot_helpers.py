import os

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import filetools


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


def plot_current_model(class_func, clf, X, y):
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
    y_grid = class_func(X_grid)
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
    if X_sampled is not None and X_sampled.shape[1] > 0:
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
    if X_sampled is not None and X_sampled.shape[1] > 0:
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


def save_and_close_figure(fig, filebasename, exts=['.png', '.eps', '.svg'], dpi=100, legend=None):

    for ext in exts:
        # save
        filepath = filebasename + ext
        if legend is not None:
            fig.savefig(filepath, dpi=dpi, bbox_extra_artists=(legend,), bbox_inches='tight')
        else:
            fig.savefig(filepath, dpi=dpi)

    plt.close(fig)


def plot_all_run_info(X, y, all_run_info):

    all_fig = []
    all_X_selected = []

    for i, run_info in enumerate(all_run_info):
        all_fig.append(plot_uncertainty_pipeline(X, y, run_info, np.atleast_2d(all_X_selected)))
        all_X_selected.append(run_info['X_selected'])

    return all_fig


def plot_iteration_in_folder(foldername, class_func, clf, X, y, all_run_info=None, exts=['.png', '.eps', '.svg']):

    fig_model = plot_current_model(class_func, clf, X, y)
    model_foldername = os.path.join(foldername, 'model')
    model_filename = filetools.generate_incremental_filename(model_foldername, n_digit=6)
    save_and_close_figure(fig_model, model_filename, exts=exts)

    if all_run_info is not None:
        all_fig = plot_all_run_info(X, y, all_run_info)
        uncertainty_foldername = os.path.join(foldername, 'uncertainty')
        for fig in all_fig:
            uncertainty_filename = filetools.generate_incremental_filename(uncertainty_foldername, n_digit=6)
            save_and_close_figure(fig, uncertainty_filename, exts=exts)


def plot_full_xp_in_folder(foldername, class_func, all_info, exts=['.png', '.eps', '.svg']):

    for info in all_info:

        if 'all_run_info' in info:
            all_run_info = info['all_run_info']
        else:
            all_run_info = None

        plot_iteration_in_folder(foldername, class_func, info['clf'], info['X'], info['y'], all_run_info, exts=exts)
