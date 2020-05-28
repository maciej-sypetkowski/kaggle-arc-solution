#!/usr/bin/env python3

from pathlib import Path
import pickle
from argparse import ArgumentParser

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import gridspec

import numpy as np
import sklearn

from main import ConfigStats, TaskStat, config_str


# Taken (modified) from: https://www.kaggle.com/boliu0/visualizing-all-task-pairs-with-gridlines
def plot_one(fig, img, grid_elem, label=''):
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)

    ax = fig.add_subplot(grid_elem)
    ax.imshow(img, cmap=cmap, norm=norm)
    ax.grid(True, which='both', color='lightgrey', linewidth=0.5)
    ax.set_yticks([x - 0.5 for x in range(1 + len(img))])
    ax.set_xticks([x - 0.5 for x in range(1 + len(img[0]))])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(label)


def plot_task(label, task, pred, model, feature_names, conf_name='', out_path=None):
    task['pred'] = [{'output': pred, 'input': inp['input']}
                    for pred, inp in zip(pred, task['test'])]

    n_train = len(task['train'])
    n_test = len(task['test'])
    assert n_test == len(pred)
    n = n_train + n_test + len(pred) + 1

    fig = plt.figure(figsize=(n * 4, 8))
    gs = gridspec.GridSpec(2, n, width_ratios=[1] * (n - 1) + [4])

    for i in range(n_train):
        plot_one(fig, task['train'][i]['input'], gs[0, i], 'train input')
        plot_one(fig, task['train'][i]['output'], gs[1, i], 'train output')

    for i in range(n_test):
        plot_one(fig, task['test'][i]['input'],
                 gs[0, n_train + i], 'test input')
        plot_one(fig, task['test'][i]['output'],
                 gs[1, n_train + i], 'test output')

    for i in range(n_test):
        plot_one(fig, task['pred'][i]['input'],
                 gs[0, n_train + n_test + i], 'test input')
        plot_one(fig, task['pred'][i]['output'],
                 gs[1, n_train + n_test + i], 'predicted output')

    ax = fig.add_subplot(gs[:, n - 1])
    plt.sca(ax)
    try:
        sklearn.tree.plot_tree(model, feature_names=feature_names, filled=True)
        ax.set_title(conf_name)
    except (TypeError, AttributeError):
        pass

    plt.tight_layout()
    if out_path is None:
        plt.show()
    else:
        out_path = Path(out_path) / conf_name / f'{label}.png'
        print('Saving:', out_path)
        out_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(out_path)

    plt.close(fig)


def parse_args(args=None):
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', type=Path,
                        help="Path to pickled results.")
    parser.add_argument('-o', '--output', type=Path, default=Path('out'),
                        help="Where to save plots.")
    args = parser.parse_args(args)
    return args


def plot_task_stat(task_index, task_stat, conf_str, out_path=None):
    """
    If out_path is None, use plt.show().
    """
    pred = list(map(
        lambda x: x[0][0] if len(x) > 0 else np.array([[0]]), task_stat.predictions))
    plot_task(label=str(task_index),
              task=task_stat.task,
              pred=pred,
              model=task_stat.model,
              feature_names=task_stat.feature_names,
              conf_name=conf_str,
              out_path=out_path,
              )


def main(args):
    with args.input.open('rb') as f:
        stats = pickle.load(f)

    # expose config str to each task stat
    for config_stat in stats:
        conf_str = config_str(config_stat.config)
        for task_stat in config_stat:
            task_stat.conf_str = conf_str

    # for each task, plot one task with the smallest tree that gives correct answer
    for task_index, task_stats in enumerate(zip(*stats)):
        task_stat = min(task_stats, key=lambda x: (not x.correct().all(),
            x.model.tree_.node_count if x.model is not None else float('inf')))
        conf_str = task_stat.conf_str
        if task_stat.correct().all():  # when to plot
            plot_task_stat(task_index, task_stat, conf_str, out_path=args.output)

    # for each task, plot  all correct trees with <20 nodes
    # for config_stat in stats:
    #     conf_str = config_str(config_stat.config)
    #     for task_index, task_stat in enumerate(config_stat):
    #         if task_stat.correct() and task_stat.model.tree_.node_count < 20:  # when to plot
    #             plot_task_stat(task_index, task_stat, conf_str, out_path=args.output)


if __name__ == '__main__':
    main(parse_args())
