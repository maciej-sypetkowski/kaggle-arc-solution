#!/usr/bin/env python3

import json
import operator
import pickle
import random
from argparse import ArgumentParser
from functools import partial, reduce
from itertools import product, starmap, zip_longest
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.tree

from transform import TException, TIdentity, TSeparationRemover, \
        TCCount, TCCountBg, TCPos, TAugFlips, TAugColor, TSequential
from featurize import FException, FGlobal, FConstant, \
        FTile, FScale, FSquare, FFactorDown2, FFactorDown1
from confidence import calc_confidences
from utils import flatten


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def parse_args(args=None):
    def bool_type(x):
        if x.lower() in ['1', 'true']:
            return True
        if x.lower() in ['0', 'false']:
            return False
        raise ValueError()

    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset-root', type=Path,
                        default='/kaggle/input/abstraction-and-reasoning-challenge/')
    parser.add_argument('-s', '--dataset-suffix', type=str, default='evaluation')
    parser.add_argument('-g', '--gen-submission', type=bool_type, default=True)
    parser.add_argument('-t', '--threads', type=int, default=2)
    parser.add_argument('-p', '--pickle-results', type=Path,
            help="Path to generated predictions and models. "
            "If the file exists it is loaded, otherwise it is calculated and written.")
    parser.add_argument('--save-all-predictions', type=bool_type, default=False,
            help="Allow writing more than 3 answers per test image to resulting submission file.")
    parser.add_argument('--train-confidence', type=Path, default=None,
            help="If specified, train and save confidence model pickle to a given path.")
    parser.add_argument('--load-confidence', type=Path, default=None,
            help="Confidence model pickle path.")
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args(args)

    if args.train_confidence is None and args.load_confidence is None:
        raise ValueError("Either --train-confidence or --load-confidence required.")

    if args.seed < 0:
        args.seed = np.random.randint(2 ** 16)
        print('Seed:', args.seed)

    set_seed(args.seed)
    return args


def save_output_csv(stats, save_all_predictions):
    final_rows = []
    for task in stats:
        for i, pr in enumerate(task.get_topk()):
            out_str = []
            for t in pr:
                s = [''.join(map(str, p)) for p in t]
                s = '|'.join(s)
                s = f'|{s}|'
                out_str.append(s)
            if save_all_predictions:
                if not out_str:
                    out_str = ['|0|']
            else:
                out_str = (out_str + ['|0|'] * 3)[:3]
            out_str = ' '.join(out_str)
            final_rows.append(('{}_{}'.format(task.task_id, i), out_str))
    df = pd.DataFrame(final_rows, columns=['output_id', 'output'])
    df.to_csv('submission.csv', index=False)


class TaskStat:
    def __init__(self, task):
        self.task = task  # used in plot_pickled_results.py

        self.task_id = ''
        self.transformer_error = False
        self.featurizer_error = False
        self.error = False
        self.deducible = False
        self.predictions = [] # [test_image][prediction_number] -> (prediction, confidence)
        self.ground_truth = []

        self.model = None
        self.feature_names = None
        self.transformer = None
        self.featurizer = None

    def is_prediction_done(self):
        return not self.error

    def __or__(self, other):
        ret = TaskStat(self.task)
        ret.transformer_error = self.transformer_error and other.transformer_error
        ret.featurizer_error = self.featurizer_error and other.featurizer_error
        ret.error = self.error and other.error
        ret.deducible = self.deducible or other.deducible
        ret.predictions = [x + y for x, y in zip(self.predictions, other.predictions)]
        ret.ground_truth = self.ground_truth
        assert all(starmap(lambda x, y: (x == y).all(),
                           zip(self.ground_truth, other.ground_truth)))
        ret.task_id = self.task_id
        assert self.task_id == other.task_id
        return ret

    def correct(self, k=None):
        return np.array(list(starmap(
            lambda xs, gt: any([np.array([x == gt]).all() for x in xs]) if gt is not None else False,
            zip_longest(self.get_topk(k), self.ground_truth))))

    def accuracy(self, k=None):
        return self.correct(k).mean()

    def get_topk(self, k=None):
        ret = []
        for preds in self.predictions:
            added = set()
            r = []
            for pred in sorted(filter(lambda x: x[1] is not None, preds), key=lambda x: -x[1]):
                tup = tuple(pred[0].reshape(-1))
                if tup not in added:
                    added.add(tup)
                    r.append(pred[0])
                    if k is not None and len(r) >= k:
                        break
            ret.append(r)
        return ret


class ConfigStats:

    def __init__(self, config):
        self.task_stats = []
        self.config = config  # used in plot_pickled_results.py

    def append_task_stat(self, task_stat):
        self.task_stats.append(task_stat)

    def __or__(self, other):
        ret = ConfigStats(self.config)
        ret.task_stats = [a | b for a, b in zip(self.task_stats, other.task_stats)]
        return ret

    def __iter__(self):
        return iter(self.task_stats)

    def __len__(self):
        return len(self.task_stats)

    def __getitem__(self, i):
        return self.task_stats[i]

    def __str__(self):
        def samples_correct(k=None):
            return sum(map(lambda x: sum(x.correct(k)), self))

        samples_num = sum(map(lambda x: len(x.predictions), self))

        def tasks_correct_list(k=None):
            return list(map(lambda x: (reduce(operator.mul, x.correct(k)) == 1)
                                      if len(x.predictions) else False, self))

        def tasks_correct(k=None): return sum(tasks_correct_list(k))
        tasks_num = len(self)
        transformer_errors = sum(map(lambda x: x.transformer_error, self))
        featurizer_errors = sum(map(lambda x: x.featurizer_error, self))
        errors = sum(map(lambda x: x.error, self))
        deducible_counts = sum(map(lambda x: x.deducible, self))
        return f'''\
Accuracy            {samples_correct() / max(samples_num, 1)} ({samples_correct()} / {samples_num})
Accuracy-top1       {samples_correct(1) / max(samples_num, 1)} ({samples_correct(1)} / {samples_num})
Accuracy-top3       {samples_correct(3) / max(samples_num, 1)} ({samples_correct(3)} / {samples_num})
Task-accuracy       {tasks_correct() / tasks_num} ({tasks_correct()} / {tasks_num})
Task-accuracy-top1  {tasks_correct(1) / tasks_num} ({tasks_correct(1)} / {tasks_num})
Task-accuracy-top3  {tasks_correct(3) / tasks_num} ({tasks_correct(3)} / {tasks_num})
Transformer-errors  {transformer_errors}
Featurizer-errors   {featurizer_errors}
Errors              {errors}
Deducible           {deducible_counts}
Deducible&Correct   {sum(np.array(list(map(lambda x: x.deducible, self))) & np.array(tasks_correct_list()))}
'''


def is_deducible(x, y):
    data = x.join(pd.DataFrame(y, columns=['label']))
    cols = list(data.columns)
    cols.pop(cols.index('label'))
    for i, j in data.groupby(cols):
        if len(j.label.unique()) != 1:
            return False
    return True


def _process(config, task, in_images, out_images):
    TransformerClass, FeaturizerClass, ModelClass = config
    task_stat = TaskStat(task)

    task_transformer = TransformerClass(in_images, out_images)
    transformers = [task_transformer.get_transformer(in_image) for in_image in in_images]
    transformed_in_images = flatten(
        [transformer.transformed_images for transformer in transformers])
    transformed_out_images = flatten(
            [transformer.transform_output(out_image)
                for transformer, out_image in zip(transformers, out_images)])

    task_featurizer = FeaturizerClass(transformed_in_images, transformed_out_images)
    featurizers = [task_featurizer.get_featurizer(in_image) for in_image in transformed_in_images]
    for featurizer in featurizers:
        featurizer.calc_features()

    features = [featurizer.features for featurizer in featurizers]

    def column_order(name):
        if 'Neigh' in name:
            if '0,0' in name or '0, 0' in name:
                return 0, name
            else:
                return 1, name
        elif 'Ray' in name:
            return 2, name
        else:
            return 3, name

    columns = list(features[0].columns)
    columns = sorted(columns, key=column_order, reverse=False)
    features = [pd.DataFrame(np.array(f[columns]), columns=columns)
                for f in features]

    labels = [featurizer.get_labels(out_image) for featurizer, out_image in zip(
        featurizers, transformed_out_images)]

    train_features = pd.concat(features[:len(labels)], ignore_index=True)

    model = ModelClass(max_features=len(columns))
    task_stat.deducible = is_deducible(train_features, np.concatenate(labels))
    model.fit(train_features, np.concatenate(labels))
    raw_predictions = [model.predict(features)
                       for features in features[len(labels):]]
    assembled_predictions = [featurizer.assemble(prediction)
            for featurizer, prediction in zip(featurizers[len(labels):], raw_predictions)]

    nested_predictions = []
    k = 0
    for transformer in transformers[len(out_images):]:
        nested_predictions.append(
            assembled_predictions[k: k + len(transformer.transformed_images)])
        k += len(transformer.transformed_images)

    final_predictions = [transformer.transform_output_inverse(prediction)
            for transformer, prediction in zip(transformers[len(out_images):], nested_predictions)]

    task_stat.model = model
    task_stat.feature_names = columns
    task_stat.transformer = task_transformer
    task_stat.featurizer = task_featurizer

    confidences = [0] * len(raw_predictions)
    task_stat.predictions = list(
        starmap(lambda x, c: [[x, c]], zip(final_predictions, confidences)))

    return task_stat


def process(task_index, task, config, in_images, out_images, valid_out_images, seed):
    set_seed(seed)

    task_stat = TaskStat(task)
    try:
        task_stat = _process(config, task, in_images, out_images)
    except FException:
        task_stat.featurizer_error = True
        task_stat.error = True
        for _ in range(len(in_images) - len(out_images)):
            task_stat.predictions.append([])
    except TException:
        task_stat.transformer_error = True
        task_stat.error = True
        for _ in range(len(in_images) - len(out_images)):
            task_stat.predictions.append([])

    task_stat.ground_truth = valid_out_images
    return task_stat


def config_str(x):
    if isinstance(x, partial):
        return '|'.join(map(config_str, x.args))
    if isinstance(x, tuple):
        return '(' + ','.join(map(config_str, x)) + ')'
    if isinstance(x, list):
        return '-'.join(list(map(config_str, x)))
    return x.__name__


def build_configs(args):
    """ Build multiple configurations that use (or not) various transforms and featurizers.
    """
    configs = []
    for model in [
        partial(sklearn.tree.DecisionTreeClassifier, criterion='entropy'),
        partial(sklearn.tree.DecisionTreeClassifier, criterion='gini'),
    ]:
        for ts in product([
                TIdentity,
                # TTransposeInput,
            ], [
                TIdentity,
                TSeparationRemover,
            ], [
                TIdentity,
                TCCount,
                TCCountBg,
                TCPos
            ], [
                TIdentity,
                TAugFlips,
                # TAugFlipsWRotation,
            ], [
                TIdentity,
                TAugColor,
        ]):
            transforms = partial(TSequential.make, ts)
            for featurizer in [
                    # partial(FGlobal, FConstant),
                    # partial(FGlobal, FTile),
                    # partial(FGlobal, FScale),
                    # partial(FGlobal, FSquare),
                    # partial(FGlobal, FFactorDown2),
                    # partial(FGlobal, FFactorDown1),
                    FConstant,
                    FTile,
                    FScale,
                    FSquare,
                    FFactorDown2,
                    FFactorDown1,
            ]:
                configs.append([transforms, featurizer, model])
    return configs


def parse_task(path):
    with open(path, 'r') as f:
        task = json.load(f)

    in_images = [sample['input'] for sample in task['train']
                 ] + [sample['input'] for sample in task['test']]
    out_images = [sample['output'] for sample in task['train']]
    valid_out_images = [sample['output']
                        for sample in task['test'] if 'output' in sample]

    return (task, *(list(map(np.array, i)) for i in (in_images, out_images, valid_out_images)))


def _main(args, data):
    configs = build_configs(args)

    with Pool(args.threads) as pool:
        if args.pickle_results is None or not args.pickle_results.exists():
            exclude = set()

            tasks = []
            for i, path in enumerate(data):
                task, in_images, out_images, valid_out_images = parse_task(path)

                if i not in exclude:
                    handles = [pool.apply_async(
                        process, (i, task, config, in_images, out_images,
                                  valid_out_images, np.random.randint(2 ** 32))) for config in configs]
                else:
                    handles = []

                tasks.append((task, in_images, out_images,
                              valid_out_images, handles))

            stats = [ConfigStats(c) for c in configs]
            for i, path in enumerate(data):
                if i in exclude:
                    continue

                task, in_images, out_images, valid_out_images, handles = tasks[i]

                print('Processing task:', i)

                for config, handle, stat in zip(configs, handles, stats):
                    task_stats = handle.get()
                    task_stats.task_id = path.stem
                    stat.append_task_stat(task_stats)

            for i, (config, stat) in enumerate(zip(configs, stats)):
                print()
                print('----------', i, '-----', config_str(config))
                print(stat)

            if args.pickle_results is not None:
                with args.pickle_results.open('wb') as f:
                    pickle.dump(stats, f)

        else:
            with args.pickle_results.open('rb') as f:
                stats = pickle.load(f)

        calc_confidences(args, pool, stats)

    print()
    print('SUM')
    all_stats = reduce(operator.or_, stats)
    print(all_stats)

    if args.gen_submission:
        save_output_csv(all_stats, args.save_all_predictions)


def main(args):
    dataset = sorted((args.dataset_root / args.dataset_suffix).glob('*.json'))
    if not dataset:
        raise ValueError(f'no data found in {args.dataset_root / args.dataset_suffix}')

    _main(args, dataset)


if __name__ == '__main__':
    main(parse_args())
