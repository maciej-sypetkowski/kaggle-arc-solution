import math
import pickle

import pandas as pd
import sklearn

from transform import TAugColor, TAugFlips, TAugFlipsWRotation


def featurize_task_stat(task_stat):

    def used_unique_features(i):
        if i < 0:
            return set()

        ret = set()
        ret = ret.union(used_unique_features(
            task_stat.model.tree_.children_right[i]))
        ret = ret.union(used_unique_features(
            task_stat.model.tree_.children_left[i]))
        ret.add(task_stat.model.tree_.feature[i])
        return ret

    rows = []
    assert len(task_stat.correct()) == len(task_stat.predictions)

    tr_mappings_count = 1
    for level, tr in enumerate(task_stat.transformer.unsequence()):
        if isinstance(tr, TAugColor):
            tr_mappings_count = len(tr.mappings)

    # add to features binary information about using selected transforms
    types_to_featurize = [TAugFlips, TAugFlipsWRotation]
    all_transformers = set(map(type, task_stat.transformer.unsequence()))
    tfeatures = [t in all_transformers for t in types_to_featurize]
    tfeatures_columns_names = list(t.__name__ for t in types_to_featurize)

    for i, correct in enumerate(task_stat.correct()):
        rows.append(dict(
            correct=correct,  # label
            used_unique_features_count=math.log(len(used_unique_features(0))),
            nodes_count=math.log(task_stat.model.tree_.node_count),
            deductible=task_stat.deducible,
            tr_mappings_count=math.log(tr_mappings_count),
            **{k: v for k, v in zip(tfeatures_columns_names, tfeatures)}
        ))

    df = pd.DataFrame(rows)
    return df.reindex(sorted(df.columns), axis=1)


def iterate_stats(stats):
    for config_stat in stats:
        for task_stat in config_stat:
            if not task_stat.error:
                yield task_stat


def calc_confidences(args, pool, stats):
    handles = []
    for task_stat in iterate_stats(stats):
        handles.append(pool.apply_async(
            featurize_task_stat, args=(task_stat,)))
    if not len(handles):
        return
    df = pd.concat([h.get() for h in handles], ignore_index=True)
    df, labels = df[df.columns.difference(['correct'])], df['correct']

    if args.train_confidence is not None:
        print(df.head())
        print(df.mean(axis=0))

        model = sklearn.linear_model.LogisticRegression()
        model.fit(df, labels)

        with args.train_confidence.open('wb') as f:
            pickle.dump(model, f)

        # convenient to copy and paste into a notebook for kaggle submission
        print('Pickled confidence model (hex):')
        print(pickle.dumps(model).hex())
    else:
        with args.load_confidence.open('rb') as f:
            model = pickle.load(f)

    print("Linear model coefficients", model.coef_)

    confidence_iter = iter(model.predict_proba(df))
    for task_stat in iterate_stats(stats):
        for preds_per_test_img in task_stat.predictions:
            confidence = next(confidence_iter)[1]
            for i, prediction in enumerate(preds_per_test_img):
                preds_per_test_img[i] = (prediction[0], confidence)
    try:
        next(confidence_iter)
        assert 0
    except StopIteration:
        pass
