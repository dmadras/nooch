import os
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch import nn


def evaluate(task, labels, logits, image_ids, metric, subgroup, criterion, phase='test'):

    assert metric in ('auc', 'nll', 'err')
    assert subgroup in ('all', 'hard', 'easy', 'hard-positive', 'hard-negative')
    assert criterion in ('CE', 'gist')

    if metric == 'auc':
        assert not subgroup in ('hard-positive', 'hard-negative')

    labels = labels.flatten()
    logits = logits.flatten()
    image_ids = image_ids.flatten()

    # criterion - CE or gist 
    hard_positive_ids = np.load(os.path.join('nooch', 
        'nooch_ids_{}_{}_hard_positive_{}.npy'.format(criterion, task, phase)))
    hard_negative_ids = np.load(os.path.join('nooch', 
        'nooch_ids_{}_{}_hard_negative_{}.npy'.format(criterion, task, phase)))
    hard_positive_mask = np.isin(image_ids, hard_positive_ids)
    hard_negative_mask = np.isin(image_ids, hard_negative_ids)

    # subgroup - get mask
    if subgroup == 'all':
        mask = np.ones_like(labels) > 0
    elif subgroup == 'hard-positive':
        mask = hard_positive_mask
    elif subgroup == 'hard-negative':
        mask = hard_negative_mask
    elif subgroup == 'hard':
        mask = hard_negative_mask | hard_positive_mask
    elif subgroup == 'easy':
        mask = ~(hard_negative_mask | hard_positive_mask)

    # metric - get function
    if metric == 'auc':
        f = auc
    elif metric == 'nll':
        f = nll
    elif metric == 'err':
        f = err

    result = f(labels[mask], logits[mask])
    return result

def auc(labels, logits):
    return roc_auc_score(labels, logits)

def nll(labels, logits):
    return nn.functional.binary_cross_entropy_with_logits(
        torch.Tensor(logits), torch.Tensor(labels), 
        reduction='mean').numpy()

def err(labels, logits):
    return ((labels > 0.5) != (logits > 0)).mean()


if __name__ == '__main__':
    expdir_erm = '/scratch/gobi2/madras/struct-robust/results/Apr13_train_cocostuff_labels_sgd_vaughan'
    expname_template_erm = 'Apr13_train_cocostuff_labels_sgd_vaughan_label{}_seed{:d}_nfl6'
    thing = 'surfboard'
    thing = thing.replace('-', '_')
    print(thing)
    for metric in ['auc', 'nll', 'err']:
        if metric == 'auc':
            subgroups = ['hard', 'easy']
        else:
            subgroups = ['hard-positive', 'hard-negative', 'easy']
        for subgroup in subgroups:
            for crit in ['CE', 'gist']:
                ttl = 0
                seeds = [11, 12, 13]
                for seed in seeds:
                    labels = np.load(os.path.join(expdir_erm, expname_template_erm.format(thing, seed), 'test_labels.npy'))
                    logits = np.load(os.path.join(expdir_erm, expname_template_erm.format(thing, seed), 'test_logits.npy'))
                    image_ids = np.load(os.path.join(expdir_erm, 
                        expname_template_erm.format(thing, seed), 'test_other_info.npy'))[:, -1]
                    val = evaluate(thing, labels, logits, image_ids, metric, subgroup, crit)
                    ttl += val
                print('{}, {}, {}: {:.3f}'.format(crit, subgroup, metric, ttl / len(seeds)))

