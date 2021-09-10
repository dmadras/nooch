import torch
import numpy as np
import os
import json
from utils import cocostuff_loading_utils as cclu
import utils.utils as utils

def load_cocostuff_category_maps(d=None):
    if d is None:
        d = 'data'
    # a dict from category names -> category ids
    cat_map = np.load(os.path.join(d, 'category_map.npy'), allow_pickle=True).item()
    # a dict from supercategory names -> lists of category ids
    supercat_map = np.load(os.path.join(d, 'supercategory_map.npy'), allow_pickle=True).item()
    # a list of category ids corresponding to their index in 
    # the second dimension of the loaded label/area tensor
    cat_id_list = list(np.load(os.path.join(d, 'category_id_list.npy')))
    return cat_map, supercat_map, cat_id_list

def create_cocostuff_category_maps(annotations):
    # map category names to ids
    cat_map = {
        c['name'].replace(' ', '_'): c['id'] for c in annotations['categories']
    }

    # map supercategory names to ids
    inverted_supercat_map = {
        c['id']: c['supercategory'] for c in annotations['categories']
    }
    supercat_map = {
        scat: list(filter(lambda cid: 
            inverted_supercat_map[cid] == scat, 
            inverted_supercat_map.keys()))        
        for scat in set(inverted_supercat_map.values())
    }

    # get list of actually occurring id numbers in the dataset
    category_id_list = sorted(list(set(
        [c['id'] for c in annotations['categories']])))

    return cat_map, supercat_map, category_id_list

def define_labels(dataset, labels, cocodir):
    # here, we return a function which takes a tensor of all object labels,
    # and returns a tensor only containing the labels of the objects
    # specified in "labels"
    if dataset in ('cocostuff', 'cocostuff10k'):
        # load information about how the object categories are ordered in the data
        if dataset == 'cocostuff10k':
            anno_json_10k = os.path.join(cocodir, 'annotations-json/cocostuff-10k-v1.1.json')
            with open(anno_json_10k, 'r') as f:
                annotations = json.load(f)
            cat_map, supercat_map, cat_id_list = create_cocostuff_category_maps(annotations)
        else:
            cat_map, supercat_map, cat_id_list = load_cocostuff_category_maps()
        if labels == 'all' or labels == '':
            label_function = lambda x: x
            n_labels = len(cat_id_list)
        elif labels in cat_map:
            label_ind = cat_id_list.index(cat_map[labels])
            label_function = lambda x: x[:, label_ind:label_ind + 1]
            n_labels = 1
        elif labels in supercat_map:
            label_inds = sorted([cat_id_list.index(l) for l in supercat_map[labels]])
            label_function = lambda x: x[:, label_inds]
            n_labels = len(label_inds)
        else:
            labels_list = labels.split(',')
            label_inds = [cat_id_list.index(cat_map[label]) for label in labels_list]
            label_function = lambda x: x[:, label_inds]
            n_labels = len(label_inds)
    else:
        raise Exception('{} not a supported dataset'.format(dataset))
    return label_function, n_labels

def define_side_info(dataset, side_info, cocodir, no_cuda):
    if dataset in ('cocostuff', 'cocostuff10k'):
        # this function returns the areas of the objects of interest
        side_info_function_real_valued, n_side_info = define_labels(dataset, side_info, cocodir)
        # this converts it to a binary value
        side_info_function = lambda x: (side_info_function_real_valued(x) > 0).float() 
    else:
        raise Exception('{} not a supported dataset'.format(dataset))
    return side_info_function, n_side_info

def define_environments(dataset, environments='', cocodir=''):
    # here, we return a function which takes a tensor of all the object areas
    # and returns the environment for each example, where the environment
    # is a one-hot variable defined by a cross product of the categories (binarized).
    if dataset in ('cocostuff', 'cocostuff10k'):
        assert len(cocodir) > 0
        if dataset == 'cocostuff10k':
            anno_json_10k = os.path.join(cocodir, 'annotations-json/cocostuff-10k-v1.1.json')
            with open(anno_json_10k, 'r') as f:
                annotations = json.load(f)
            cat_map, supercat_map, cat_id_list = create_cocostuff_category_maps(annotations)
        else:
            cat_map, supercat_map, cat_id_list = load_cocostuff_category_maps()
        envs_list = environments.split(',')
        env_inds = [cat_id_list.index(cat_map[label]) for label in envs_list]

        place_values = np.array([2 ** i 
            for i in range(len(envs_list))]).reshape([1, -1])
        env_function = lambda x: np.sum(
                np.digitize(np.clip(x[:, env_inds], 0, 1),
                    (np.linspace(0, 1, num=2)), right=True) * place_values,
                axis=1, keepdims=True)
        n_environments = 2 ** len(env_inds)
    else:
        raise Exception('{} not a supported dataset'.format(dataset))
    return env_function, n_environments

