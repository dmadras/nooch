import os
import numpy as np
import utils.cocostuff_loading_utils as cslu

def load_data(
        flags, 
        label_function=None, 
        environment_function=None):
    if flags.dataset in ('cocostuff', 'cocostuff10k'):
        train_loader, valid_loader, test_loader = cslu.prepare_cocostuff_minibatching(
            datadir=flags.datadir, 
            batch_size=flags.batch_size, 
            label_function=label_function,
            no_cuda=flags.no_cuda,
            split_npz=flags.coco_split_npz,
            use_10k='10k' in flags.dataset,
            environment_function=environment_function,
            weighted_sampler=flags.weighted_sampler,
            reweight_type=flags.reweight_type
            )
    else:
        raise Exception('{} is not a supported dataset'.format(flags.dataset))
    return train_loader, valid_loader, test_loader

