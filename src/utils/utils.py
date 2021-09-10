import logging
import os
import numpy as np
import torch
from torch import nn
import torchvision
from sklearn.metrics import roc_auc_score

def init_model(model_name, load_pretrained, train_from_scratch, 
        n_finetune_layers, n_labels):
    if model_name == 'resnet18':
        model_class = torchvision.models.resnet18
        layer_names = ['fc', 'layer4', 'layer3', 'layer2', 'layer1', '1'] 
        model = model_class(pretrained=load_pretrained)
    elif 'resnet50' in model_name:
        model_class = torchvision.models.resnet50
        layer_names = ['fc', 'layer4', 'layer3', 'layer2', 'layer1', '1']
        model = model_class(pretrained=load_pretrained)

    if not train_from_scratch:
        # specify which layers are frozen and which are finetuned
        finetune_layer_names = layer_names[:n_finetune_layers]
        for name, param in model.named_parameters():
            if not any([layer_name in name for layer_name in finetune_layer_names]):
                logging.info('No grad: {}'.format(name))
                param.requires_grad = False
    d = model.fc.in_features
    # new readout head
    model.fc = nn.Linear(d, n_labels)
    model.fc.requires_grad = True
    return model

def device(x, no_cuda):
    # helper function: apply .cuda() to tensor or not
    if no_cuda:
        return x
    else:
        return x.cuda()

def gong(f, args, grad):
    # gong = Grad Or No Grad. if use_grad is false, apply this function f with torch no grad
    if grad:
        return f(*args)
    else:
        with torch.no_grad():
            return f(*args)

def mean_nll(logits, y):
    return nn.functional.binary_cross_entropy_with_logits(logits, y, 
        reduction='mean')

def nll(logits, y):
    return nn.functional.binary_cross_entropy_with_logits(logits, y,
            reduction='none').sum(dim=1, keepdim=True)

def mean_accuracy(logits, y):
    return correct(logits, y).mean()

def correct(logits, y):
    # if multilabel is specified, accuracy is average of binary accuracy 
    # across labels. if not, it is multi-class classification accuracy
    preds = (logits > 0.).float()
    return ((preds - y).abs() < 1e-2).float()

def auc(logits, y):
    try:
        return torch.Tensor([roc_auc_score(y.flatten().cpu().numpy(), logits.detach().flatten().cpu().numpy())])[0]
    except ValueError as e:
        # if only one label in this minibatch, AUC defaults to 0.5
        return torch.Tensor([y.sum() * 0 + 0.5])[0]

def pretty_print(*values):
    # from https://github.com/facebookresearch/InvariantRiskMinimization
    col_width = 13
    def format_val(v):
      if not isinstance(v, str):
        v = np.array2string(v, precision=5, floatmode='fixed')
      return v.ljust(col_width)
    str_values = [format_val(v) for v in values]
    logging.info("   ".join(str_values))

def get_print_metrics(dataset):
    if dataset in ('cocostuff', 'cocostuff10k'):
        print_metrics = {
                'train': ['loss', 'nll', 'acc'],
                'valid': ['loss', 'nll', 'acc'],
                'test': ['nll', 'acc']
                }
    else:
        raise Exception('{} not a supported dataset'.format(dataset))
    return print_metrics

def logging_setup(results_dir, filemode):
    # thanks https://stackoverflow.com/questions/9321741/printing-to-screen-and-writing-to-a-file-at-the-same-time
    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%m-%d %H:%M:%S',
                        filename=os.path.join(results_dir, 'out.log'),
                        filemode=filemode)
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    # formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    formatter = logging.Formatter('%(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger().addHandler(console)
    # now use logging.info to do any logging!
    return logging

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

