import itertools
import json
import os
import numpy as np
import torch
from torch import nn, optim
import argparse_run_model as argparser
import attr_functions as af
import load_data
import utils.irm_utils as iu
import utils.utils as utils
import random

BIG = 1e22
SMALL = 1e-40
MODEL_BEST = 'model_best.pt'

flags = argparser.parse_args() 

# set seeds
random.seed(flags.seed)
np.random.seed(flags.seed)
torch.manual_seed(flags.seed)
if not flags.no_cuda:
    torch.cuda.manual_seed(flags.seed)
torch.backends.cudnn.deterministic = True

# make experiment directory
if not os.path.exists(flags.results_dir):
    os.makedirs(flags.results_dir)

# set up logging
logging = utils.logging_setup(flags.results_dir,
        filemode='a' if flags.load_recent_model 
        or flags.load_best_model else 'w')
logging.info('Flags:')
for k,v in sorted(vars(flags).items()):
  logging.info("\t{}: {}".format(k, v))

# write input args to file
argsfile = os.path.join(flags.results_dir, 'args.json')
with open(argsfile, 'w') as f:
    f.write(json.dumps(vars(flags), indent=4, sort_keys=True))
logging.info('Wrote args to {}.'.format(argsfile))

# define label(s) and side information, ie areas of each object class
label_function, n_labels = af.define_labels(
        flags.dataset, flags.labels, flags.datadir)
# if we're using a method that requires environments, make them
if flags.make_environments:
    environment_function, n_environments = af.define_environments(
            flags.dataset, flags.environments, flags.datadir)

# load data
train_loader, valid_loader, test_loader = load_data.load_data(
        flags, label_function, environment_function=None 
        if not flags.make_environments else environment_function,
        )
logging.info('Data loading complete')

# Define and instantiate the model 
model = utils.init_model(flags.model, not flags.train_from_scratch, 
        flags.train_from_scratch, flags.n_finetune_layers, n_labels)
model = utils.device(model, flags.no_cuda)
parameters = model.parameters()
logging.info('Number of model parameter blocks: {:d}'
        .format(len(list(model.parameters()))))

# Define optimizer
if flags.optimizer == 'SGD':
    optimizer_class = optim.SGD
    optimizer_kwargs = {'lr': flags.lr, 'momentum': 0.9, 
            'weight_decay': flags.l2_regularizer_weight}
elif flags.optimizer == 'Adam':
    optimizer_class = optim.Adam
    optimizer_kwargs = {'lr': flags.lr, 
            'weight_decay': flags.l2_regularizer_weight}
optimizer = optimizer_class(parameters, **optimizer_kwargs)

# set the metric that we're doing early stopping on
criterion = flags.val_criterion
comparison = lambda new, old: (new > old if 'acc' in criterion else new < old)

# set defaults, may be overwritten by checkpoint
best_epoch_results = {'valid': {criterion: BIG if 'acc' not in criterion else -BIG}}
start_epoch = 0
best_epoch = 0

# the path where we will save our best model so far
best_model_ckpt_path = os.path.join(flags.results_dir, MODEL_BEST)

# the metrics we will print at the end of each epoch
print_metrics = utils.get_print_metrics(flags.dataset)
# some string manipulation for printing purposes
print_names = list(itertools.chain(
    *[['{} {}'.format(phase, metric) for metric in print_metrics[phase]] 
        for phase in ['train', 'valid', 'test' ]]))

# now we train
for epoch in range(start_epoch, flags.max_epochs):
    utils.pretty_print(*(['Epoch'] + print_names))
    epoch_results = {'train': {}, 'valid': {}, 'test': {}}
    for loader in [train_loader, valid_loader, test_loader]:
        if loader.name == 'test':
            # only look at test data if we are at our best val loss so far
            if not best_so_far:
                logging.info('Skipping test because validation loss was not best so far')
                break
        logging.info('Starting {}'.format(loader.name))
        
        if loader.name != 'train':
            model.eval()
        else:
            model.train()

        # the metrics we will print every flags.print_freq minibatches
        print_metrics_minibatch = {
                'train': ['loss', 'nll', 'acc', 'auc'],
                'valid': ['loss', 'nll', 'acc', 'auc'], 
                'test': ['nll', 'acc', 'auc']}
        # other things we need to track through a minibatch
        batch_trackers = ['loss', 'nll', 'acc', 'auc', 
            'loss_vec', 'logits', 'labels', 'attrs', 'n', 'other_info']
        if flags.method == 'irm':
            print_metrics_minibatch['train'].append('penalty')
            print_metrics_minibatch['valid'].append('penalty')
            batch_trackers.append('penalty')
        batch_results = {m: list() for m in batch_trackers}
        curr_minibatch = 0
        
        utils.pretty_print(*[loader.name, 'Minibatch nn / NNN'] + print_metrics_minibatch[loader.name])

        # side_info is areas of objects, other_info is image_ids, and
        # possibly environments as well
        for images, labels, side_info, other_info in loader:
            # if we want to get through epochs quickly for debugging purposes
            if flags.debug:
                if loader.name == 'train' or not flags.debug_train_only:
                    if curr_minibatch > len(loader) / flags.debug_pct: # for now
                        break

            optimizer.zero_grad()
            # forward pass of the model
            logits = utils.gong(lambda x: model(x), [images], loader.name == 'train')
            loss_vec = utils.nll(logits, labels.float())
            nll = torch.mean(loss_vec)
            acc = utils.mean_accuracy(logits, labels)
            auc = utils.device(utils.auc(logits, labels), flags.no_cuda)

            batch_results['nll'].append(nll)
            batch_results['acc'].append(acc)
            batch_results['auc'].append(auc)
            batch_results['logits'].append(logits)
            batch_results['labels'].append(labels)
            batch_results['attrs'].append(side_info)
            batch_results['n'].append(len(images))
            batch_results['other_info'].append(other_info)

            if flags.make_environments:
                envs = other_info[:, 0]
                # find the environments that are present in this minibatch
                unique_envs = torch.unique(envs)

            if flags.focal_loss:
                # calculate focal loss TODO change this pending experimental results?
                probs = torch.sigmoid(logits)
                p_t = probs * labels + (1 - probs) * (1 - labels)
                focal_loss_coeffs = 1 - loader.class_probs[labels.long()]
                focal_loss_vec = (-(focal_loss_coeffs * ((1 - p_t) ** flags.focal_eta))
                        * torch.log(torch.clip(p_t, SMALL, 1)))
                loss_vec = focal_loss_vec
                loss = torch.mean(loss_vec)

            if flags.reweighted_erm:
                probs = torch.sigmoid(logits)
                p_t = probs * labels + (1 - probs) * (1 - labels)
                if flags.reweight_type == 'class':
                    reweight_loss_coeffs = 1 - loader.class_probs[labels.long()]
                elif flags.reweight_type == 'envs':
                    reweight_loss_coeffs = 1 - loader.env_probs[envs.reshape([-1, 1]).long()]
                reweight_loss_vec = (-(reweight_loss_coeffs ** flags.reweight_exponent)
                        * torch.log(torch.clip(p_t, 1e-40, 1)))
                loss_vec = reweight_loss_vec
                loss = torch.mean(loss_vec)

            # loss_vec is a vector of losses, 1 for each minibatch example
            batch_results['loss_vec'].append(loss_vec)

            if loader.name != 'test':
                if flags.method == 'erm':
                    loss = loss_vec.mean()

                if flags.method == 'irm':
                    env_loss_total = nll * 0
                    env_penalty_total = nll * 0
                    for env in unique_envs:
                        logits_env = logits[envs == env]
                        labels_env = labels[envs == env]
                        env_nll, env_penalty = iu.irm_penalize(logits_env, 
                                labels_env.float(), flags.no_cuda)
                        env_loss_total = env_loss_total + env_nll
                        env_penalty_total = env_penalty_total + env_penalty
                    loss = env_loss_total + flags.irm_penalty_coefficient * env_penalty_total
                    batch_results['penalty'].append(env_penalty_total)
                    
                    if flags.irm_penalty_coefficient > 1:
                        loss = loss / flags.irm_penalty_coefficient

                if flags.method == 'gdro':
                    max_env_loss = -1
                    for env in unique_envs:
                        env_loss = (loss_vec[envs == env].mean() + 
                                (flags.gdro_adjustment / np.sqrt(loader.environment_counter[int(env)])))
                        if env_loss > max_env_loss:
                            max_env_loss = env_loss
                    loss = max_env_loss
                
                # loss is the scalar we backprop through
                batch_results['loss'].append(loss)

            curr_minibatch += 1
            print_args = [loader.name, 
                    'Minibatch {:d} / {:d}'.format(curr_minibatch, len(loader))] + \
                            [torch.Tensor(batch_results[m])[-1].detach().cpu().numpy() 
                                    for m in print_metrics_minibatch[loader.name]]
            if curr_minibatch % flags.print_freq == 0:
                utils.pretty_print(*print_args)

            if loader.name == 'train':
                loss.backward()
                optimizer.step()

        # aggregate end of epoch metrics
        epoch_vecs = {m: torch.cat(batch_results[m]) for m in ['loss_vec', 'logits', 'labels', 'attrs', 'other_info']}
        epoch_vecs['n'] = utils.device(torch.Tensor(batch_results['n']), flags.no_cuda)
        total_n = sum(batch_results['n'])
        
        # these are the metrics we will save in the results dir
        metrics_save = {'test': ['nll', 'acc', 'logits', 'labels', 'attrs', 'other_info'], 
                'train': ['nll', 'acc', 'logits', 'labels', 'attrs', 'other_info', 'loss'],
                'valid': ['nll', 'acc', 'logits', 'labels', 'attrs', 'other_info', 'loss']}
        for m in metrics_save[loader.name]:
            if len(batch_results[m][0].shape) == 2: # want to concatenate vectors (e.g. labels)
                epoch_results[loader.name][m] = torch.cat(batch_results[m])
            else: # want to average metrics (e.g. accuracy or nll)
                epoch_results[loader.name][m] = torch.sum(torch.stack(batch_results[m]) * epoch_vecs['n']) / total_n

        # check if our validation metric is the best one so far
        if loader.name == 'valid':
            best_so_far = comparison(epoch_results['valid'][criterion], best_epoch_results['valid'][criterion])

    if best_so_far:
        phases = ['train', 'valid', 'test']
    else:
        phases = ['train', 'valid']
    print_data = ['Epoch {:d}'.format(epoch)] + list(itertools.chain(
            *[[epoch_results[phase][met].detach().cpu().numpy() for met in print_metrics[phase]]
                for phase in phases]))
    # print our end of epoch results
    utils.pretty_print(*print_data)

    # Do a bunch of checkpointing
    ckpt_dict = {
            'epoch': epoch,
            'best_epoch': best_epoch,
            'best_epoch_results': best_epoch_results,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'torch_random_state': torch.random.get_rng_state(),
            'torch_cuda_random_state': torch.cuda.random.get_rng_state(),
            'np_random_state': np.random.get_state()
            }
    # if this is our best model so far by val criterion
    if best_so_far:
        logging.info('Achieved best validation {} so far: {:3f}'.format(criterion,
            epoch_results['valid'][criterion].detach().cpu().numpy()))
        best_epoch = epoch
        best_epoch_results = epoch_results
        if not flags.dont_checkpoint:
            torch.save(ckpt_dict, best_model_ckpt_path)
            logging.info('Checkpointed best model so far to {}'.format(best_model_ckpt_path))
        else:
            logging.info('flags.dont_checkpoint=False, skipping checkpointing')
    ckpt_dict['best_epoch'] = best_epoch
    ckpt_dict['best_epoch_results'] = best_epoch_results

    if epoch - best_epoch > flags.patience:
        logging.info('Stopping optimization at epoch {:d} with val loss/acc: {:.3f}/{:.3f}, test nll/acc: {:.3f}/{:.3f}'.format(
            best_epoch, best_epoch_results['valid']['loss'], best_epoch_results['valid']['acc'],
            best_epoch_results['test']['nll'], best_epoch_results['test']['acc']))
        break

for phase in ['train', 'valid', 'test']:
    logging.info('Final {} acc'.format(phase))
    logging.info('{:.3f}'.format(best_epoch_results[phase]['acc']))

# Do results saving
savedict = {}
for split in best_epoch_results:
    for met in best_epoch_results[split]:
        savedict['{}_{}'.format(split, met)] = best_epoch_results[split][met].detach().cpu().numpy()

for k, v in savedict.items():
    savepath = os.path.join(flags.results_dir, '{}.npy'.format(k))
    np.save(savepath, v)
logging.info('Saved results to {}'.format(flags.results_dir))

# say we're done
donefile = os.path.join(flags.results_dir, 'done.txt')
with open(donefile, 'w') as f:
    f.write('Done!\n')

