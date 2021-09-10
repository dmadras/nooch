import itertools
import json
import shutil
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
import math

BIG = 1e22
SMALL = 1e-40
MODEL_BEST = 'model_best.pt'
MODEL_RECENT = 'model_recent.pt'
MODEL_RECENT_TEMP = 'model_recent_temp.pt'

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

# if we want to load the best model, we are going to run it 
# forward under a new experiment
if flags.load_best_model:
    assert flags.load_checkpoint_dir != ''
# lets say that if you specify a loading checkpoint, it must the the best model
if flags.load_checkpoint_dir != '':
    assert flags.load_best_model
# might want to specify saving our checkpoint somewhere other than 
# the results dir (eg /checkpoint/username/JOBID)
if flags.save_checkpoint_dir == '':
    flags.save_checkpoint_dir = flags.results_dir

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
load_checkpoint = (flags.load_checkpoint_dir != '')
# load pretrained if flags.load_checkpoint_dir is empty
load_pretrained = not load_checkpoint and not flags.train_from_scratch 
model = utils.init_model(flags.model, load_pretrained, flags.train_from_scratch, 
        flags.n_finetune_layers, n_labels)
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

### CHECKPOINTING STARTS ###

# the path where we will save our best model so far
best_model_ckpt_path = os.path.join(flags.save_checkpoint_dir, MODEL_BEST)

# this next part is written so that this code can be run on a system where 
# the job may be interrupted arbitrarily, such that the same command
# can be rerun and pick up from the end of the last completed epoch. These
# are due to particularities of an internal cluster

# the path where we will save our most recent model
recent_model_ckpt_path = os.path.join(flags.save_checkpoint_dir, MODEL_RECENT)
# the path where we will save our last recent model, in case we are
# interrupted in the middle of checkpointing
recent_model_temp_ckpt_path = os.path.join(flags.save_checkpoint_dir, MODEL_RECENT_TEMP)

# we now ensure that the same command can train from scratch if nothing 
# is there, and loads from a specified checkpoint directory if it is there
load_checkpoint = False
if flags.load_recent_model:
    load_checkpoint = True
    # need to see if model exists. if load_checkpoint_dir is not specified, 
    # we check save_checkpoint_dir, then we check the main results dir
    # we'll assume that if we're loading from a specified load_checkpoint_dir, we want the best model rather than the recent one
    if flags.save_checkpoint_dir != '':
        recent_model_dir = flags.save_checkpoint_dir
    else:
        recent_model_dir = flags.results_dir

    # the most recent model will be either at model_recent or model_recent_temp
    model_path_recent = os.path.join(recent_model_dir, MODEL_RECENT)
    model_path_recent_temp = os.path.join(recent_model_dir, MODEL_RECENT_TEMP)
    if os.path.exists(model_path_recent):
        model_path = model_path_recent
        logging.info('Loading recent model path: {}.'.format(model_path))
    elif os.path.exists(model_path_recent_temp):
        model_path = model_path_recent_temp
        logging.info('Loading recent model path: {}.'.format(model_path))
    else:
        # if neither, we start a new model
        logging.info('Could not find either recent model path: {} or {}. Starting a new model'.format(
            model_path_recent, model_path_recent_temp))
        # in this case we are not loading a checkpoint
        load_checkpoint = False

if flags.load_best_model and not load_checkpoint:
    # kidding, we're not starting a new model! reloading a previously trained checkpoint
    load_checkpoint = True
    # load the best model so far from model.pt
    model_path = os.path.join(flags.load_checkpoint_dir, MODEL_BEST)
    logging.info('Loading pre-trained best model path: {}.'.format(model_path))
    # in this case, we are re-training but starting from new weights,
    # so we only want the model, not the optimizer or random state
    load_model_only = True

if load_checkpoint:
    starting_new_model = False
    try:
        # Overwrite defaults
        checkpoint = torch.load(model_path)
    except RuntimeError as e:
        logging.info('Couldnt load model from {}'.format(model_path))
        if load_model_only:
            # then nothing we can do, we need to go rerun that model
            logging.info('The previous best model were loading from is corrupted, need to rerun that')
            raise e
        else:
            # then were just trying to load a recent checkpoint
            if MODEL_RECENT in model_path:
                logging.info('Previous recent checkpoint was corrupted, trying recent_temp')
                if os.path.exists(model_path_recent_temp):
                    try:
                        checkpoint = torch.load(model_path_recent_temp)
                    except RuntimeError as e:
                        logging.info('Previous recent_temp checkpoint was also corrupted, starting a new model')
                        starting_new_model = True
                else:
                        logging.info('Previous recent_temp checkpoint did not exist, starting a new model')
                        starting_new_model = True
            else:
                logging.info('Previous recent checkpoints were corrupted, starting a new model')
                starting_new_model = True

    if not starting_new_model:
        model.load_state_dict(checkpoint['model_state_dict'])
        if not load_model_only:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_epoch_results = checkpoint['best_epoch_results']
            best_epoch = checkpoint['best_epoch']
            torch_random_state = checkpoint['torch_random_state']
            torch.random.set_rng_state(torch_random_state)
            torch_cuda_random_state = checkpoint['torch_cuda_random_state']
            torch.cuda.random.set_rng_state(torch_cuda_random_state)
            np_random_state = checkpoint['np_random_state']
            np.random.set_state(np_random_state)

### CHECKPOINTING DONE ###

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

        # the metrics we will print ever flags.print_freq minibatches
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

                if flags.method == 'cvar':
                    sorted_loss_vec, _ = torch.sort(loss_vec.flatten(), descending=True)
                    n_values_below_cutoff = math.ceil(len(sorted_loss_vec) * flags.cvar_risk)
                    loss = sorted_loss_vec[:n_values_below_cutoff].mean()
                
                # loss is the scalar we backprop through
                batch_results['loss'].append(loss)

            curr_minibatch += 1
            print_args = [loader.name, 
                    'Minibatch {:d} / {:d}'.format(curr_minibatch, len(loader))] + \
                            [torch.Tensor(batch_results[m])[-1].detach().cpu().numpy() 
                            # [torch.mean(torch.Tensor(batch_results[m])).detach().cpu().numpy() 
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
    # if we've specified we want to save checkpoints from every n epochs
    if flags.checkpoint_every > 0 and epoch % flags.checkpoint_every == 0:
        ckpt_path_this_epoch = os.path.join(flags.save_checkpoint_dir, 
                'model_checkpoint_epoch_{:d}.pt'.format(epoch))
        torch.save(ckpt_dict, ckpt_path_this_epoch)
        logging.info('Checkpointed current model to {}'.format(ckpt_path_this_epoch))
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

    ### CHECKPOINTING STARTS ### 
    # if we are saving the model at each epoch for interruptability purposes
    if flags.save_recent_model:
        if os.path.exists(recent_model_ckpt_path):
            # copy recent checkpoint in case we are interrupted while checkpointing
            shutil.copyfile(recent_model_ckpt_path, recent_model_temp_ckpt_path)
            logging.info('Copied last recent model to {}'.format(recent_model_temp_ckpt_path))
        # save the most recent model
        torch.save(ckpt_dict, recent_model_ckpt_path)
        logging.info('Checkpointed most recent model to {}'.format(recent_model_ckpt_path))
        # then delete the temporarily copied previous checkpoint
        if os.path.exists(recent_model_temp_ckpt_path):
            os.remove(recent_model_temp_ckpt_path)
            logging.info('Removed second-most recent model from {}'.format(recent_model_temp_ckpt_path))
    ### CHECKPOINTING DONE ### 

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

# if we said the checkpoint directory is different from the results directory
# and we didn't specify *not* to checkpoint to the results directory ...
# copy the best model over to the results directory
if flags.save_checkpoint_dir != flags.results_dir and not flags.dont_checkpoint:
    new_best_model_checkpoint_path = os.path.join(flags.results_dir, MODEL_BEST)
    shutil.copyfile(best_model_ckpt_path, new_best_model_checkpoint_path)

# say we're done
donefile = os.path.join(flags.results_dir, 'done.txt')
with open(donefile, 'w') as f:
    f.write('Done!\n')

