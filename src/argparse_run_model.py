import argparse

def parse_args():

    parser = argparse.ArgumentParser(description='Run a model!')

    # major pieces of the learning problem
    parser.add_argument('--model', type=str, default='resnet50',
            help='resnet50 or resnet18?')
    parser.add_argument('--optimizer', type=str, default='SGD',
            help='SGD or Adam?')
    parser.add_argument('--labels', type=str, default='all',
            help='comma-separated list of COCO-Stuff categories, or "all"')
    parser.add_argument('--dataset', type=str, default='cocostuff',
            help='At the moment, this has to be cocostuff')

    # basic training hyperparameters
    parser.add_argument('--seed', type=int, default=0, help='reproducibility')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--l2_regularizer_weight', type=float,default=0.0001,
            help='l2 weight decay parameter')
    parser.add_argument('--patience', type=int, default=2, 
            help='number of epochs without validation loss improvement '
            'before early stopping kicks in')
    parser.add_argument('--batch_size', type=int, default=32,
            help='size of minibatch')
    parser.add_argument('--max_epochs', type=int, default=500,
            help='maximum number of epochs to train for')
    parser.add_argument('--train_from_scratch', action='store_true',
            help='if specified, do not load pretrained weights')
    parser.add_argument('--n_finetune_layers', type=int, default=6,
            help='number of resnet parameter blocks to *not* freeze, from 1-6')
    parser.add_argument('--val_criterion', type=str, default='loss',
            help='loss or acc, what should the early stopping criterion be?')
    parser.add_argument('--no_cuda', action='store_true',
            help='if specified, use cpu rather than gpu')
    parser.add_argument('--n_channels', type=int, default=3,
            help='number of image channels in data - 3 for color images')
    parser.add_argument('--input_dim', type=int, default=321,
            help='length of each side of image after resizing')

    # Cocostuff args
    parser.add_argument('--datadir', type=str, default='cocostuff', 
            help='where COCOStuff dataset is stored')
    parser.add_argument('--coco_split_npz', type=str, 
            default='data/cocostuff_split_image_ids.npz',
            help='where dataset splits specification is stored')

    # baselines
    parser.add_argument('--method', type=str, default='erm',
            help='what learning method to use: erm, gdro, irm, focal')
    parser.add_argument('--make_environments', action='store_true',
            help='if specified, calculated environment variables')
    parser.add_argument('--environments', type=str, default='',
            help='if make_environments is specified, a comma-separated list'
            ' of object names to take a cross-product over to define'
            ' environments')
    parser.add_argument('--irm_penalty_coefficient', type=float, default=10000,
            help='coefficient on the IRM gradient penalty term')
    parser.add_argument('--gdro_adjustment', type=float, default=1,
            help='constant on the adjustment term in the GDRO loss')
    parser.add_argument('--focal_loss', action='store_true',
            help='if specified, calculate focal loss')
    parser.add_argument('--focal_eta', type=float, default=1,
            help='focal loss hyperparamter')
    parser.add_argument('--weighted_sampler', type=float, default=0,
            help='parameter for undersampling')
    parser.add_argument('--reweighted_erm', action='store_true', 
            help='run ERM with loss reweighting')
    parser.add_argument('--reweight_exponent', type=float, default=0,
            help='parameter for ERM loss reweighting')
    parser.add_argument('--reweight_type', type=str, default='class',
            help='weight ERM by class or environments ("class", "envs")')

    # results saving and debugging
    parser.add_argument('--results_dir', type=str, 
            default='/scratch/gobi1/madras/struct-robust/results/temp',
            help='directory to save all results to')
    parser.add_argument('--debug', action='store_true', 
            help='if specified, just take a fraction of minibatches each epoch')
    parser.add_argument('--debug_pct', type=int, default=100,
            help='if debug specified, fraction of minibatches to take')
    parser.add_argument('--debug_train_only', action='store_true',
            help='if debug specified, and this is specified, look at'
            ' full valid/test sets but only a fraction of training set')
    parser.add_argument('--print_freq', type=int, default=10,
            help='print metrics every print_freq minibatches')

    # checkpointing
    parser.add_argument('--dont_checkpoint', action='store_true',
            help='if specified, dont checkpoint when model reaches best'
            ' validation metrics')
    parser.add_argument('--checkpoint_every', type=int, default=-1,
            help='if we want to save checkpoints every n epochs into different'
            ' files, this is the value of n')

    # advanced checkpointing - you can mostly ignore this, allows for
    # interruptability and automatic restarting. Only works for
    # run_model_interruptable.py.

    # To run interruptable command, need --load_recent_model --save_recent_model and
    #   may also want to specify save_checkpoint_dir, otherwise it defaults to results_dir
    # If you want to start training from a previous model other than the
    #   torch default, specify --load_checkpoint_dir
    # If you do not want to checkpoint the best model so far in the results_dir, use --dont_checkpoint;
    #   this will still checkpoint it to save_checkpoint_dir though (which defaults to results_dir if not specified)
    parser.add_argument('--load_checkpoint_dir', type=str, default='',
            help='if we want to start training from a previously trained model,'
            ' what directory is that model stored in?')
    parser.add_argument('--save_checkpoint_dir', type=str, default='',
            help='what directory to checkpoint to? if not specified, defaults'
            ' to results_dir')
    parser.add_argument('--load_best_model', action='store_true',
            help='if we want to start training from a previous best model.'
            ' if specified, load_checkpoint_dir must also be specified')
    parser.add_argument('--load_recent_model', action='store_true',
            help='if we want to start training from a previous most recent model')
    parser.add_argument('--save_recent_model', action='store_true',
            help='if we want to save the most recent model at each epoch,'
            ' overwriting the same checkpoint each time')
    flags = parser.parse_args()

    if flags.method in ('irm', 'gdro'):
        assert flags.make_environments

    return flags
