import torch
import torch.utils.data
import torch.nn as nn
from torch import optim
from torch.nn import functional as F

import numpy as np
import pandas as pd
import os
import time
import random
import importlib
from shutil import copyfile
from functools import partial
import argparse

from util import *
from util.format_data import CathData

# python dave_understanding.py --model SE3ResNet34Small --data-filename cath_3class_ca.npz --training-epochs 100 --batch-size 8 --batchsize-multiplier 1 --kernel-size 3 --initial_lr=0.0001 --lr_decay_base=.996 --p-drop-conv 0.1 --downsample-by-pooling --restore-checkpoint-filename init_100_epoch.ckpt

def execute(checkpoint):
    ### Load model
    model = network_module.network(n_input=n_input, n_output=n_output, args=args)
    for block in model.blocks:
        pass
        #print(block)

    # This line prints something about the kernel weights in the first outer block
    #print(model.blocks[0].layers[0].layers[0].conv._modules['kernel']._parameters['weight'])

    # model.blocks[5].register_forward_hook(cnn_hook)
    # model.blocks[6].layers[1].register_forward_hook(latent_mean_hook)
    # model.blocks[6].layers[2].register_forward_hook(latent_var_hook)
    # if use_gpu:
    #     model.cuda()





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # required
    parser.add_argument("--model", required=True,
                        help="Which model definition to use")
    parser.add_argument("--data-filename", required=True,
                        help="The name of the data file, e.g. cath_3class.npz, cath_10arch.npz (will automatically downloaded if not found)")
    parser.add_argument("--report-frequency", default=1, type=int,
                        help="The frequency with which status reports will be written")
    parser.add_argument("--report-on-test-set", action="store_true", default=False,
                        help="Whether to include accuracy on test set in output")
    parser.add_argument("--burnin-epochs", default=0, type=int,
                        help="Number of epochs to discard when dumping the best model")
    # cath specific
    parser.add_argument("--data-discretization-bins", type=int, default=50,
                        help="Number of bins used in each dimension for the discretization of the input data")
    parser.add_argument("--data-discretization-bin-size", type=float, default=2.0,
                        help="Size of bins used in each dimension for the discretization of the input data")

    parser.add_argument("--mode", choices=['train', 'test', 'validate'], default="train",
                        help="Mode of operation (default: %(default)s)")
    parser.add_argument("--training-epochs", default=100, type=int,
                        help="Which model definition to use")
    parser.add_argument("--randomize-orientation", action="store_true", default=False,
                        help="Whether to randomize the orientation of the structural input during training (default: %(default)s)")
    parser.add_argument("--batch-size", default=18, type=int,
                        help="Size of mini batches to use per iteration, can be accumulated via argument batchsize_multiplier (default: %(default)s)")
    parser.add_argument("--batchsize-multiplier", default=1, type=int,
                        help="number of minibatch iterations accumulated before applying the update step, effectively multiplying batchsize (default: %(default)s)")
    parser.add_argument("--restore-checkpoint-filename", type=str, default=None,
                        help="Read model from checkpoint given by filename (assumed to be in checkpoint folder)")
    parser.add_argument("--initial_lr", default=1e-1, type=float,
                        help="Initial learning rate (without decay)")
    parser.add_argument("--lr_decay_start", type=int, default=40,
                        help="epoch after which the exponential learning rate decay starts")
    parser.add_argument("--lr_decay_base", type=float, default=.94,
                        help="exponential decay factor per epoch")
    # model
    parser.add_argument("--kernel-size", type=int, default=3,
                        help="convolution kernel size")
    parser.add_argument("--p-drop-conv", type=float, default=None,
                        help="convolution/capsule dropout probability")
    parser.add_argument("--p-drop-fully", type=float, default=None,
                        help="fully connected layer dropout probability")
    parser.add_argument("--bandlimit-mode", choices={"conservative", "compromise", "sfcnn"}, default="compromise",
                        help="bandlimiting heuristic for spherical harmonics")
    parser.add_argument("--SE3-nonlinearity", choices={"gated", "norm"}, default="gated",
                        help="Which nonlinearity to use for non-scalar capsules")
    parser.add_argument("--normalization", choices={'batch', 'group', 'instance', None}, default='batch',
                        help="Which nonlinearity to use for non-scalar capsules")
    parser.add_argument("--downsample-by-pooling", action='store_true', default=True,
                        help="Switches from downsampling by striding to downsampling by pooling")
    # WEIGHTS
    parser.add_argument("--lamb_conv_weight_L1", default=0, type=float,
                        help="L1 regularization factor for convolution weights")
    parser.add_argument("--lamb_conv_weight_L2", default=0, type=float,
                        help="L2 regularization factor for convolution weights")
    parser.add_argument("--lamb_bn_weight_L1", default=0, type=float,
                        help="L1 regularization factor for batchnorm weights")
    parser.add_argument("--lamb_bn_weight_L2", default=0, type=float,
                        help="L2 regularization factor for batchnorm weights")
    parser.add_argument("--lamb_linear_weight_L1", default=0, type=float,
                        help="L1 regularization factor for fully connected layer weights (except last / classification layer)")
    parser.add_argument("--lamb_linear_weight_L2", default=0, type=float,
                        help="L2 regularization factor for fully connected layer weights (except last / classification layer)")
    parser.add_argument("--lamb_softmax_weight_L1", default=0, type=float,
                        help="L1 regularization factor for classification layer weights")
    parser.add_argument("--lamb_softmax_weight_L2", default=0, type=float,
                        help="L2 regularization factor for classification layer weights")
    # BIASES
    parser.add_argument("--lamb_conv_bias_L1", default=0, type=float,
                        help="L1 regularization factor for convolution biases")
    parser.add_argument("--lamb_conv_bias_L2", default=0, type=float,
                        help="L2 regularization factor for convolution biases")
    parser.add_argument("--lamb_norm_activ_bias_L1", default=0, type=float,
                        help="L1 regularization factor for norm activation biases")
    parser.add_argument("-lamb_norm_activ_bias_L2", default=0, type=float,
                        help="L2 regularization factor for norm activation biases")
    parser.add_argument("--lamb_bn_bias_L1", default=0, type=float,
                        help="L1 regularization factor for batchnorm biases")
    parser.add_argument("--lamb_bn_bias_L2", default=0, type=float,
                        help="L2 regularization factor for batchnorm biases")
    parser.add_argument("--lamb_linear_bias_L1", default=0, type=float,
                        help="L1 regularization factor for fully connected layer biases (except last / classification layer)")
    parser.add_argument("--lamb_linear_bias_L2", default=0, type=float,
                        help="L2 regularization factor for fully connected layer biases (except last / classification layer)")
    parser.add_argument("--lamb_softmax_bias_L1", default=0, type=float,
                        help="L1 regularization factor for classification layer biases")
    parser.add_argument("--lamb_softmax_bias_L2", default=0, type=float,
                        help="L2 regularization factor for classification layer biases")

    args, unparsed = parser.parse_known_args()


    ### Loading Data
    print("Loading data...")
    validation_set = CathData(
             args.data_filename, split=0, download=False,
             randomize_orientation=args.randomize_orientation,
             discretization_bins=args.data_discretization_bins,
             discretization_bin_size=args.data_discretization_bin_size)

    validation_set[0]

    # print('This is len of positions \n', len(validation_set.positions))
    # print('This is positions[0][0] \n',validation_set.positions[0][0])
    # print('This is validation_set.__getitem__(0) \n',validation_set.__getitem__(0))
    #
    # print('The atom types array is \n', validation_set.atom_types)
    # print(vars(validation_set))

    # print('Length of the validation set is ',len(validation_set))
    # print(validation_set[0],'\n')
    # print('validation_set[0][0][0][0][0] is\n',validation_set[1][0][0][0][0])
    #
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
    # n_input = validation_set.n_atom_types
    # #print(f'n_input is {n_input}')
    # #print(f'type is {type(n_input)}')
    # n_output = len(validation_set.label_set)
    #
    # ### Checkpoint loading
    # network_module = importlib.import_module('networks.{:s}.{:s}'.format(args.model, args.model))
    # basepath = 'networks/{:s}'.format(args.model)

    # if args.restore_checkpoint_filename is not None:
    #     checkpoint_path_restore = '{:s}/checkpoints/{:s}'.format(basepath, args.restore_checkpoint_filename)
    #     checkpoint = torch.load(checkpoint_path_restore, map_location=torch.device('cpu'))
    #     timestamp = checkpoint['timestamp']
    # else:
    #     checkpoint = None
    #     timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
    #     os.makedirs('{:s}/checkpoints'.format(basepath), exist_ok=True)
    #
    # torch.backends.cudnn.benchmark = True
    # use_gpu = torch.cuda.is_available()
    #
    # execute(checkpoint)
