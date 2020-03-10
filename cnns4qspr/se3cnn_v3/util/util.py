import os
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

# from util import *
# from format_data import CathData

def load_model(model, latent_size, checkpoint_fn):
    parser = argparse.ArgumentParser()
    # required
    parser.add_argument("--model", default=None,
                        help="Which model definition to use")
    parser.add_argument("--data-filename", default=None,
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
    parser.add_argument("--latent-size", default=20, type=int,
                        help="Size of latent space")
    parser.add_argument("--batchsize-multiplier", default=1, type=int,
                        help="number of minibatch iterations accumulated before applying the update step, effectively multiplying batchsize (default: %(default)s)")
    parser.add_argument("--restore-checkpoint-filename", type=str, default=None,
                        help="Read model from checkpoint given by filename (assumed to be in checkpoint folder)")
    parser.add_argument("--initial_lr", default=1e-4, type=float,
                        help="Initial learning rate (without decay)")
    parser.add_argument("--lr_decay_start", type=int, default=40,
                        help="epoch after which the exponential learning rate decay starts")
    parser.add_argument("--lr_decay_base", type=float, default=.996,
                        help="exponential decay factor per epoch")
    # model
    parser.add_argument("--kernel-size", type=int, default=3,
                        help="convolution kernel size")
    parser.add_argument("--p-drop-conv", type=float, default=0.1,
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

    n_input = 1
    n_output = 3
    latent_size = 20

    print(os.getcwd())
    network_module = importlib.import_module('se3cnn_v3.networks.{:s}.{:s}'.format(model, model))
    basepath = 'se3cnn_v3/networks/{:s}'.format(model)

    checkpoint_path_restore = '{:s}/checkpoints/{:s}'.format(basepath, checkpoint_fn)
    checkpoint = torch.load(checkpoint_path_restore, map_location=torch.device('cpu'))
    timestamp = checkpoint['timestamp']

    model = network_module.network(n_input=n_input, n_output=n_output, latent_size=20, args=args)
    model.load_state_dict(checkpoint['state_dict'])

    return model
