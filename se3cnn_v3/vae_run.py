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
from util.arch_blocks import VAE
from util.format_data import CathData

def vae_loss(vae_in, vae_out, mu, logvar):
    BCE = F.binary_cross_entropy(vae_out, vae_in.view(-1, 256), reduction='mean')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # print('recon_l_tr={} kld_l_tr={} predict_l_tr = {} total_l_tr={}'.format(abs(BCE), abs(KLD), abs(PCE), abs(BCE)+abs(KLD)+abs(PCE)))
    return abs(BCE) + abs(KLD)

def predictor_loss(labels, predictions):
    PCE = F.cross_entropy(predictions, labels, reduction='mean')

    # print('recon_l_tr={} kld_l_tr={} predict_l_tr = {} total_l_tr={}'.format(abs(BCE), abs(KLD), abs(PCE), abs(BCE)+abs(KLD)+abs(PCE)))
    return PCE

def train_loop(model, loader, optimizer, epoch):
    """Main training loop
    :param model: Model to be trained
    :param train_loader: DataLoader object for training set
    :param optimizer: Optimizer object
    :param epoch: Current epoch index
    """
    model.train()
    total_losses = []
    vae_losses = []
    predictor_losses = []
    for batch_idx, data in enumerate(loader):
        if use_gpu:
            data = data.cuda()
        input = data[:,:-1]
        label = data[:,-1]

        vae_in = torch.autograd.Variable(input)
        labels = torch.autograd.Variable(label).long()

        # forward and backward propagation
        vae_out, mu, logvar, predictions = model(vae_in)

        loss_vae = vae_loss(vae_in, vae_out, mu, logvar)
        loss_predictor = predictor_loss(labels, predictions)
        loss = loss_vae + loss_predictor
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_losses.append(loss.item())
        vae_losses.append(loss_vae.item())
        predictor_losses.append(loss_predictor.item())
    print('vae_loss={} predictor_loss={}'.format(np.mean(vae_losses), np.mean(predictor_losses)))
    return 0

def train(checkpoint):

    vae = VAE(n_input, 20, n_output)
    vae = vae.double()
    if use_gpu:
        vae.cuda()

    optimizer = optim.Adam(vae.parameters(), lr=args.initial_lr)
    optimizer.zero_grad()

    print("Network built...")

    epoch_start_index = 0

    for epoch in range(epoch_start_index, args.training_epochs):
        train_loop(vae, train_loader, optimizer, epoch)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--training-epochs", default=100, type=int,
                        help="Which model definition to use")
    parser.add_argument("--batch-size", default=8, type=int,
                        help="Size of mini batches to use per iteration, can be accumulated via argument batchsize_multiplier (default: %(default)s)")
    parser.add_argument("--restore-checkpoint-filename", type=str, default=None,
                        help="Read model from checkpoint given by filename (assumed to be in checkpoint folder)")
    parser.add_argument("--initial_lr", default=1e-4, type=float,
                        help="Initial learning rate (without decay)")
    parser.add_argument("--lr_decay_start", type=int, default=40,
                        help="epoch after which the exponential learning rate decay starts")
    parser.add_argument("--lr_decay_base", type=float, default=.996,
                        help="exponential decay factor per epoch")

    args, unparsed = parser.parse_known_args()

    ### Data loading
    print("Loading data...")
    datapath = 'datasets/'
    train_data = np.load(datapath+'train_data.npy')
    val_data = np.load(datapath+'val_data.npy')
    test_data = np.load(datapath+'test_data.npy')

    n_input = train_data.shape[1] - 1
    n_output = 3

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                               shuffle=True, num_workers=0,
                                               pin_memory=False, drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size,
                                               shuffle=True, num_workers=0,
                                               pin_memory=False, drop_last=True)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                                               shuffle=True, num_workers=0,
                                               pin_memory=False, drop_last=True)

    ### Checkpoint loading
    basepath = 'networks/VAE'

    if args.restore_checkpoint_filename is not None:
        checkpoint_path_restore = '{:s}/checkpoints/{:s}'.format(basepath, args.restore_checkpoint_filename)
        checkpoint = torch.load(checkpoint_path_restore, map_location=torch.device('cpu'))
        timestamp = checkpoint['timestamp']
    else:
        checkpoint = None
        timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
        os.makedirs('{:s}/checkpoints'.format(basepath), exist_ok=True)

    torch.backends.cudnn.benchmark = True
    use_gpu = torch.cuda.is_available()

    ### Train model
    train(checkpoint)
