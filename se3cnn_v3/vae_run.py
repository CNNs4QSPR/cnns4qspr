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
    training_accs = []
    latent_space = []
    training_labels = []
    for batch_idx, data in enumerate(loader):
        if use_gpu:
            data = data.cuda()
        input = data[:,:-1]
        label = data[:,-1]

        vae_in = torch.autograd.Variable(input)
        labels = torch.autograd.Variable(label).long()

        # forward and backward propagation
        vae_out, mu, logvar, predictions = model(vae_in)
        z = model.sample_latent_space(vae_in)

        loss_vae = vae_loss(vae_in, vae_out, mu, logvar)
        loss_predictor = predictor_loss(labels, predictions)
        loss = loss_vae + loss_predictor
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        _, argmax = torch.max(predictions, 1)
        acc = (argmax.squeeze() == labels).float().mean()

        total_losses.append(loss.item())
        vae_losses.append(loss_vae.item())
        predictor_losses.append(loss_predictor.item())
        training_accs.append(acc.data.cpu().numpy())
        latent_space.append(mu.data.cpu().numpy())
        training_labels.append(labels.data.cpu().numpy())
    avg_vae_loss = np.mean(vae_losses)
    avg_pred_loss = np.mean(predictor_losses)
    avg_acc = np.mean(training_accs)
    latent_space = np.concatenate(latent_space)
    training_labels = np.concatenate(training_labels)
    return avg_vae_loss, avg_pred_loss, avg_acc, latent_space, training_labels

def predict(model, loader):
    """Make prediction for all entries in loader
    :param model: Model used for prediction
    :param loader: DataLoader object
    """
    model.eval()
    val_accs = []
    latent_space = []
    val_labels = []
    for batch_idx, data in enumerate(loader):
        if use_gpu:
            data = data.cuda()
        input = data[:,:-1]
        label = data[:,-1]

        vae_in = torch.autograd.Variable(input)
        labels = torch.autograd.Variable(label).long()

        # forward propagation
        vae_out, mu, logvar, predictions = model(vae_in)
        z = model.sample_latent_space(vae_in)

        _, argmax = torch.max(predictions, 1)
        acc = (argmax.squeeze() == labels).float().mean()

        val_accs.append(acc.data.cpu().numpy())
        latent_space.append(mu.data.cpu().numpy())
        val_labels.append(labels.data.cpu().numpy())
    avg_val_acc = np.mean(val_accs)
    latent_space = np.concatenate(latent_space)
    val_labels = np.concatenate(val_labels)
    return avg_val_acc, latent_space, val_labels

def train(checkpoint):

    vae = VAE(n_input, latent_size, n_output)
    vae = vae.double()
    if use_gpu:
        vae.cuda()

    optimizer = optim.Adam(vae.parameters(), lr=args.initial_lr)
    optimizer.zero_grad()

    print("Network built...")

    epoch_start_index = 0

    accs = np.zeros((args.training_epochs,))
    val_accs = np.zeros((args.training_epochs,))
    train_latent_space = np.zeros((args.training_epochs, len(train_loader)*args.batch_size, latent_size))
    train_labels = np.zeros((args.training_epochs, len(train_loader)*args.batch_size))
    val_latent_space = np.zeros((args.training_epochs, len(val_loader)*args.batch_size, latent_size))
    val_labels = np.zeros((args.training_epochs, len(val_loader)*args.batch_size))

    for epoch in range(epoch_start_index, args.training_epochs):
        vae_loss, pred_loss, acc, tr_z, tr_lab = train_loop(vae, train_loader, optimizer, epoch)
        val_acc, val_z, val_lab = predict(vae, val_loader)
        acc = np.float64(acc)
        val_acc = np.float64(val_acc)
        print("Epoch {}:".format(epoch+1))
        print("vae_loss={} pred_loss={} acc={} val_acc={}".format(round(vae_loss, 3), round(pred_loss, 3), round(acc, 3), round(val_acc, 3))+'\n')
        accs[epoch] = acc
        val_accs[epoch] = val_acc
        train_latent_space[epoch,:,:] = tr_z
        train_labels[epoch,:] = tr_lab
        val_latent_space[epoch,:,:] = val_z
        val_labels[epoch,:] = val_lab

    np.save('{:s}/data/accs.npy'.format(basepath), accs)
    np.save('{:s}/data/val_accs.npy'.format(basepath), val_accs)
    np.save('{:s}/data/train_latent_space.npy'.format(basepath), train_latent_space)
    np.save('{:s}/data/train_labels.npy'.format(basepath), train_labels)
    np.save('{:s}/data/val_latent_space.npy'.format(basepath), val_latent_space)
    np.save('{:s}/data/val_labels.npy'.format(basepath), val_labels)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--training-epochs", default=100, type=int,
                        help="Which model definition to use")
    parser.add_argument("--batch-size", default=8, type=int,
                        help="Size of mini batches to use per iteration, can be accumulated via argument batchsize_multiplier (default: %(default)s)")
    parser.add_argument("--latent-size", default=20, type=int,
                        help="Size of latent space")
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
    latent_size = args.latent_size

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

    os.makedirs('{:s}/data'.format(basepath), exist_ok=True)

    torch.backends.cudnn.benchmark = True
    use_gpu = torch.cuda.is_available()

    ### Train model
    train(checkpoint)
