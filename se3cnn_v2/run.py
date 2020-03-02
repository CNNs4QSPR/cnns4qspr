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

# python run.py --model SE3ResNet34Small --data-filename cath_3class_ca.npz --training-epochs 100 --batch-size 8 --batchsize-multiplier 1 --kernel-size 3 --initial_lr=0.0001 --lr_decay_base=.996 --p-drop-conv 0.1 --downsample-by-pooling

cnn_out = None
latent_space_mean = None
latent_space_var = None

def cnn_hook(module, input_, output):
    global cnn_out
    cnn_out = output

def latent_mean_hook(module, input_, output):
    global latent_space_mean
    latent_space_mean = output

def latent_var_hook(module, input_, output):
    global latent_space_var
    latent_space_var = output

def loss_function(recon_x, x, mu, logvar, mode='train'):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    global BCE, KLD
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 256), reduction='mean')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    if mode == 'train':
        print('recon_l_tr={} kld_l_tr={} total_l_tr={}'.format(abs(BCE), abs(KLD), abs(BCE)+abs(KLD)))
        return abs(BCE) + abs(KLD)
    elif mode == 'validate':
        print('recon_l_val={} kld_l_val={} total_l_val={}'.format(abs(BCE), abs(KLD), abs(BCE)+abs(KLD)))
        return abs(BCE) + abs(KLD), abs(BCE), abs(KLD)

def train_loop(model, train_loader, optimizer, epoch):
    """Main training loop
    :param model: Model to be trained
    :param train_loader: DataLoader object for training set
    :param optimizer: Optimizer object
    :param epoch: Current epoch index
    """
    model.train()
    training_losses = []
    training_losses_BCE = []
    training_losses_KLD = []
    latent_mean_outs = []
    latent_var_outs = []
    training_labels = []
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx > 3:
            break
        time_start = time.perf_counter()

        if use_gpu:
            data, target = data.cuda(), target.cuda()
        x = torch.autograd.Variable(data)
        label = torch.autograd.Variable(target)

        # forward and backward propagation
        vae_out, mu, logvar = model(x)
        vae_in = torch.autograd.Variable(cnn_out)

        loss = loss_function(vae_out, vae_in, mu, logvar)
        loss.backward()
        if batch_idx % args.batchsize_multiplier == args.batchsize_multiplier-1:
            optimizer.step()
            optimizer.zero_grad()

        training_losses.append(loss.data.cpu().numpy())
        training_losses_BCE.append(BCE.data.cpu().numpy())
        training_losses_KLD.append(KLD.data.cpu().numpy())
        latent_mean_outs.append(latent_space_mean.data.cpu().numpy())
        latent_var_outs.append(latent_space_var.data.cpu().numpy())
        training_labels.append(label.data.cpu().numpy()[0])

        log_obj.write("[{}:{}/{}] loss={:.4} time={:.2}".format(
            epoch, batch_idx, len(train_loader),
            float(loss.item()), time.perf_counter() - time_start))

    loss_avg = np.mean(training_losses)
    loss_avg_BCE = np.mean(training_losses_BCE)
    loss_avg_KLD = np.mean(training_losses_KLD)
    training_losses = np.array(training_losses)
    training_losses_BCE = np.array(training_losses_BCE)
    training_losses_KLD = np.array(training_losses_KLD)
    latent_mean_outs = np.concatenate(latent_mean_outs)
    latent_var_outs = np.concatenate(latent_var_outs)
    training_labels = np.array(training_labels)
    return loss_avg, loss_avg_BCE, loss_avg_KLD, training_losses, latent_mean_outs, latent_var_outs, training_labels


def infer(model, loader):
    """Make prediction for all entries in loader
    :param model: Model used for prediction
    :param loader: DataLoader object
    """
    model.eval()
    training_losses = []
    training_losses_BCE = []
    training_losses_KLD = []
    latent_mean_outs = []
    latent_var_outs = []
    training_labels = []
    for batch_idx, (data,target) in enumerate(loader):
        if use_gpu:
            data, target = data.cuda(), target.cuda()
        x = torch.autograd.Variable(data, volatile=True)
        label = torch.autograd.Variable(target, volatile=True)

        # Forward propagation
        vae_out, mu, logvar = model(x)
        vae_in = torch.autograd.Variable(cnn_out, volatile=True)

        loss, bce_loss, kld_loss = loss_function(vae_out, vae_in, mu, logvar, mode='validate')
        training_losses.append(loss.data.cpu().numpy())
        training_losses_BCE.append(bce_loss.data.cpu().numpy())
        training_losses_KLD.append(kld_loss.data.cpu().numpy())
        latent_mean_outs.append(latent_space_mean.data.cpu().numpy())
        latent_var_outs.append(latent_space_var.data.cpu().numpy())
        training_labels.append(label.data.cpu().numpy()[0])

    loss_avg = np.mean(training_losses)
    loss_avg_BCE = np.mean(training_losses_BCE)
    loss_avg_KLD = np.mean(training_losses_KLD)
    training_losses = np.array(training_losses)
    training_losses_BCE = np.array(training_losses_BCE)
    training_losses_KLD = np.array(training_losses_KLD)
    latent_mean_outs = np.concatenate(latent_mean_outs)
    latent_var_outs = np.concatenate(latent_var_outs)
    training_labels = np.array(training_labels)
    return loss_avg, loss_avg_BCE, loss_avg_KLD, training_losses, latent_mean_outs, latent_var_outs, training_labels

def train(checkpoint):

    model = network_module.network(n_input=n_input, n_output=n_output, args=args)
    model.blocks[5].register_forward_hook(cnn_hook)
    model.blocks[6].layers[1].register_forward_hook(latent_mean_hook)
    model.blocks[6].layers[2].register_forward_hook(latent_var_hook)
    if use_gpu:
        model.cuda()

    param_groups = get_param_groups.get_param_groups(model, args)
    optimizer = optimizers_L1L2.Adam(param_groups, lr=args.initial_lr)
    optimizer.zero_grad()

    print("Network built...")

    epoch_start_index = 0
    if checkpoint is not None:
        log_obj.write("Restoring model from: " + checkpoint_path_restore)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_start_index = checkpoint['epoch']+1
        best_validation_loss = checkpoint['best_validation_loss']

    if args.mode == 'train':

        print("Training...")
        train_batch_size = len(train_loader)
        val_batch_size = len(validation_loader)
        loss_avg_store = np.zeros((args.training_epochs,))
        loss_avg_BCE_store = np.zeros((args.training_epochs,))
        loss_avg_KLD_store = np.zeros((args.training_epochs,))
        loss_avg_val_store = np.zeros((args.training_epochs,))
        loss_avg_BCE_val_store = np.zeros((args.training_epochs,))
        loss_avg_KLD_val_store = np.zeros((args.training_epochs,))
        training_losses_store = np.zeros((args.training_epochs, train_batch_size))
        training_labels_store = np.zeros((args.training_epochs, train_batch_size))
        training_losses_val_store = np.zeros((args.training_epochs, val_batch_size))
        training_labels_val_store = np.zeros((args.training_epochs, val_batch_size))
        latent_mean_outs_store = np.zeros((args.training_epochs, int(train_batch_size*args.batch_size), 20))
        latent_var_outs_store = np.zeros((args.training_epochs, int(train_batch_size*args.batch_size), 20))
        latent_mean_outs_val_store = np.zeros((args.training_epochs, int(val_batch_size*args.batch_size), 20))
        latent_var_outs_val_store = np.zeros((args.training_epochs, int(val_batch_size*args.batch_size), 20))

        for epoch in range(epoch_start_index, args.training_epochs):
            epoch_id = epoch - epoch_start_index

            # decay learning rate
            optimizer, _ = lr_schedulers.lr_scheduler_exponential(optimizer, epoch, args.initial_lr, args.lr_decay_start,
                                                                  args.lr_decay_base, verbose=True)

            loss_avg, loss_avg_BCE, loss_avg_KLD, training_losses, latent_mean_outs, latent_var_outs, training_labels = train_loop(model, train_loader, optimizer, epoch)

            if (epoch+1) % args.report_frequency != 0:
                continue

            loss_avg_val, loss_avg_BCE_val, loss_avg_KLD_val, training_losses_val, latent_mean_outs_val, latent_var_outs_val, training_labels_val = infer(model, validation_loader)

            loss_avg_store[epoch_id] = loss_avg
            loss_avg_BCE_store[epoch_id] = loss_avg_BCE
            loss_avg_KLD_store[epoch_id] = loss_avg_KLD
            loss_avg_val_store[epoch_id] = loss_avg_val
            loss_avg_BCE_val_store[epoch_id] = loss_avg_BCE_val
            loss_avg_KLD_val_store[epoch_id] = loss_avg_KLD_val
            training_losses_store[epoch_id,:] = training_losses
            training_labels_store[epoch_id,:] = training_labels
            training_losses_val_store[epoch_id,:] = training_losses_val
            training_labels_val_store[epoch_id,:] = training_labels_val
            latent_mean_outs_store[epoch_id,:,:] = latent_mean_outs
            latent_var_outs_store[epoch_id,:,:] = latent_var_outs
            latent_mean_outs_val_store[epoch_id,:,:] = latent_mean_outs_val
            latent_var_outs_val_store[epoch_id,:,:] = latent_var_outs_val

            np.save('{:s}/data/loss_avg.npy'.format(basepath), loss_avg_store)
            np.save('{:s}/data/loss_avg_BCE.npy'.format(basepath), loss_avg_store)
            np.save('{:s}/data/loss_avg_KLD.npy'.format(basepath), loss_avg_store)
            np.save('{:s}/data/loss_avg_val.npy'.format(basepath), loss_avg_store)
            np.save('{:s}/data/loss_avg_BCE_val.npy'.format(basepath), loss_avg_store)
            np.save('{:s}/data/loss_avg_KLD_val.npy'.format(basepath), loss_avg_store)
            np.save('{:s}/data/training_losses.npy'.format(basepath), loss_avg_store)
            np.save('{:s}/data/training_labels.npy'.format(basepath), loss_avg_store)
            np.save('{:s}/data/training_losses_val.npy'.format(basepath), loss_avg_store)
            np.save('{:s}/data/training_labels_val.npy'.format(basepath), loss_avg_store)
            np.save('{:s}/data/latent_mean_outs.npy'.format(basepath), loss_avg_store)
            np.save('{:s}/data/latent_var_outs.npy'.format(basepath), loss_avg_store)
            np.save('{:s}/data/latent_mean_outs_val.npy'.format(basepath), loss_avg_store)
            np.save('{:s}/data/latent_var_outs_val.npy'.format(basepath), loss_avg_store)

            log_obj.write('TRAINING SET [{}:{}/{}] loss={:.4}'.format(
                epoch, len(train_loader)-1, len(train_loader),
                loss_avg))
            log_obj.write('VALIDATION SET [{}:{}/{}] loss={:.4}'.format(
                epoch, len(validation_loader)-1, len(validation_loader),
                loss_avg_val))

            ### TEST SET DOES NOT WORK YET
            # if args.report_on_test_set:
            #     test_outs, test_ys, test_losses = infer(model,
            #                                        test_loader)
            #
            #     # compute the accuracy
            #     test_acc = np.sum(test_outs.argmax(-1) == test_ys) / len(test_ys)
            #
            #     test_loss_avg = np.mean(test_losses)
            #
            #     log_obj.write(
            #         'TEST SET [{}:{}/{}] loss={:.4} acc={:.3}'.format(
            #             epoch, len(train_loader) - 1, len(train_loader),
            #             test_loss_avg, test_acc))

            # saving of latest state
            # for n in range(0, checkpoint_latest_n-1)[::-1]:
            #     source = checkpoint_path_latest_n.replace('__n__', '_'+str(n))
            #     target = checkpoint_path_latest_n.replace('__n__', '_'+str(n+1))
            #     if os.path.exists(source):
            #         os.rename(source, target)
            #
            # improved=False
            # if loss_avg_val < best_validation_loss:
            #     best_validation_loss = loss_avg_val
            #     improved = True
            #
            # torch.save({'state_dict': model.state_dict(),
            #             'optimizer': optimizer.state_dict(),
            #             'epoch': epoch,
            #             'best_validation_loss': best_validation_loss,
            #             'timestamp': timestamp},
            #             checkpoint_path_latest_n.replace('__n__', '_0'))
            #
            # # optional saving of best validation state
            # if improved is True:
            #     if epoch > args.burnin_epochs:
            #         copyfile(src=checkpoint_path_latest_n.replace('__n__', '_0'), dst=checkpoint_path_best)
            #         log_obj.write('Best validation loss until now - updated best model')
            # elif not improved:
            #     log_obj.write('Validation loss did not improve')

    ### THESE MODES DON'T WORK YET
    elif args.mode == 'validate':
        out, y, validation_losses = infer(model, validation_loader)

        # compute the accuracy
        validation_acc = np.sum(out.argmax(-1) == y) / len(y)
        validation_loss_avg = np.mean(validation_losses)

        log_obj.write('VALIDATION SET: loss={:.4} acc={:.3}'.format(
            validation_loss_avg, validation_acc))

    elif args.mode == 'test':
        out, y, test_losses = infer(model, test_loader)

        # compute the accuracy
        test_acc = np.sum(out.argmax(-1) == y) / len(y)
        test_loss_avg = np.mean(test_losses)

        log_obj.write('TEST SET: loss={:.4} acc={:.3}'.format(
            test_loss_avg, test_acc))


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
    train_set = torch.utils.data.ConcatDataset([
        CathData(args.data_filename, split=i, download=False,
             randomize_orientation=args.randomize_orientation,
             discretization_bins=args.data_discretization_bins,
             discretization_bin_size=args.data_discretization_bin_size) for i in range(7)])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                               shuffle=True, num_workers=0,
                                               pin_memory=False, drop_last=True)
    n_input = train_set.datasets[0].n_atom_types
    n_output = len(train_set.datasets[0].label_set)

    validation_set = CathData(
             args.data_filename, split=7, download=False,
             randomize_orientation=args.randomize_orientation,
             discretization_bins=args.data_discretization_bins,
             discretization_bin_size=args.data_discretization_bin_size)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)

    ### Checkpoint loading
    network_module = importlib.import_module('networks.{:s}.{:s}'.format(args.model, args.model))
    basepath = 'networks/{:s}'.format(args.model)

    if args.restore_checkpoint_filename is not None:
        checkpoint_path_restore = '{:s}/checkpoints/{:s}'.format(basepath, args.restore_checkpoint_filename)
        checkpoint = torch.load(checkpoint_path_restore, map_location=torch.device('cpu'))
        timestamp = checkpoint['timestamp']
    else:
        checkpoint = None
        timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
        os.makedirs('{:s}/checkpoints'.format(basepath), exist_ok=True)

    checkpoint_latest_n = 1
    checkpoint_path_latest_n = '{:s}/checkpoints/{:s}_latest__n__.ckpt'.format(basepath, timestamp)
    checkpoint_path_best   = '{:s}/checkpoints/{:s}_best.ckpt'.format(basepath, timestamp)
    checkpoint_path_best_window_avg   = '{:s}/checkpoints/{:s}_best_window_avg.ckpt'.format(basepath, timestamp)

    log_obj = logger.logger(basepath=basepath, timestamp=timestamp)
    if checkpoint != None:
        log_obj.write('\n' + 42*'=' + '\n')
        log_obj.write('\n model restored from checkpoint\n')
    log_obj.write('basepath = {:s}'.format(basepath))
    log_obj.write('timestamp = {:s}'.format(timestamp))
    log_obj.write('\n# Options')
    for key, value in sorted(vars(args).items()):
        log_obj.write('\t'+str(key)+'\t'+str(value))

    ### Set up data storage
    os.makedirs('{:s}/data'.format(basepath), exist_ok=True)

    torch.backends.cudnn.benchmark = True
    use_gpu = torch.cuda.is_available()

    ### Train model
    train(checkpoint)
