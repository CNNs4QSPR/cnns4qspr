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

# python cnn_run.py --model SE3ResNet34Small --data-filename cath_3class_ca.npz --training-epochs 100 --batch-size 8 --latent-size 3

cnn_out = None

def cnn_hook(module, input_, output):
    global cnn_out
    cnn_out = output

def vae_loss(vae_in, vae_out, mu, logvar):
    BCE = F.binary_cross_entropy(vae_out, vae_in.view(-1, 256), reduction='mean')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return abs(BCE) + abs(KLD)

def predictor_loss(labels, predictions):
    PCE = F.cross_entropy(predictions, labels, reduction='mean')
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
    pred_losses = []
    training_accs = []
    latent_space = []
    training_labels = []
    for batch_idx, (data, target) in enumerate(loader):
        time_start = time.perf_counter()

        if use_gpu:
            data, target = data.cuda(), target.cuda()
        x = torch.autograd.Variable(data)
        labels = torch.autograd.Variable(target)

        # forward and backward propagation
        vae_out, mu, logvar, predictions = model(x)
        vae_in = torch.autograd.Variable(cnn_out)

        loss_vae = vae_loss(vae_in, vae_out, mu, logvar)
        loss_pred = predictor_loss(labels, predictions)
        loss = loss_vae + loss_pred
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        _, argmax = torch.max(predictions, 1)
        acc = (argmax.squeeze() == labels).float().mean()

        total_losses.append(loss.item())
        vae_losses.append(loss_vae.item())
        pred_losses.append(loss_pred.item())
        training_accs.append(acc.data.cpu().numpy())
        latent_space.append(mu.data.cpu().numpy())
        training_labels.append(labels.data.cpu().numpy())
        log_obj.write("[{}:{}/{}] loss={:.4} time={:.2}".format(
            epoch, batch_idx, len(train_loader),
            float(loss.item()), time.perf_counter() - time_start))

    avg_loss = np.mean(total_losses)
    avg_vae_loss = np.mean(vae_losses)
    avg_pred_loss = np.mean(pred_losses)
    avg_acc = np.mean(training_accs)
    latent_space = np.concatenate(latent_space)
    training_labels = np.concatenate(training_labels)
    return avg_loss, avg_vae_loss, avg_pred_loss, avg_acc, latent_space, training_labels

def predict(model, loader):
    """Make prediction for all entries in loader
    :param model: Model used for prediction
    :param loader: DataLoader object
    """
    model.eval()
    val_accs = []
    latent_space = []
    val_labels = []

    total_losses = []
    vae_losses = []
    pred_losses = []
    for batch_idx, (data,target) in enumerate(loader):
        if use_gpu:
            data, target = data.cuda(), target.cuda()
        x = torch.autograd.Variable(data, volatile=True)
        labels = torch.autograd.Variable(target, volatile=True)

        # Forward propagation
        vae_out, mu, logvar, predictions = model(x)
        vae_in = torch.autograd.Variable(cnn_out, volatile=True)

        loss_vae = vae_loss(vae_in, vae_out, mu, logvar)
        loss_pred = predictor_loss(labels, predictions)
        loss = loss_vae + loss_pred

        _, argmax = torch.max(predictions, 1)
        acc = (argmax.squeeze() == labels).float().mean()

        val_accs.append(acc.data.cpu().numpy())
        latent_space.append(mu.data.cpu().numpy())
        val_labels.append(labels.data.cpu().numpy())
        total_losses.append(loss.item())
        vae_losses.append(loss_vae.item())
        pred_losses.append(loss_pred.item())

    avg_val_acc = np.mean(val_accs)
    latent_space = np.concatenate(latent_space)
    val_labels = np.concatenate(val_labels)
    avg_loss = np.mean(total_losses)
    avg_vae_loss = np.mean(vae_losses)
    avg_pred_loss = np.mean(pred_losses)
    return avg_val_acc, latent_space, val_labels, avg_loss, avg_vae_loss, avg_pred_loss

def train(checkpoint):

    model = network_module.network(n_input=n_input, n_output=n_output, latent_size=latent_size, args=args)
    model.blocks[5].register_forward_hook(cnn_hook)
    if use_gpu:
        model.cuda()

    # param_groups = get_param_groups.get_param_groups(model, args)
    # optimizer = optimizers_L1L2.Adam(param_groups, lr=args.initial_lr)
    optimizer = optim.Adam(model.parameters(), lr=args.initial_lr)
    optimizer.zero_grad()

    print("Network built...")

    epoch_start_index = 0
    best_acc = 0
    if checkpoint is not None:
        try:
            log_obj.write("Restoring model from: " + checkpoint_path_restore)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            epoch_start_index = checkpoint['epoch']+1
            best_validation_loss = checkpoint['best_validation_loss']
        except RuntimeError:
            checkpoint['state_dict']['blocks.6.predictor.0.weight'] = model.blocks[6].predictor[0].weight
            checkpoint['state_dict']['blocks.6.predictor.0.bias'] = model.blocks[6].predictor[0].bias
            checkpoint['state_dict']['blocks.6.predictor.1.weight'] = model.blocks[6].predictor[1].weight
            checkpoint['state_dict']['blocks.6.predictor.1.bias'] = model.blocks[6].predictor[1].bias
            checkpoint['state_dict']['blocks.6.predictor.2.weight'] = model.blocks[6].predictor[2].weight
            checkpoint['state_dict']['blocks.6.predictor.2.bias'] = model.blocks[6].predictor[2].bias
            model.load_state_dict(checkpoint['state_dict'])
            best_acc = 0

    if args.mode == 'train':

        accs = np.zeros((args.training_epochs,))
        val_accs = np.zeros((args.training_epochs,))
        avg_loss = np.zeros((args.training_epochs,))
        avg_vae_loss = np.zeros((args.training_epochs,))
        avg_pred_loss = np.zeros((args.training_epochs,))
        avg_loss_val = np.zeros((args.training_epochs,))
        avg_vae_loss_val = np.zeros((args.training_epochs,))
        avg_pred_loss_val = np.zeros((args.training_epochs,))
        train_latent_space = np.zeros((args.training_epochs, len(train_loader)*args.batch_size, latent_size))
        train_labels = np.zeros((args.training_epochs, len(train_loader)*args.batch_size))
        val_latent_space = np.zeros((args.training_epochs, len(validation_loader)*args.batch_size, latent_size))
        val_labels = np.zeros((args.training_epochs, len(validation_loader)*args.batch_size))

        for epoch in range(epoch_start_index, args.training_epochs):
            epoch_id = epoch - epoch_start_index

            # decay learning rate
            optimizer, _ = lr_schedulers.lr_scheduler_exponential(optimizer, epoch, args.initial_lr, args.lr_decay_start,
                                                                  args.lr_decay_base, verbose=True)

            avg_l, avg_vae_l, avg_pred_l, avg_acc, latent_space, training_labels = train_loop(model, train_loader, optimizer, epoch)

            if (epoch+1) % args.report_frequency != 0:
                continue

            avg_acc_val, latent_space_val, val_lab, avg_l_val, avg_vae_l_val, avg_pred_l_val = predict(model, validation_loader)
            avg_acc = np.float64(avg_acc)
            avg_acc_val = np.float64(avg_acc_val)
            print("Epoch {}:".format(epoch+1))
            print("vae_loss={} pred_loss={} acc={} val_acc={}".format(round(avg_vae_l, 3), round(avg_pred_l, 3), round(avg_acc, 3), round(avg_acc_val, 3))+'\n')


            accs[epoch] = avg_acc
            val_accs[epoch] = avg_acc_val
            avg_loss[epoch] = avg_l
            avg_vae_loss[epoch] = avg_vae_l
            avg_pred_loss[epoch] = avg_pred_l
            avg_loss_val[epoch] = avg_l_val
            avg_vae_loss_val[epoch] = avg_vae_l_val
            avg_pred_loss_val[epoch] = avg_pred_l_val
            train_latent_space[epoch,:,:] = latent_space
            train_labels[epoch,:] = training_labels
            val_latent_space[epoch,:,:] = latent_space_val
            val_labels[epoch,:] = val_lab

            np.save('{:s}/data/accs.npy'.format(basepath), accs)
            np.save('{:s}/data/val_accs.npy'.format(basepath), val_accs)
            np.save('{:s}/data/avg_loss.npy'.format(basepath), avg_loss)
            np.save('{:s}/data/avg_vae_loss.npy'.format(basepath), avg_vae_loss)
            np.save('{:s}/data/avg_pred_loss.npy'.format(basepath), avg_pred_loss)
            np.save('{:s}/data/avg_loss_val.npy'.format(basepath), avg_loss_val)
            np.save('{:s}/data/avg_vae_loss_val.npy'.format(basepath), avg_vae_loss_val)
            np.save('{:s}/data/avg_pred_loss_val.npy'.format(basepath), avg_pred_loss_val)
            np.save('{:s}/data/train_latent_space.npy'.format(basepath), train_latent_space)
            np.save('{:s}/data/train_labels.npy'.format(basepath), train_labels)
            np.save('{:s}/data/val_latent_space.npy'.format(basepath), val_latent_space)
            np.save('{:s}/data/val_labels.npy'.format(basepath), val_labels)

            # log_obj.write('TRAINING SET [{}:{}/{}] loss={:.4}'.format(
            #     epoch, len(train_loader)-1, len(train_loader),
            #     loss_avg))
            # log_obj.write('VALIDATION SET [{}:{}/{}] loss={:.4}'.format(
            #     epoch, len(validation_loader)-1, len(validation_loader),
            #     loss_avg_val))

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
            for n in range(0, checkpoint_latest_n-1)[::-1]:
                source = checkpoint_path_latest_n.replace('__n__', '_'+str(n))
                target = checkpoint_path_latest_n.replace('__n__', '_'+str(n+1))
                if os.path.exists(source):
                    os.rename(source, target)

            improved=False
            if avg_acc_val > best_acc:
                best_acc = avg_acc_val
                improved = True

            torch.save({'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'best_accuracy': best_acc,
                        'timestamp': timestamp},
                        checkpoint_path_latest_n.replace('__n__', '_0'))

            # optional saving of best validation state
            if improved is True:
                if epoch > args.burnin_epochs:
                    copyfile(src=checkpoint_path_latest_n.replace('__n__', '_0'), dst=checkpoint_path_best)
                    log_obj.write('Best validation loss until now - updated best model')
            elif not improved:
                log_obj.write('Validation loss did not improve')

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
    latent_size = args.latent_size

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
