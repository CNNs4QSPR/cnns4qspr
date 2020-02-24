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

from cath.datasets.cath.cath import Cath
from cath.util import *

cnn_out = None

def cnn_hook(module, input_, output):
    global cnn_out
    cnn_out = output

def loss_function(recon_x, x, mu, logvar, mode='train'):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 256), reduction='mean')
    #criterion = nn.MSELoss()
    #BCE = criterion(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    print(BCE, KLD)

    if mode == 'train':
        print('train')
        print('recon_l={}, kld_l={}, total_l={}'.format(abs(BCE), abs(KLD), abs(BCE)+abs(KLD)))
        return abs(BCE) + abs(KLD)
    elif mode == 'validate':
        print('train')
        print('recon_l_val={}, kld_l_val={}, total_l_val={}'.format(abs(BCE), abs(KLD), abs(BCE)+abs(KLD)))
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
    training_outs = []
    training_accs = []
    for batch_idx, (data, target) in enumerate(train_loader):
        print(batch_idx)
        time_start = time.perf_counter()

        # target = torch.LongTensor(target)
        if use_gpu:
            data = data.cuda()
        x = torch.autograd.Variable(data)
        # y = torch.autograd.Variable(target)
        # forward and backward propagation
        recon_out, mu, logvar = model(x)
        t = torch.autograd.Variable(cnn_out)

        loss = loss_function(recon_out, t, mu, logvar)
        # losses = nn.functional.cross_entropy(out, y, reduce=False)
        # loss = losses.mean()
        loss.backward()
        if batch_idx % args.batchsize_multiplier == args.batchsize_multiplier-1:
            optimizer.step()
            optimizer.zero_grad()

        _, argmax = torch.max(recon_out, 1)
 #       acc = (argmax.squeeze() == t).float().mean()

        training_losses.append(loss.data.cpu().numpy())
        training_outs.append(recon_out.data.cpu().numpy())
 #       training_accs.append(acc.item())

        log_obj.write("[{}:{}/{}] loss={:.4} time={:.2}".format(
            epoch, batch_idx, len(train_loader),
            float(loss.item()),
            time.perf_counter() - time_start))

    loss_avg = np.mean(training_losses)
#    acc_avg = np.mean(training_accs)
    training_outs = np.concatenate(training_outs)
    training_losses = np.array(training_losses)
    return loss_avg, training_outs, training_losses


def infer(model, loader):
    """Make prediction for all entries in loader
    :param model: Model used for prediction
    :param loader: DataLoader object
    """
    model.eval()
    recon_losses = []
    kld_losses = []
    total_losses = []
    recon_outs = []
    cnn_outs = []
    ys = []
    for data,target in loader:
        if use_gpu:
            data = data.cuda()
        x = torch.autograd.Variable(data, volatile=True)
        #y = torch.autograd.Variable(target, volatile=True)
        recon_out, mu, logvar = model(x)
        t = torch.autograd.Variable(cnn_out, volatile=True)
        #recon_outs.append(recon_out.data.cpu().numpy())
        #cnn_outs.append(cnn_out.data.cpu().numpy())
        #ys.append(y.data.cpu().numpy())
        #BCE_val = F.binary_cross_entropy(recon_out, t.view(-1, 256), reduction='mean')
        #criterion = nn.MSELoss()
        #BCE_val = criterion(recon_out, t)
        #KLD_val = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        #losses.append(BCE_val+KLD_val)
        #losses.append(nn.functional.cross_entropy(out, y, reduce=False).data.cpu().numpy())
        loss, recon_l, kld_l = loss_function(recon_out, t, mu, logvar,
        mode='validate')
        recon_losses.append(recon_l.data.cpu().numpy())
        kld_losses.append(kld_l.data.cpu().numpy())
        total_losses.append(loss.data.cpu().numpy())
    #recon_outs = np.concatenate(recon_outs)
    #cnn_outs = np.concatenate(cnn_outs)
    #ys = np.concatenate(ys)
    return total_losses, recon_losses, kld_losses

### Arguments

# python orion_test2.py --model SE3ResNet34Small --data-filename cath_3class_ca.npz --training-epochs 100 --batch-size 8 --batchsize-multiplier 1 --kernel-size 3 --initial_lr=0.0001 --lr_decay_base=.996 --p-drop-conv 0.1 --downsample-by-pooling

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
train_set = Cath(
         args.data_filename, split=0, download=False,
         randomize_orientation=args.randomize_orientation,
         discretization_bins=args.data_discretization_bins,
         discretization_bin_size=args.data_discretization_bin_size)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
n_input = train_set.n_atom_types
n_output = len(train_set.label_set)

# train_set = torch.utils.data.ConcatDataset([
#     Cath(args.data_filename, split=i, download=False,
#          randomize_orientation=args.randomize_orientation,
#          discretization_bins=args.data_discretization_bins,
#          discretization_bin_size=args.data_discretization_bin_size) for i in range(7)])
# train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
# n_input = train_set.datasets[0].n_atom_types
# n_output = len(train_set.datasets[0].label_set)

validation_set = Cath(
         args.data_filename, split=7, download=False,
         randomize_orientation=args.randomize_orientation,
         discretization_bins=args.data_discretization_bins,
         discretization_bin_size=args.data_discretization_bin_size)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
n_input = validation_set.n_atom_types
n_output = len(validation_set.label_set)

print("Data loaded...")
network_module = importlib.import_module('cath.scripts.cath.networks.{:s}.{:s}'.format(args.model, args.model))
basepath = 'cath/scripts/cath/networks/{:s}'.format(args.model)

if args.restore_checkpoint_filename is not None:
    checkpoint_path_restore = '{:s}/checkpoints/{:s}'.format(basepath, args.restore_checkpoint_filename)
    checkpoint = torch.load(checkpoint_path_restore)
    timestamp = checkpoint['timestamp']
else:
    checkpoint = None
    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
    os.makedirs('{:s}/checkpoints'.format(basepath), exist_ok=True)

checkpoint_latest_n = 5
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

torch.backends.cudnn.benchmark = True
use_gpu = torch.cuda.is_available()


model = network_module.network(n_input=n_input, n_output=n_output, args=args)
if use_gpu:
    model.cuda()

param_groups = get_param_groups.get_param_groups(model, args)
optimizer = optimizers_L1L2.Adam(param_groups, lr=args.initial_lr)
# optimizer = optim.Adam(model.parameters(), lr=args.initial_lr)
optimizer.zero_grad()

print("Network built...")

### Test Space
# counter = 0
# for data in train_loader:
#     inputs, labels = data
#     if counter == 0:
#         # x = inputs.numpy()[0,:,:,:,:]
#         # x = torch.from_numpy(x)
#         x = inputs
#         break
#
#
model.blocks[5].register_forward_hook(cnn_hook)
# # model.blocks[4].register_forward_hook(cnn_hook)
# model.eval()
# recon_out, mu, logvar = model(x)
# print(recon_out.shape, mu.shape, logvar.shape)
# print(cnn_output.shape)


epoch_start_index = 0
best_validication_acc = -float('inf')
best_avg_validation_acc = -float('inf')
latest_validation_accs = []
if checkpoint is not None:
    log_obj.write("Restoring model from: " + checkpoint_path_restore)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch_start_index = checkpoint['epoch']+1
    #best_validation_acc = checkpoint['best_validation_acc']
    #best_avg_validation_acc = checkpoint['best_avg_validation_acc']
    #latest_validation_accs = checkpoint['latest_validation_accs']

tf_logger, tensorflow_available = tensorflow_logger.get_tf_logger(basepath=basepath, timestamp=timestamp)

# if args.mode == 'train':
#
#     for epoch in range(epoch_start_index, args.training_epochs):
#
#         # decay learning rate
#         optimizer, _ = lr_schedulers.lr_scheduler_exponential(optimizer, epoch, args.initial_lr, args.lr_decay_start,
#                                                               args.lr_decay_base, verbose=True)
#
#         loss_avg, training_outs, training_losses = train_loop(model, train_loader, optimizer, epoch)
#
#         if (epoch+1) % args.report_frequency != 0:
#             continue
#
#         # validation_outs, ys, validation_losses = infer(model, validation_loader)
#         #
#         # # compute the accuracy
#         # validation_acc = np.sum(validation_outs.argmax(-1) == ys) / len(ys)
#         #
#         # validation_loss_avg = np.mean(validation_losses)
#         #
#         # log_obj.write('TRAINING SET [{}:{}/{}] loss={:.4} acc={:.3}'.format(
#         #     epoch, len(train_loader)-1, len(train_loader),
#         #     loss_avg, acc_avg))
#         # log_obj.write('VALIDATION SET [{}:{}/{}] loss={:.4} acc={:.3}'.format(
#         #     epoch, len(train_loader)-1, len(train_loader),
#         #     validation_loss_avg, validation_acc))
#         #
#         # log_obj.write('VALIDATION losses: ' + str(validation_losses))
#         #
#         # if args.report_on_test_set:
#         #     test_outs, test_ys, test_losses = infer(model,
#         #                                        test_loader)
#         #
#         #     # compute the accuracy
#         #     test_acc = np.sum(test_outs.argmax(-1) == test_ys) / len(test_ys)
#         #
#         #     test_loss_avg = np.mean(test_losses)
#         #
#         #     log_obj.write(
#         #         'TEST SET [{}:{}/{}] loss={:.4} acc={:.3}'.format(
#         #             epoch, len(train_loader) - 1, len(train_loader),
#         #             test_loss_avg, test_acc))
#         #
#         # ============ TensorBoard logging ============ #
#         # if tensorflow_available:
#         #     # (1) Log the scalar values
#         #     info = {'training set avg loss': loss_avg,
#         #             'training set accuracy': acc_avg,
#         #             'validation set avg loss': validation_loss_avg,
#         #             'validation set accuracy': validation_acc}
#         #     if args.report_on_test_set:
#         #         info.update({'test set avg loss': test_loss_avg,
#         #                      'test set accuracy': test_acc})
#         #     for tag, value in info.items():
#         #         tf_logger.scalar_summary(tag, value, step=epoch+1)
#         #
#         #     # (2) Log values and gradients of the parameters (histogram)
#         #     for tag, value in model.named_parameters():
#         #         tag = tag.replace('.', '/')
#         #         tf_logger.histo_summary(tag,         value.data.cpu().numpy(),      step=epoch+1)
#         #         tf_logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), step=epoch+1)
#         #
#         #     # (3) Log losses for all datapoints in validation and training set
#         #     tf_logger.histo_summary("losses/validation/", validation_losses, step=epoch+1)
#         #     tf_logger.histo_summary("losses/training",    training_losses,   step=epoch+1)
#         #
#         #     # (4) Log logits for all datapoints in validation and training set
#         #     for i in range(n_output):
#         #         tf_logger.histo_summary("logits/%d/validation" % i, validation_outs[:, i], step=epoch+1)
#         #         tf_logger.histo_summary("logits/%d/training" % i,   training_outs[:, i],   step=epoch+1)
#         #
#         #
#         # # saving of latest state
#         # for n in range(0, checkpoint_latest_n-1)[::-1]:
#         #     source = checkpoint_path_latest_n.replace('__n__', '_'+str(n))
#         #     target = checkpoint_path_latest_n.replace('__n__', '_'+str(n+1))
#         #     if os.path.exists(source):
#         #         os.rename(source, target)
#
#         torch.save({'state_dict': model.state_dict(),
#                     'optimizer': optimizer.state_dict(),
#                     'epoch': epoch,
#                     #'best_validation_acc': best_validation_acc,
#                     #'best_avg_validation_acc': best_avg_validation_acc,
#                     #'latest_validation_accs': latest_validation_accs,
#                     'timestamp': timestamp},
#                     checkpoint_path_latest_n.replace('__n__', '_0'))
#
#         # latest_validation_accs.append(validation_acc)
#         # if len(latest_validation_accs) > checkpoint_latest_n:
#         #     latest_validation_accs.pop(0)
#         # latest_validation_accs_avg = -float('inf')
#         # if len(latest_validation_accs) == checkpoint_latest_n:
#         #     latest_validation_accs_avg = np.average(latest_validation_accs)
#         #
#         # # optional saving of best validation state
#         # improved = False
#         # if epoch > args.burnin_epochs:
#         #     if validation_acc > best_validation_acc:
#         #         best_validation_acc = validation_acc
#         #         copyfile(src=checkpoint_path_latest_n.replace('__n__', '_0'), dst=checkpoint_path_best)
#         #         log_obj.write('Best validation accuracy until now - updated best model')
#         #         improved = True
#         #     if latest_validation_accs_avg > best_avg_validation_acc:
#         #         best_avg_validation_acc = latest_validation_accs_avg
#         #         copyfile(src=checkpoint_path_latest_n.replace('__n__', '_'+str(checkpoint_latest_n//2)),
#         #                  dst=checkpoint_path_best_window_avg)
#         #         log_obj.write('Best validation accuracy (window average)_until now - updated best (window averaged) model')
#         #         improved = True
#         #     if not improved:
#         #         log_obj.write('Validation loss did not improve')

if args.mode == 'train':

    for epoch in range(epoch_start_index, args.training_epochs):

        # decay learning rate
        optimizer, _ = lr_schedulers.lr_scheduler_exponential(optimizer, epoch, args.initial_lr, args.lr_decay_start,
                                                              args.lr_decay_base, verbose=True)

        loss_avg, training_outs, training_losses = train_loop(model, train_loader, optimizer, epoch)

        if (epoch+1) % args.report_frequency != 0:
            continue

        total_losses, recon_losses, kld_losses = infer(model, validation_loader)

 #       validation_recon_outs, validation_cnn_outs, validation_losses = infer(model, validation_loader)
        #
        # # compute the accuracy
        # validation_acc = np.sum(validation_outs.argmax(-1) == ys) / len(ys)
        #
#        validation_loss_avg = np.mean(validation_losses)

        log_obj.write('TRAINING SET [{}:{}/{}] loss={:.4}'.format(
            epoch, len(train_loader)-1, len(train_loader),
            loss_avg))
#        log_obj.write('VALIDATION SET [{}:{}/{}] loss={:.4}'.format(
#            epoch, len(train_loader)-1, len(train_loader),
#            validation_loss_avg))
#        
#        log_obj.write('VALIDATION losses: ' + str(validation_losses))
        #
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
        #
        # ============ TensorBoard logging ============ #
        # if tensorflow_available:
        #     # (1) Log the scalar values
        #     info = {'training set avg loss': loss_avg,
        #             'training set accuracy': acc_avg,
        #             'validation set avg loss': validation_loss_avg,
        #             'validation set accuracy': validation_acc}
        #     if args.report_on_test_set:
        #         info.update({'test set avg loss': test_loss_avg,
        #                      'test set accuracy': test_acc})
        #     for tag, value in info.items():
        #         tf_logger.scalar_summary(tag, value, step=epoch+1)
        #
        #     # (2) Log values and gradients of the parameters (histogram)
        #     for tag, value in model.named_parameters():
        #         tag = tag.replace('.', '/')
        #         tf_logger.histo_summary(tag,         value.data.cpu().numpy(),      step=epoch+1)
        #         tf_logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), step=epoch+1)
        #
        #     # (3) Log losses for all datapoints in validation and training set
        #     tf_logger.histo_summary("losses/validation/", validation_losses, step=epoch+1)
        #     tf_logger.histo_summary("losses/training",    training_losses,   step=epoch+1)
        #
        #     # (4) Log logits for all datapoints in validation and training set
        #     for i in range(n_output):
        #         tf_logger.histo_summary("logits/%d/validation" % i, validation_outs[:, i], step=epoch+1)
        #         tf_logger.histo_summary("logits/%d/training" % i,   training_outs[:, i],   step=epoch+1)


        # saving of latest state
        for n in range(0, checkpoint_latest_n-1)[::-1]:
            source = checkpoint_path_latest_n.replace('__n__', '_'+str(n))
            target = checkpoint_path_latest_n.replace('__n__', '_'+str(n+1))
            if os.path.exists(source):
                os.rename(source, target)

        torch.save({'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    #'best_validation_acc': best_validation_acc,
                    #'best_avg_validation_acc': best_avg_validation_acc,
                    #'latest_validation_accs': latest_validation_accs,
                    'timestamp': timestamp},
                    checkpoint_path_latest_n.replace('__n__', '_0'))

        # latest_validation_accs.append(validation_acc)
        # if len(latest_validation_accs) > checkpoint_latest_n:
        #     latest_validation_accs.pop(0)
        # latest_validation_accs_avg = -float('inf')
        # if len(latest_validation_accs) == checkpoint_latest_n:
        #     latest_validation_accs_avg = np.average(latest_validation_accs)
        #
        # # optional saving of best validation state
        # improved = False
        # if epoch > args.burnin_epochs:
        #     if validation_acc > best_validation_acc:
        #         best_validation_acc = validation_acc
        #         copyfile(src=checkpoint_path_latest_n.replace('__n__', '_0'), dst=checkpoint_path_best)
        #         log_obj.write('Best validation accuracy until now - updated best model')
        #         improved = True
        #     if latest_validation_accs_avg > best_avg_validation_acc:
        #         best_avg_validation_acc = latest_validation_accs_avg
        #         copyfile(src=checkpoint_path_latest_n.replace('__n__', '_'+str(checkpoint_latest_n//2)),
        #                  dst=checkpoint_path_best_window_avg)
        #         log_obj.write('Best validation accuracy (window average)_until now - updated best (window averaged) model')
        #         improved = True
        #     if not improved:
        #         log_obj.write('Validation loss did not improve')


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
