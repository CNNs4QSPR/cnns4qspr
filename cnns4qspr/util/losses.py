import numpy as np
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F

def vae_loss(vae_in, vae_out, mu, logvar):
    BCE = F.binary_cross_entropy(vae_out, vae_in.view(-1, 256), reduction='mean')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return abs(BCE) + abs(KLD)

def classifier_loss(labels, predictions):
    labels = labels.long()
    PCE = F.cross_entropy(predictions, labels, reduction='mean')
    return PCE

def regressor_loss(labels, predictions):
    labels = labels.view(-1,1)
    MSE = F.mse_loss(predictions, labels, reduction='mean')
    return MSE
