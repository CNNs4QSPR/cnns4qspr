"""
This module contains a Trainer class for quickly and easily building
VAE or FeedForward property predictors from the extracted structural protein
features.
"""

import os
import sys
import copy
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import numpy as np
import importlib
from se3cnn.util.get_param_groups import get_param_groups
from se3cnn.util.optimizers_L1L2 import Adam
from se3cnn.util.lr_schedulers import lr_scheduler_exponential
from cnns4qspr.util.pred_blocks import VAE, FeedForward
from cnns4qspr.util.losses import vae_loss, classifier_loss, regressor_loss
from cnns4qspr.util.tools import list_parser

class Trainer():
    """
    Loads VAE or FeedForward pytorch network classes and allows the user
    to quickly define its architecture and train. This class has pre-defined
    options for both classifier and regressor model types. If classifier is
    chosen, the output size must match the number of labels being trained on.

    Trained models can be saved as .ckpt files which can later be loaded for
    further model evaluation and/or training.

    Parameters:
        type (str): The target of the neural network (classifier or regressor)

        network_type (str): The type of neural network (vae or feedforward)

        output_size (int): Number of output nodes (defaults to 1 if regressor)

        input_size (int): The number of input nodes (defaults to 256,
            the size of the featurized protein vectors)

        latent_size (int): The number of latent nodes in VAE

        encoder (list): A list of hidden layer sizes for encoder network

        decoder (list): A list of hidden layer sizes for decoder network

        predictor (list): A list of hidden layer sizes for predictor network
            (*note* if network_type is vae then predictor is appended to the
            latent space, if network_type is feedforward then predictor takes
            the structural feature vector as input)

    Attributes:
        type (str): The target of the neural network (classifier or regressor)

        network_type (str): The type of neural network (vae or feedforward)

        latent_size (int): The number of latent nodes in VAE

        network (pytorch Module): Instantiation of VAE or FeedForward pytorch
            network class

        architecture (dict): Dictionary containing the size and number of all
            neural net layers

        current_state (dict): Dictionary containing the most recent values
            of neural net parameters (weights, biases, optimizer history,
            epoch #)

        checkpoint (dict): Dictionary containing the saved values of the
            neural net parameters at the networks' best validation
            performance (user defines based on accuracy or loss)

        load_state (bool): Indicates whether a previous model has been loaded

        total_losses (dict): Saved series of the total loss during both training
            and validation

        vae_losses (dict): Saved series of the VAE loss during both training
            and validation

        predictor_losses (dict): Saved series of the predictor loss during both
            training and validation

        accuracies (dict): Saved series of predictor accuracies during both
            training and validation

        latent_means (dict): Saved series of latent space mean values during
            both training and validation

        label_history (dict): Saved series of label values during both
            training and validation

        n_epochs (int): Number of epochs the model has been trained for
    """
    def __init__(self,
                 type='classifier',
                 network_type='vae',
                 output_size=3,
                 input_size=256,
                 latent_size=3,
                 encoder=[128],
                 decoder=[128],
                 predictor=[128, 128]):
        self.type = type
        self.network_type = network_type
        self.latent_size = latent_size
        if self.type == 'regressor':
            output_size = 1
        if self.network_type == 'vae':
            self.network = VAE(latent_size=latent_size,
                               type=type,
                               n_output=output_size,
                               input_size=input_size,
                               encoder=encoder,
                               decoder=decoder,
                               predictor=predictor)
            self.architecture = {'input_size': input_size,
                                 'output_size': output_size,
                                 'latent_size': latent_size,
                                 'encoder': encoder,
                                 'decoder': decoder,
                                 'predictor': predictor}
        elif self.network_type == 'feedforward':
            self.network = FeedForward(type=type,
                                       n_output=output_size,
                                       input_size=input_size,
                                       predictor=predictor)
            self.architecture = {'input_size': input_size,
                                 'output_size': output_size,
                                 'predictor': predictor}
        self.current_state = {'state_dict': None,
                              'optimizer': None,
                              'epoch': None,
                              'best_accuracy': 0,
                              'best_loss': np.inf}
        self.checkpoint = {'state_dict': None,
                           'optimizer': None,
                           'epoch': None,
                           'best_accuracy': 0,
                           'best_loss': np.inf}
        self.load_state = False
        self.total_losses = {'train': [], 'validation': []}
        self.vae_losses = {'train': [], 'validation': []}
        self.predictor_losses = {'train': [], 'validation': []}
        self.accuracies = {'train': [], 'validation': []}
        self.latent_means = {'train': None, 'validation': None}
        self.label_history = {'train': None, 'validation': None}
        self.n_epochs = 0

    def save(self, mode='best', path='./', fn='default'):
        """
        Saves the model to .ckpt file. Mode can be chosen to save either the
        best performing model or the most recent state of the model.

        Parameters:
            mode (str): Indicates which model to save (best or current)
            path (str): Path to directory where model will be stored
            fn (str): Filename of saved model
        """
        if fn == 'default':
            fn = '{}_vae.ckpt'.format(mode)

        if mode == 'best':
            torch.save({'state_dict': self.checkpoint['state_dict'],
                        'optimizer': self.checkpoint['optimizer'],
                        'architecture': self.architecture,
                        'epoch': self.checkpoint['epoch'],
                        'best_accuracy': self.checkpoint['best_accuracy'],
                        'best_loss': self.checkpoint['best_loss']},
                        os.path.join(path, fn))
        elif mode == 'current':
            torch.save({'state_dict': self.current_state['state_dict'],
                        'optimizer': self.current_state['optimizer'],
                        'architecture': self.architecture,
                        'epoch': self.current_state['epoch'],
                        'best_accuracy': self.current_state['best_accuracy'],
                        'best_loss': self.current_state['best_loss']},
                        os.path.join(path, fn))

    def load(self, checkpoint_path):
        """
        Loads model from saved .ckpt checkpoint state.

        Parameters:
            checkpoint_path (str): Path to checkpoint file (must include
                .ckpt filename)
        """
        loaded_checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        for key in self.checkpoint.keys():
            self.checkpoint[key] = loaded_checkpoint[key]
        architecture = loaded_checkpoint['architecture']
        self.network = VAE(latent_size=architecture['latent_size'],
                           type=type,
                           n_output=architecture['output_size'],
                           input_size=architecture['input_size'],
                           encoder=architecture['encoder'],
                           decoder=architecture['decoder'],
                           predictor=architecture['predictor'])
        self.load_state = True

    def train(self,
              data,
              labels,
              n_epochs,
              batch_size,
              val_split=0.2,
              optimizer='default',
              learning_rate=1e-4,
              store_best=True,
              criterion='acc',
              verbose=True):
        """
        Trains model. Feature data and labels (or targets) must be input
        separately. By default, 20% of training data is used for model
        validation and the Adam optimizer is used with a learning rate of
        1e-4. Data and model states are stored as Trainer attributes (defined
        during instantiation).

        Parameters:
            data (np.array): nxm feature matrix where n = # of samples and
                m = # of features

            labels (np.array): nx1 target matrix where n = # of samples. Only
                one target can be trained on at a time

            n_epochs (int): Number of epochs to train the model

            batch_size (int): Size of training and validation batches

            val_split (float): Fraction of data to be used for model validation

            optimizer (torch.optim): Model optimizer. Must be a pytorch
                optimizer object

            learning_rate (float): Learning rate of optimizer

            store_best (bool): Checks model performance and stores as an
                attribute if improved

            criterion (str): Criterion for model improvement (acc or loss)

            verbose (bool): Prints progress bar and metrics during training
        """
        if self.type == 'regressor':
            criterion = 'loss'
        self.n_samples = data.shape[0]
        self.batch_size = batch_size
        self.n_val_samples = int(val_split*self.n_samples)
        self.n_train_samples = self.n_samples - self.n_val_samples
        self.learning_rate = learning_rate
        torch.backends.cudnn.benchmark = True
        self.use_gpu = torch.cuda.is_available()

        latent_means_train = np.zeros((n_epochs, self.n_train_samples, self.latent_size))
        latent_means_val = np.zeros((n_epochs, self.n_val_samples, self.latent_size))
        label_history_train = np.zeros((n_epochs, self.n_train_samples))
        label_history_val = np.zeros((n_epochs, self.n_val_samples))

        labels = labels.reshape(-1, 1)
        concat_data = np.concatenate([data, labels], axis=1)

        train_indices = np.random.choice(np.arange(self.n_samples), \
                        size=self.n_train_samples, replace=False)
        val_indices = np.array(list(set(np.arange(self.n_samples)) - set(train_indices)))
        train_data = concat_data[train_indices, :]
        val_data = concat_data[val_indices, :]

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                                   shuffle=True, num_workers=0,
                                                   pin_memory=False)

        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
                                                 shuffle=True, num_workers=0,
                                                 pin_memory=False)

        if verbose:
            sys.stdout.write('\r'+'Data loaded...\n')

        self.network = self.network.double()
        if self.use_gpu:
            self.network.cuda()
        if optimizer == 'default':
            self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        else:
            self.optimizer = optimizer(self.network.parameters(), lr=learning_rate)
        self.optimizer.zero_grad()

        if self.load_state:
            self.network.load_state_dict(self.checkpoint['state_dict'])
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])
            self.n_epochs = self.checkpoint['epoch']

        if verbose:
            sys.stdout.write('\r'+'Network built...\n')

        for epoch in range(n_epochs):
            if verbose:
                epoch_counter = '[{}/{}]'.format(epoch+1, n_epochs)
                progress_bar = '['+'-'*50+']'
                sys.stdout.write("\r\033[K"+epoch_counter+' '+progress_bar)

            ### Train
            self.network.train()
            total_losses = []
            vae_losses = []
            predictor_losses = []
            train_accs = []
            latent_space = []
            train_labels = []

            for batch_idx, data in enumerate(train_loader):
                if verbose:
                    if batch_idx % 10 == 0:
                        n_hash = int(batch_idx*self.batch_size / self.n_samples * 50)+1
                        progress_bar = '['+'#'*n_hash+'-'*(50-n_hash)+']'
                        sys.stdout.write("\r\033[K"+epoch_counter+' '+progress_bar)
                if self.use_gpu:
                    data = data.cuda()
                input = torch.autograd.Variable(data[:, :-1])
                if self.type == 'classifier':
                    label = torch.autograd.Variable(data[:, -1]).long()
                elif self.type == 'regressor':
                    label = torch.autograd.Variable(data[:, -1])

                ### Forward and backward propagation
                if self.network_type == 'vae':
                    output, mu, logvar, prediction = self.network(input)
                    vae_l = vae_loss(input, output, mu, logvar)
                    vae_losses.append(vae_l.item())
                    mu = mu.data.cpu().numpy()
                elif self.network_type == 'feedforward':
                    prediction = self.network(input)
                    vae_l = 0
                    vae_losses.append(vae_l)
                    mu = np.zeros((prediction.shape[0], self.latent_size))

                if self.type == 'classifier':
                    pred_l = classifier_loss(label, prediction)
                    _, argmax = torch.max(prediction, 1)
                    acc = (argmax.squeeze() == label).float().mean()
                    acc = acc.data.cpu().numpy()
                elif self.type == 'regressor':
                    pred_l = regressor_loss(label, prediction)
                    acc = np.nan
                loss = vae_l + pred_l
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                total_losses.append(loss.item())
                predictor_losses.append(pred_l.item())
                train_accs.append(acc)
                latent_space.append(mu)
                train_labels.append(label.data.cpu().numpy())

            avg_train_loss = np.mean(total_losses)
            avg_train_vae_loss = np.mean(vae_losses)
            avg_train_pred_loss = np.mean(predictor_losses)
            avg_train_acc = np.float64(np.mean(train_accs))
            train_latent_space = np.concatenate(latent_space)
            train_labels = np.concatenate(train_labels)
            self.n_epochs += 1
            self.total_losses['train'].append(avg_train_loss)
            self.vae_losses['train'].append(avg_train_vae_loss)
            self.predictor_losses['train'].append(avg_train_pred_loss)
            self.accuracies['train'].append(avg_train_acc)
            latent_means_train[epoch, :, :] = train_latent_space
            label_history_train[epoch, :] = train_labels

            ### Validate
            self.network.eval()
            total_losses = []
            vae_losses = []
            predictor_losses = []
            val_accs = []
            latent_space = []
            val_labels = []

            for batch_idx, data in enumerate(val_loader):
                if verbose:
                    if batch_idx % 10 == 0:
                        n_hash = int((batch_idx*self.batch_size+ \
                            len(train_loader)*self.batch_size) / self.n_samples * 50)+1
                        progress_bar = '['+'#'*n_hash+'-'*(50-n_hash)+']'
                        sys.stdout.write("\r\033[K"+epoch_counter+' '+progress_bar)
                    elif batch_idx == len(val_loader)-1:
                        n_hash = int((batch_idx*self.batch_size+ \
                            len(train_loader)*self.batch_size) / self.n_samples * 50)+1
                        progress_bar = '['+'#'*n_hash+'-'*(50-n_hash)+']'
                        sys.stdout.write("\r\033[K"+epoch_counter+' '+progress_bar)
                if self.use_gpu:
                    data = data.cuda()
                input = torch.autograd.Variable(data[:, :-1])
                if self.type == 'classifier':
                    label = torch.autograd.Variable(data[:, -1]).long()
                elif self.type == 'regressor':
                    label = torch.autograd.Variable(data[:, -1])

                ### Forward propagation
                if self.network_type == 'vae':
                    output, mu, logvar, prediction = self.network(input)
                    vae_l = vae_loss(input, output, mu, logvar)
                    vae_losses.append(vae_l.item())
                    mu = mu.data.cpu().numpy()
                elif self.network_type == 'feedforward':
                    prediction = self.network(input)
                    vae_l = 0
                    vae_losses.append(vae_l)
                    mu = np.zeros((prediction.shape[0], self.latent_size))

                if self.type == 'classifier':
                    pred_l = classifier_loss(label, prediction)
                    _, argmax = torch.max(prediction, 1)
                    acc = (argmax.squeeze() == label).float().mean()
                    acc = acc.data.cpu().numpy()
                elif self.type == 'regressor':
                    pred_l = regressor_loss(label, prediction)
                    acc = np.nan
                loss = vae_l + pred_l

                total_losses.append(loss.item())
                predictor_losses.append(pred_l.item())
                val_accs.append(acc)
                latent_space.append(mu)
                val_labels.append(label.data.cpu().numpy())

            avg_val_loss = np.mean(total_losses)
            avg_val_vae_loss = np.mean(vae_losses)
            avg_val_pred_loss = np.mean(predictor_losses)
            avg_val_acc = np.float64(np.mean(val_accs))
            val_latent_space = np.concatenate(latent_space)
            val_labels = np.concatenate(val_labels)
            self.total_losses['validation'].append(avg_val_loss)
            self.vae_losses['validation'].append(avg_val_vae_loss)
            self.predictor_losses['validation'].append(avg_val_pred_loss)
            self.accuracies['validation'].append(avg_val_acc)
            latent_means_val[epoch, :, :] = val_latent_space
            label_history_val[epoch, :] = val_labels

            self.current_state['state_dict'] = self.network.state_dict()
            self.current_state['optimizer'] = self.optimizer.state_dict()
            self.current_state['epoch'] = self.n_epochs

            if store_best:
                if criterion == 'acc':
                    if avg_val_acc > self.checkpoint['best_accuracy']:
                        self.checkpoint['state_dict'] = self.network.state_dict()
                        self.checkpoint['optimizer'] = self.optimizer.state_dict()
                        self.checkpoint['epoch'] = self.n_epochs
                        self.checkpoint['best_accuracy'] = avg_val_acc
                    else:
                        pass
                elif criterion == 'loss':
                    if avg_val_loss < self.checkpoint['best_loss']:
                        self.checkpoint['state_dict'] = self.network.state_dict()
                        self.checkpoint['optimizer'] = self.optimizer.state_dict()
                        self.checkpoint['epoch'] = self.n_epochs
                        self.checkpoint['best_loss'] = avg_val_loss

            if verbose:
                if self.type == 'classifier':
                    loss_statement = ' train_loss={} val_loss={} train_acc={} val_acc={}' \
                                    .format(round(avg_train_loss, 2), \
                                round(avg_val_loss, 2), round(avg_train_acc, 2), round(avg_val_acc, 2))
                elif self.type == 'regressor':
                    loss_statement = ' train_loss={} val_loss={}'.format(round(avg_train_loss, 2),\
                                    round(avg_val_loss, 2))
                sys.stdout.write("\r\033[K"+epoch_counter+' '+progress_bar+loss_statement+'\n')

        self.latent_means['train'] = latent_means_train
        self.latent_means['validation'] = latent_means_val
        self.label_history['train'] = label_history_train
        self.label_history['validation'] = label_history_val

    def predict(self, data):
        """
        Predicts output given a set of input data for a fully trained model.

        Parameters:
            data (np.array): nxm feature matrix where n = # of samples and
                m = # of features

        Returns:
            prediction (np.array): nx1 prediction matrix where n = # of samples
        """
        self.network = self.network.double()
        self.network.eval()
        input = torch.autograd.Variable(torch.from_numpy(data))
        if self.type == 'classifier':
            if self.network_type == 'vae':
                output, mu, logvar, prediction = self.network(input)
            elif self.network_type == 'feedforward':
                prediction = self.network(input)
            _, prediction = torch.max(prediction, 1)
            return prediction.data.cpu().numpy()
        elif self.type == 'regressor':
            if self.network_type == 'vae':
                output, mu, logvar, prediction = self.network(input)
            elif self.network_type == 'feedforward':
                prediction = self.network(input)
            return prediction.data.cpu().numpy()

class CNNTrainer():
    def __init__(self,
                 input_size,
                 output_size,
                 args):
        self.args = args
        self.model = self.args.model
        self.network_module = importlib.import_module('cnns4qspr.se3cnn_v3.networks.{:s}.{:s}'.format(self.model, self.model))
        self.input_size = input_size
        self.output_size = output_size
        self.network = self.network_module.network(n_input=self.input_size,
                                                   args=self.args)
        self.predictor_type = self.args.predictor_type
        if self.predictor_type == 'regressor':
            self.output_size = 1
        self.build_predictor(predictor_class=self.args.predictor,
                             type=self.predictor_type,
                             input_size=self.network.output_size,
                             output_size=self.output_size,
                             latent_size=self.args.latent_size,
                             predictor=self.args.predictor_layers,
                             encoder=self.args.encoder_layers,
                             decoder=self.args.decoder_layers)
        self.blocks = [block for block in self.network.blocks if block is not None]
        self.blocks.append(self.predictor)
        self.network.blocks = nn.Sequential(*[block for block in self.blocks if block is not None])
        self.history = {'train_loss': [],
                        'val_loss': [],
                        'train_predictor_loss': [],
                        'val_predictor_loss': [],
                        'train_vae_loss': [],
                        'val_vae_loss': [],
                        'train_acc': [],
                        'val_acc': []}
        self.current_state = {'state_dict': None,
                              'optimizer': None,
                              'epoch': None,
                              'best_acc': np.nan,
                              'best_loss': np.nan,
                              'history': self.history,
                              'input_size': self.input_size,
                              'output_size': self.output_size,
                              'predictor_architecture': self.predictor_architecture,
                              'args': self.args}
        self.best_state = {'state_dict': None,
                           'optimizer': None,
                           'epoch': None,
                           'best_acc': 0,
                           'best_loss': np.inf,
                           'history': None,
                           'input_size': self.input_size,
                           'output_size': self.output_size,
                           'predictor_architecture': self.predictor_architecture,
                           'args': self.args}
        self.n_epochs = 0
        self.best_acc = 0
        self.best_loss = np.inf
        self.loaded = False

    def build_predictor(self,
                        predictor_class,
                        type,
                        input_size,
                        output_size,
                        latent_size,
                        predictor,
                        encoder,
                        decoder):
        predictor = list_parser(predictor)
        encoder = list_parser(encoder)
        decoder = list_parser(decoder)
        if predictor_class == 'feedforward':
            self.predictor = FeedForward(type=type,
                                         n_output=output_size,
                                         input_size=input_size,
                                         predictor=predictor)
            self.predictor_architecture = {'output_size': self.output_size,
                                           'input_size': input_size,
                                           'predictor': self.predictor.predictor}
        elif predictor_class == 'vae':
            self.predictor = VAE(latent_size=latent_size,
                                 type=type,
                                 n_output=output_size,
                                 input_size=input_size,
                                 encoder=encoder,
                                 decoder=decoder,
                                 predictor=predictor)
            self.predictor_architecture = {'output_size': self.output_size,
                                           'input_size': input_size,
                                           'latent_size': self.args.latent_size,
                                           'encoder': self.predictor.encoder,
                                           'decoder': self.predictor.decoder,
                                           'predictor': self.predictor.predictor}
            self.network.blocks[5].register_forward_hook(self.vae_hook)

    def vae_hook(self, module, input_, output):
        global cnn_out
        cnn_out = output

    def save(self, mode='best', save_fn='best_model.ckpt', save_path='checkpoints'):
        os.makedirs(save_path, exist_ok=True)
        if os.path.splitext(save_fn)[1] == '':
            save_fn += '.ckpt'
        ckpt_save = {}
        if mode == 'best':
            for key in self.best_state.keys():
                ckpt_save[key] = self.best_state[key]
            torch.save(ckpt_save, os.path.join(save_path, save_fn))
        elif mode == 'current':
            for key in self.current_state.keys():
                ckpt_save[key] = self.current_state[key]
            torch.save(ckpt_save, os.path.join(save_path, save_fn))

    def load(self, checkpoint_path):
        loaded_checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        for key in self.current_state.keys():
            self.current_state[key] = loaded_checkpoint[key]

        self.args = self.current_state['args']
        self.model = self.args.model
        self.network_module = importlib.import_module('cnns4qspr.se3cnn_v3.networks.{:s}.{:s}'.format(self.model, self.model))
        self.input_size = self.current_state['input_size']
        self.output_size = self.current_state['output_size']
        self.network = self.network_module.network(n_input=self.input_size,
                                                   args=self.args)
        self.predictor_type = self.args.predictor_type
        self.predictor_architecture = self.current_state['predictor_architecture']
        self.build_predictor(predictor_class=self.args.predictor,
                             type=self.predictor_type,
                             input_size=self.network.output_size,
                             output_size=self.output_size,
                             latent_size=self.args.latent_size,
                             predictor=self.args.predictor_layers,
                             encoder=self.args.encoder_layers,
                             decoder=self.args.decoder_layers)
        self.blocks = [block for block in self.network.blocks if block is not None]
        self.blocks.append(self.predictor)
        self.network.blocks = nn.Sequential(*[block for block in self.blocks if block is not None])

        self.history = self.current_state['history']
        self.n_epochs = self.current_state['epoch']
        self.best_acc = self.current_state['best_acc']
        self.best_loss = self.current_state['best_loss']
        self.loaded = True

    def train(self,
              data,
              targets,
              epochs='args',
              val_split=0.2,
              store_best=True,
              save_best=False,
              save_fn='best_model.ckpt',
              save_path='checkpoints',
              verbose=True):
        torch.backends.cudnn.benchmark = True
        use_gpu = torch.cuda.is_available()
        if epochs == 'args':
            epochs = self.args.training_epochs

        ### Configure settings for model type
        if self.predictor_type == 'classifier':
            predictor_loss = classifier_loss
            store_criterion = 'val_acc'
        elif self.predictor_type == 'regressor':
            predictor_loss = regressor_loss
            store_criterion = 'val_loss'

        ### Load data and format for training
        self.n_samples = data.shape[0]
        self.batch_size = self.args.batch_size
        self.n_val_samples = int(val_split*self.n_samples)
        self.n_train_samples = self.n_samples - self.n_val_samples

        targets = targets.squeeze()
        data = np.expand_dims(data, 5)
        data[:,0,0,0,0,0] = targets

        train_indices = np.random.choice(np.arange(self.n_samples), \
                        size=self.n_train_samples, replace=False)
        val_indices = np.array(list(set(np.arange(self.n_samples)) - set(train_indices)))
        train_data = data[train_indices, :]
        val_data = data[val_indices, :]

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.args.batch_size,
                                                   shuffle=True, num_workers=0,
                                                   pin_memory=False)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=self.args.batch_size,
                                                 shuffle=True, num_workers=0,
                                                 pin_memory=False)

        if verbose:
            sys.stdout.write('\r'+'Data loaded...\n')

        ### Set up network model and optimizer
        if use_gpu:
            self.network.cuda()

        self.param_groups = get_param_groups(self.network, self.args)
        self.optimizer = Adam(self.param_groups, lr=self.args.initial_lr)
        self.optimizer.zero_grad()

        if self.loaded:
            self.network.load_state_dict(self.current_state['state_dict'])
            self.optimizer.load_state_dict(self.current_state['optimizer'])

        if verbose:
            sys.stdout.write('\r'+'Network built...\n')

        ### Train
        for epoch in range(epochs):
            self.optimizer, _ = lr_scheduler_exponential(self.optimizer,
                                                         epoch+self.n_epochs,
                                                         self.args.initial_lr,
                                                         self.args.lr_decay_start,
                                                         self.args.lr_decay_base,
                                                         verbose=False)

            ### Batch Loop
            self.network.train()
            total_losses = []
            vae_losses = []
            predictor_losses = []
            accs = []

            for batch_idx, data in enumerate(train_loader):
                print('Train {}_{}'.format(epoch, batch_idx))
                if use_gpu:
                    data = data.cuda()
                inputs = data[:,:,:,:,:,0]
                targets = data[:,0,0,0,0,0]

                x = torch.autograd.Variable(inputs)
                y = torch.autograd.Variable(targets)

                # Forward and backward propagation
                if self.args.predictor == 'feedforward':
                    predictions = self.network(x)
                    pred_l = predictor_loss(y, predictions)
                    vae_l = torch.Tensor([np.nan])
                    loss = pred_l
                elif self.args.predictor == 'vae':
                    vae_out, mu, logvar, predictions = self.network(x)
                    vae_in = torch.autograd.Variable(cnn_out)
                    pred_l = predictor_loss(y, predictions)
                    vae_l = vae_loss(vae_in, vae_out, mu, logvar)
                    loss = vae_l + pred_l

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                _, argmax = torch.max(predictions, 1)
                acc = (argmax.squeeze() == targets).float().mean()

                total_losses.append(loss.item())
                predictor_losses.append(pred_l.item())
                vae_losses.append(vae_l.item())
                accs.append(acc.item())

            self.history['train_loss'].append(np.mean(total_losses))
            self.history['train_predictor_loss'].append(np.mean(predictor_losses))
            self.history['train_vae_loss'].append(np.mean(vae_losses))
            self.history['train_acc'].append(np.mean(accs))
            self.n_epochs += 1

            ### Validation Loop
            self.network.eval()
            total_losses = []
            vae_losses = []
            predictor_losses = []
            accs = []

            for batch_idx, data in enumerate(val_loader):
                print('Val {}_{}'.format(epoch, batch_idx))
                if use_gpu:
                    data = data.cuda()
                inputs = data[:,:,:,:,:,0]
                targets = data[:,0,0,0,0,0]

                x = torch.autograd.Variable(inputs)
                y = torch.autograd.Variable(targets)

                # Forward prop
                if self.args.predictor == 'feedforward':
                    predictions = self.network(x)
                    pred_l = predictor_loss(y, predictions)
                    vae_l = torch.Tensor([np.nan])
                    loss = pred_l
                elif self.args.predictor == 'vae':
                    vae_out, mu, logvar, predictions = self.network(x)
                    vae_in = torch.autograd.Variable(cnn_out)
                    pred_l = predictor_loss(y, predictions)
                    vae_l = vae_loss(vae_in, vae_out, mu, logvar)
                    loss = vae_l + pred_l

                _, argmax = torch.max(predictions, 1)
                acc = (argmax.squeeze() == targets).float().mean()

                total_losses.append(loss.item())
                predictor_losses.append(pred_l.item())
                vae_losses.append(vae_l.item())
                accs.append(acc.item())

            self.history['val_loss'].append(np.mean(total_losses))
            self.history['val_predictor_loss'].append(np.mean(predictor_losses))
            self.history['val_vae_loss'].append(np.mean(vae_losses))
            self.history['val_acc'].append(np.mean(accs))

            self.current_state['state_dict'] = self.network.state_dict()
            self.current_state['optimizer'] = self.optimizer.state_dict()
            self.current_state['epoch'] = self.n_epochs

            if store_best:
                if store_criterion == 'val_acc':
                    if np.mean(accs) > self.best_acc:
                        self.best_acc = np.mean(accs)
                        self.best_state['state_dict'] = copy.deepcopy(self.network.state_dict())
                        self.best_state['optimizer'] = copy.deepcopy(self.optimizer.state_dict())
                        self.best_state['epoch'] = self.n_epochs
                        self.best_state['best_acc'] = self.best_acc
                        self.best_state['history'] = copy.deepcopy(self.history)
                    else:
                        pass
                    if save_best:
                        self.save(mode='best', save_fn=save_fn, save_path=save_path)
                elif store_criterion == 'val_loss':
                    if np.mean(total_losses) < self.best_loss:
                        self.best_loss = np.mean(total_losses)
                        self.best_state['state_dict'] = copy.deepcopy(self.network.state_dict())
                        self.best_state['optimizer'] = copy.deepcopy(self.optimizer.state_dict())
                        self.best_state['epoch'] = self.n_epochs
                        self.best_state['best_loss'] = self.best_loss
                        self.best_state['history'] = copy.deepcopy(self.history)
                    else:
                        pass
                    if save_best:
                        self.save(mode='best', save_fn=save_fn, save_path=save_path)
