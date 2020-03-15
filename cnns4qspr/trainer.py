import os
import sys
import torch
# import torch.nn as nn
import torch.utils.data
import torch.optim as optim
# import torch.nn.functional as F
import numpy as np
from cnns4qspr.se3cnn_v3.util.arch_blocks import VAE, FeedForward
from cnns4qspr.se3cnn_v3.util.losses import vae_loss, classifier_loss, regressor_loss

class Trainer():
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
        loaded_checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        for key in self.checkpoint.keys():
            self.checkpoint[key] = loaded_checkpoint[key]
        architecture = loaded_checkpoint['architecture']
        self.network = VAE(latent_size=architecture['latent_size'],
                           type=type,
                           n_output=architecture['output_size'],
                           input_size=architecture['input_size'],
                           encoder_depth=architecture['encoder_depth'],
                           encoder_size=architecture['encoder_size'],
                           decoder_depth=architecture['decoder_depth'],
                           decoder_size=architecture['decoder_size'],
                           predictor_depth=architecture['predictor_depth'],
                           predictor_size=architecture['predictor_size'])
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
                # print('---PREDICTION---')
                # print(prediction)
                # print('---TARGET---')
                # print(label)

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
            self.total_losses['validation'].append(avg_train_loss)
            self.vae_losses['validation'].append(avg_train_vae_loss)
            self.predictor_losses['validation'].append(avg_train_pred_loss)
            self.accuracies['validation'].append(avg_train_acc)
            latent_means_val[epoch, :, :] = val_latent_space
            label_history_val[epoch, :] = val_labels

            self.current_state['state_dict'] = self.network.state_dict()
            self.current_state['optimizer'] = self.optimizer.state_dict()
            self.current_state['epoch'] = self.n_epochs

            if store_best:
                if criterion == 'acc':
                    print(avg_val_acc, self.checkpoint['best_accuracy'])
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
        self.label_history['val'] = label_history_val

    def predict(self, data):
        self.network.eval()
        input = torch.autograd.Variable(torch.from_numpy(data))
        output, mu, logvar, prediction = self.network(input)
        return prediction
