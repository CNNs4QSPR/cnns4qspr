import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self,
                 latent_size,
                 type='classifier',
                 n_output=3,
                 input_size=256,
                 encoder=[128],
                 decoder=[128],
                 predictor=[128,128]):
        super().__init__()

        self.input_size = input_size
        self.output_size = n_output
        self.latent_size = latent_size
        self.type = type
        self.info = {'type': type,
                     'latent_size': latent_size,
                     'output_nodes': n_output,
                     'input_size': input_size,
                     'encoder': encoder,
                     'decoder': decoder,
                     'predictor': predictor}
        self.encoder_depth = len(encoder)
        self.decoder_depth = len(decoder)
        self.predictor_depth = len(predictor)+1
        self.encoder_nodes = [input_size] + encoder
        self.decoder_nodes = [latent_size] + decoder + [input_size]
        self.predictor_nodes = [latent_size] + predictor + [n_output]

        self.layers = []
        for i in range(self.encoder_depth):
            self.layers.append(nn.Linear(self.encoder_nodes[i], self.encoder_nodes[i+1]))
            if i+1 == self.encoder_depth:
                self.layers.append(nn.Linear(self.encoder_nodes[-1], latent_size))
                self.layers.append(nn.Linear(self.encoder_nodes[-1], latent_size))
        for i in range(self.decoder_depth+1):
            self.layers.append(nn.Linear(self.decoder_nodes[i], self.decoder_nodes[i+1]))
        self.layers = nn.Sequential(*self.layers)

        self.predict = []
        for i in range(self.predictor_depth):
            self.predict.append(nn.Linear(self.predictor_nodes[i], self.predictor_nodes[i+1]))
        self.predict = nn.Sequential(*self.predict)

    def encode(self, x):
        h1 = F.relu(self.layers[0](x))
        return self.layers[1](h1), self.layers[2](h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.layers[3](z))
        return torch.sigmoid(self.layers[4](h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_size))
        z = self.reparameterize(mu, logvar)
        predictions = self.predict(mu)
        return self.decode(z), mu, logvar, predictions

    def sample_latent_space(self, cnn_output):
        mu, logvar = self.encode(cnn_output.view(-1, self.input_size))
        z = self.reparameterize(mu, logvar)
        return z

class FeedForward(nn.Module):
    def __init__(self,
                 type='classifier',
                 n_output=3,
                 input_size=256,
                 predictor=[128,128]):
        super().__init__()

        self.input_size = input_size
        self.output_size = n_output
        self.type = type
        self.info = {'type': type,
                     'output_nodes': n_output,
                     'input_size': input_size,
                     'predictor': predictor}
        self.predictor_depth = len(predictor)+1
        self.predictor_nodes = [input_size] + predictor + [n_output]

        self.predict = []
        for i in range(self.predictor_depth):
            self.predict.append(nn.Linear(self.predictor_nodes[i], self.predictor_nodes[i+1]))
        self.predict = nn.Sequential(*self.predict)

    def forward(self, x):
        predictions = self.predict(x)
        return predictions
