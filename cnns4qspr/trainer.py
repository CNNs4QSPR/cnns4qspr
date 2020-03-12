import torch
import torch.nn as nn
import torch.nn.functional as F
from cnns4qspr.se3cnn_v3.util.arch_blocks import VAE

class VAETrainer():
    def __init__(self,
                 latent_size,
                 type='classifier',
                 n_output=3,
                 input_size=256,
                 encoder_depth=1,
                 encoder_size=[128],
                 decoder_depth=1,
                 decoder_size=[128],
                 predictor_depth=3,
                 predictor_size=[128,128]):
        self.network = VAE(latent_size=latent_size,
                           type=type,
                           n_output=n_output,
                           input_size=input_size,
                           encoder_depth=encoder_depth,
                           encoder_size=encoder_size,
                           decoder_depth=decoder_depth,
                           decoder_size=decoder_size,
                           predictor_depth=predictor_depth,
                           predictor_size=predictor_size
                           )
    def train(self,
              data,
              n_epochs,
              batch_size,
              mode='train',
              validation_split=0.2,
              optimizer='default',
              )
