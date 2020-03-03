import torch
import torch.nn as nn
import torch.nn.functional as F

from se3cnn.image.gated_block import GatedBlock
from se3cnn.image.norm_block import NormBlock
from se3cnn.image.batchnorm import SE3BatchNorm
from se3cnn.image.convolution import SE3Convolution
from se3cnn.image import kernel

from se3cnn.non_linearities.norm_activation import NormRelu, NormSoftplus
from se3cnn.non_linearities.scalar_activation import ScalarActivation
from se3cnn.image.gated_activation import GatedActivation


class Merge(nn.Module):
    def forward(self, x1, x2):
        return torch.cat([x1, x2], dim=1)


class AvgSpacial(nn.Module):
    def forward(self, inp):
        return inp.view(inp.size(0), inp.size(1), -1).mean(-1)


class ResBlock(nn.Module):
    def __init__(self, channels_in, channels_out, size=3, stride=1,
                 downsample_by_pooling=False,
                 conv_dropout_p=None):
        super().__init__()

        channels = [channels_in] + channels_out

        self.layers = []
        conv_stride = 1 if downsample_by_pooling else stride
        for i in range(len(channels) - 1):
            self.layers += [
                nn.BatchNorm3d(channels[i]),
                nn.Conv3d(channels[i], channels[i + 1],
                          kernel_size=size,
                          padding=size // 2,
                          stride=conv_stride if i == 0 else 1,
                          bias=False),
                # nn.BatchNorm3d(channels[i + 1])
            ]
            if conv_dropout_p is not None:
                self.layers.append(nn.Dropout3d(p=conv_dropout_p, inplace=True))
            if downsample_by_pooling and i == 0 and stride > 1:
                self.layers.append(nn.AvgPool3d(kernel_size=size,
                                                padding=size//2,
                                                stride=stride))
            if (i + 1) < len(channels) - 1:
                self.layers += [nn.ReLU(inplace=True)]
        self.layers = nn.Sequential(*self.layers)

        self.shortcut = None
        if len(channels_out) > 1:
            if channels_in == channels_out[-1] and stride == 1:
                self.shortcut = lambda x: x
            else:
                self.shortcut = [
                    nn.BatchNorm3d(channels[0]),
                    nn.Conv3d(channels[0], channels[-1],
                              kernel_size=1,
                              padding=0,
                              stride=conv_stride,
                              bias=False),
                    # nn.BatchNorm3d(channels[-1])
                    ]
                if conv_dropout_p is not None:
                    self.shortcut.append(
                        nn.Dropout3d(p=conv_dropout_p, inplace=True))
                if downsample_by_pooling and stride > 1:
                    self.shortcut.append(nn.AvgPool3d(kernel_size=size,
                                                      padding=size // 2,
                                                      stride=stride))
                self.shortcut = nn.Sequential(*self.shortcut)

        self.activation = nn.ReLU(inplace=True)

        # initialize
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.xavier_normal(module.weight.data)
            elif isinstance(module, nn.BatchNorm3d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x):
        out = self.layers(x)
        if self.shortcut is not None:
            out += self.shortcut(x)
        out = self.activation(out)
        # print(out.shape, "ResBlock")
        return out


class SE3GatedResBlock(nn.Module):
    def __init__(self, in_repr, out_reprs,
                 size=3,
                 stride=1,
                 radial_window=None,
                 batch_norm_momentum=0.01,
                 normalization="batch",
                 capsule_dropout_p=0.1,
                 scalar_gate_activation=(F.relu, F.sigmoid),
                 downsample_by_pooling=False):
        super().__init__()

        reprs = [in_repr] + out_reprs

        self.layers = []
        single_layer = len(out_reprs) == 1
        conv_stride = 1 if downsample_by_pooling else stride
        for i in range(len(reprs) - 1):
            # No activation in last block
            activation = scalar_gate_activation
            if i == (len(reprs) - 2) and not single_layer:
                activation = None
            self.layers.append(
                GatedBlock(reprs[i], reprs[i + 1],
                           size=size, padding=size//2,
                           stride=conv_stride if i == 0 else 1,
                           activation=activation,
                           radial_window=radial_window,
                           batch_norm_momentum=batch_norm_momentum,
                           normalization=normalization,
                           smooth_stride=False,
                           capsule_dropout_p=capsule_dropout_p))
            if downsample_by_pooling and i == 0 and stride > 1:
                self.layers.append(nn.AvgPool3d(kernel_size=size,
                                                padding=size//2,
                                                stride=stride))
        self.layers = nn.Sequential(*self.layers)

        self.shortcut = None
        self.activation = None
        # Add shortcut if number of layers is larger than 1
        if not single_layer:
            # Use identity is input and output reprs are identical
            if in_repr == out_reprs[-1] and stride == 1:
                self.shortcut = lambda x: x
            else:
                self.shortcut = []
                self.shortcut.append(
                    GatedBlock(reprs[0], reprs[-1],
                               size=size, padding=size//2,
                               stride=conv_stride,
                               activation=None,
                               radial_window=radial_window,
                               batch_norm_momentum=batch_norm_momentum,
                               normalization=normalization,
                               smooth_stride=False,
                               capsule_dropout_p=capsule_dropout_p))
                if downsample_by_pooling and stride > 1:
                    self.shortcut.append(nn.AvgPool3d(kernel_size=size,
                                                      padding=size//2,
                                                      stride=stride))
                self.shortcut = nn.Sequential(*self.shortcut)

            self.activation = GatedActivation(
                repr_in=reprs[-1],
                size=size,
                radial_window=radial_window,
                batch_norm_momentum=batch_norm_momentum,
                normalization=normalization)

    def forward(self, x):
        out = self.layers(x)
        if self.shortcut is not None:
            out += self.shortcut(x)
            out = self.activation(out)
        # print(out.shape, "SE3GatedResBlock")
        return out


class SE3NormResBlock(nn.Module):
    def __init__(self, in_repr, out_reprs,
                 size=3,
                 stride=1,
                 radial_window=None,
                 batch_norm_momentum=0.01,
                 normalization="batch",
                 capsule_dropout_p=0.1,
                 scalar_activation=F.relu,
                 activation_bias_min=0.5,
                 activation_bias_max=2,
                 downsample_by_pooling=False):
        super().__init__()

        reprs = [in_repr] + out_reprs

        self.layers = []
        single_layer = len(out_reprs) == 1
        conv_stride = 1 if downsample_by_pooling else stride
        for i in range(len(reprs) - 1):
            # No activation in last block
            activation = scalar_activation
            if i == (len(reprs) - 2) and not single_layer:
                activation = None
            self.layers.append(
                NormBlock(reprs[i], reprs[i + 1],
                          size=size, padding=size//2,
                          stride=conv_stride if i == 0 else 1,
                          activation=activation,
                          radial_window=radial_window,
                          normalization=normalization,
                          batch_norm_momentum=batch_norm_momentum,
                          capsule_dropout_p=capsule_dropout_p))
            if downsample_by_pooling and i == 0 and stride > 1:
                self.layers.append(nn.AvgPool3d(kernel_size=size,
                                                padding=size//2,
                                                stride=stride))
        self.layers = nn.Sequential(*self.layers)

        self.shortcut = None
        self.activation = None
        # Add shortcut if number of layers is larger than 1
        if not single_layer:
            # Use identity is input and output reprs are identical
            if in_repr == out_reprs[-1] and stride == 1:
                self.shortcut = lambda x: x
            else:
                self.shortcut = []
                self.shortcut.append(
                    NormBlock(reprs[0], reprs[-1],
                              size=size, padding=size//2,
                              stride=conv_stride,
                              activation=None,
                              activation_bias_min=activation_bias_min,
                              activation_bias_max=activation_bias_max,
                              radial_window=radial_window,
                              normalization=normalization,
                              batch_norm_momentum=batch_norm_momentum,
                              capsule_dropout_p=capsule_dropout_p))
                if downsample_by_pooling and stride > 1:
                    self.shortcut.append(nn.AvgPool3d(kernel_size=size,
                                                      padding=size//2,
                                                      stride=stride))
                self.shortcut = nn.Sequential(*self.shortcut)

            capsule_dims = [2 * n + 1 for n, mul in enumerate(out_reprs[-1]) for i in
                            range(mul)]  # list of capsule dimensionalities
            self.activation = NormSoftplus(capsule_dims,
                                           scalar_act=scalar_activation,
                                           bias_min=activation_bias_min,
                                           bias_max=activation_bias_max)

    def forward(self, x):
        out = self.layers(x)
        if self.shortcut is not None:
            out += self.shortcut(x)
            out = self.activation(out)
        return out


class OuterBlock(nn.Module):
    def __init__(self, in_repr, out_reprs, res_block, size=3, stride=1, **kwargs):
        super().__init__()

        reprs = [[in_repr]] + out_reprs

        self.layers = []
        for i in range(len(reprs) - 1):
            self.layers.append(
                res_block(reprs[i][-1], reprs[i+1],
                          size=size,
                          stride=stride if i == 0 else 1,
                          **kwargs)
            )
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        out = self.layers(x)
        # print(out.shape, "OuterBlock")
        return out


class ResNet(nn.Module):
    def __init__(self, *blocks):
        super().__init__()
        self.blocks = nn.Sequential(*[block for block in blocks if block is not None])

    def forward(self, x):
        return self.blocks(x)

class VAE(nn.Module):
    def __init__(self, in_repr, latent_size, out_repr):
        super().__init__()

        self.in_repr = in_repr
        self.layers = []
        self.layers.append(nn.Linear(in_repr, 128))
        self.layers.append(nn.Linear(128, latent_size))
        self.layers.append(nn.Linear(128, latent_size))
        self.layers.append(nn.Linear(latent_size, 128))
        self.layers.append(nn.Linear(128, in_repr))
        self.layers = nn.Sequential(*self.layers)

        self.predict = []
        self.predict.append(nn.Linear(latent_size, 128))
        self.predict.append(nn.Linear(128, 128))
        self.predict.append(nn.Linear(128, out_repr))
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
        mu, logvar = self.encode(x.view(-1, self.in_repr))
        z = self.reparameterize(mu, logvar)
        predictions = self.predict(mu)
        return self.decode(z), mu, logvar, predictions

    def get_latent_space(self, cnn_output):
        mu, logvar = self.encode(cnn_output.view(-1, self.in_repr))
        z = self.reparameterize(mu, logvar)
        return z

class NonlinearityBlock(nn.Module):
    ''' wrapper around GatedBlock and NormBlock, selects based on string SE3Nonlniearity '''
    def __init__(self, features_in, features_out, SE3_nonlinearity, **kwargs):
        super().__init__()
        if SE3_nonlinearity == 'gated':
            conv_block = GatedBlock
        elif SE3_nonlinearity == 'norm':
            conv_block = NormBlock
        else:
            raise NotImplementedError('unknown SE3_nonlinearity')
        self.conv_block = conv_block(features_in, features_out, **kwargs)
    def forward(self, x):
        return self.conv_block(x)


class SkipSumBlock(nn.Module):
    ''' skip connection module for UNets
        takes a feature map from the encoder pathway and merges it with the decoder feature map by summation
        the encoder feature map is convolved before being added to allow for aligned features
        it is assumed that the shape of both feature maps is equal
    '''
    def __init__(self, features, **common_params):
        super(SkipSumBlock, self).__init__()
        raise NotImplementedError('TODO: nonlinearity only after summation')
        # self.skip_conv = NonlinearityBlock(features, features, **common_params)
    def forward(self, enc, dec):
        raise NotImplementedError('TODO: nonlinearity only after summation')
        # assert enc.shape == dec.shape
        # enc_res = self.skip_conv(enc)
        # return enc_res + dec
