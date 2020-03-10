import torch.nn as nn

from functools import partial

from cnns4qspr.se3cnn_v3.util.arch_blocks import *
from se3cnn.image import kernel


class network(ResNet):
    def __init__(self,
                 n_input=1,
                 n_output=3):

        features = [[[(2,  2,  2,  2)]],          #  32 channels
                    [[(2,  2,  2,  2)] * 2] * 3,  #  32 channels
                    [[(4,  4,  4,  4)] * 2] * 4,  #  64 channels
                    [[(8,  8,  8,  8)] * 2] * 6,  # 128 channels
                    # [[(8, 8, 8, 8)] * 2] * 2 + [[(8, 8, 8, 8), (128, 0, 0, 0)]]]  # 256 channels
                    [[(16, 16, 16, 16)] * 2] * 2 + [[(16, 16, 16, 16), (256, 0, 0, 0)]]]  # 256 channels
        common_params = {
            'radial_window': partial(kernel.gaussian_window_wrapper,
                                     mode='compromise', border_dist=0, sigma=0.6),
            'batch_norm_momentum': 0.01,
            # TODO: probability needs to be adapted to capsule order
            'capsule_dropout_p': 0.1,  # drop probability of whole capsules
            'normalization': 'batch',
            'downsample_by_pooling': True,
        }
        res_block = SE3GatedResBlock
        global OuterBlock
        OuterBlock = partial(OuterBlock,
                             res_block=partial(res_block, **common_params))
        super().__init__(
            OuterBlock((n_input,),          features[0], size=7),
            OuterBlock(features[0][-1][-1], features[1], size=3, stride=1),
            OuterBlock(features[1][-1][-1], features[2], size=3, stride=2),
            OuterBlock(features[2][-1][-1], features[3], size=3, stride=2),
            OuterBlock(features[3][-1][-1], features[4], size=3, stride=2),
            AvgSpacial()
        )
