import torch.nn as nn

from functools import partial

from experiments.util.arch_blocks import *
from se3cnn import kernel


class network(ResNet):
    def __init__(self,
                 n_input,
                 n_output,
                 args):

        # features = [[[(4,  4,  4,  4)] * 1],  # 64 channels
        #             [[(4,  4,  4,  4)] * 1],  # 64 channels
        #             [[(8,  8,  8,  8)] * 1],  # 128 channels
        #             [[(16, 16, 16, 16)] * 1],  # 256 channels
        #             [[(256,)]]]
        features = [[[(2, 2, 2, 2)] * 1],  #  32 channels
                    [[(2, 2, 2, 2)] * 1],  #  32 channels
                    [[(3, 3, 3, 3)] * 1],  #  64 channels
                    [[(4, 4, 4, 4)] * 1],  # 128 channels
                    [[(128,)]]]
        common_params = {
            'radial_window': partial(kernel.gaussian_window_wrapper,
                                     mode=args.bandlimit_mode, border_dist=0, sigma=0.6),
            'batch_norm_momentum': 0.01,
            # TODO: probability needs to be adapted to capsule order
            'capsule_dropout_p': args.p_drop_conv,  # drop probability of whole capsules
            'downsample_by_pooling': args.downsample_by_pooling,
            'normalization': args.normalization
        }
        if args.SE3_nonlinearity == 'gated':
            res_block = SE3GatedResBlock
        else:
            res_block = SE3NormResBlock
        global OuterBlock
        OuterBlock = partial(OuterBlock,
                             res_block=partial(res_block, **common_params))
        super().__init__(
            OuterBlock((n_input,),          features[0], size=args.kernel_size),
            OuterBlock(features[0][-1][-1], features[1], size=args.kernel_size, stride=1),
            OuterBlock(features[1][-1][-1], features[2], size=args.kernel_size, stride=2),
            OuterBlock(features[2][-1][-1], features[3], size=args.kernel_size, stride=2),
            OuterBlock(features[3][-1][-1], features[4], size=args.kernel_size, stride=2),
            AvgSpacial(),
            nn.Dropout(p=args.p_drop_fully, inplace=True) if args.p_drop_fully is not None else None,
            nn.Linear(features[4][-1][-1][0], n_output)
            )