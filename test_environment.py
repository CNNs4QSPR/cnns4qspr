import os
from se3cnn_v3.util import util

model = util.load_model('SE3ResNet34Small', 20, 'trial_8_best.ckpt')
print(model)
