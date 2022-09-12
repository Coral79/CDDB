from models.cnn.abstract import AbstractCNN
from models.cnn.inception import InceptionV3
from models.cnn.senet import legacy_seresnet18 as seresnet18
from models.cnn.resnet import (
    resnet18, resnet34, resnet50, resnext50_32x4d, wide_resnet50_2
)
from models.cnn.resnet_scs import resnet18_scs, resnet18_scs_avg, resnet18_scs_max
from models.cnn.vgg import vgg16_bn, vgg16
from models.cnn.resnet_rebuffi import CifarResNet as rebuffi

