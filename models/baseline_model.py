# encoding: utf-8
"""
@author:  liaoxingyu
@contact: xyliao1993@qq.com 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import mxnet as mx
from mxnet.gluon import nn

from .resnet import ResNetV1


class ResNetBuilder(nn.HybridBlock):
    in_planes = 2048

    def __init__(self, num_classes=None):
        super(ResNetBuilder, self).__init__()
        self.base = ResNetV1()
        self.avgpool = nn.GlobalAvgPool2D()
        self.flatten = nn.Flatten()

        self.num_classes = num_classes
        self.bottleneck = nn.HybridSequential()
        self.bottleneck.add(nn.Dense(512, in_units=2048, weight_initializer=mx.initializer.MSRAPrelu('out', 0)))
        self.bottleneck.add(nn.BatchNorm(in_channels=512))
        self.bottleneck.add(nn.LeakyReLU(0.1))
        self.bottleneck.add(nn.Dropout(0.5))

        self.classifier = nn.Dense(self.num_classes, in_units=512,
                                   weight_initializer=mx.initializer.Normal(0.001))

    def hybrid_forward(self, F, x):
        global_feat = self.base(x)
        global_feat = self.avgpool(global_feat)
        global_feat = self.flatten(global_feat)
        feat = self.bottleneck(global_feat)
        cls_score = self.classifier(feat)
        return cls_score, global_feat


if __name__ == '__main__':
    net = ResNetBuilder(233)
    net.initialize()
    from mxnet import nd

    x = nd.ones((2, 3, 256, 128))
    y = net(x)
    from IPython import embed

    embed()
