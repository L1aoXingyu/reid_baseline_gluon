# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random

import mxnet as mx
from mxnet.gluon.data.vision import transforms as T


class Random2DTranslation(object):
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.

    Args:
        height (int): target height.
        width (int): target width.
        p (float): probability of performing this transformation. Default: 0.5.
    """

    def __init__(self, height, width, p=0.5, interpolation=1):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (mx.image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if random.random() < self.p:
            return T.Resize((self.width, self.height), interpolation=self.interpolation)(img)
        new_width, new_height = int(
            round(self.width * 1.125)), int(round(self.height * 1.125))
        resized_img = T.Resize((new_width, new_height), interpolation=self.interpolation)(img)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = mx.image.fixed_crop(resized_img, x1, y1, self.width, self.height)
        return croped_img


class TrainTransform(object):
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, x):
        x = Random2DTranslation(self.h, self.w)(x)
        x = T.RandomFlipLeftRight()(x)
        x = T.ToTensor()(x)
        x = T.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225))(x)
        return x


class TestTransform(object):
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, x=None):
        x = T.Resize((self.w, self.h))(x)
        x = T.ToTensor()(x)
        x = T.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225))(x)
        return x
