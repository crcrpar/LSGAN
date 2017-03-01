#!/usr/bin/env python
# coding: utf-8
# TODO (crcrpar): make concat (x, y) method.
import numpy

import chainer
from chainer import functions as F
from chainer import links as L


class Generator(chainer.Chain):
    """Generator for Chinese characters generation"""

    def __init__(self, name=None):
        _layers = {
                'fc_y': L.Linear(in_size=3740, out_size=256),
                'fc_1': L.Linear(in_size=1024 + 256, out_size=7 * 7 * 128),
                'bn': L.BatchNormalization(size=7 * 7 * 128),
                'deconv_1': L.Deconvolution2D(in_channels=128, out_channels=128, ksize=(5, 5), stride=2),
                'bn_1': L.BatchNormalization(size=128),
                'deconv_2': L.Deconvolution(in_channels=128, out_channels=1, kszie=(5, 5), stride=2)
                }
        if name is None:
            name = "Generator"
        self.name = name
        self.train = True
        super(Generator, self).__init__(**_layers)

    def __call__(self, y, z=None):
        if y.dtype == numpy.int:
            y = y.astype(numpy.float32)
        _y = self.fc_y(y)
        if z is None:
            z = numpy.random.uniform(low=-1.0, high=1.0, size=1024)
        yz = F.concatenate((z, _y))
        h1 = F.relu(self.bn(self.fc_1(yz), test=not self.train))
        h2 = F.relu(self.bn_1(self.deconv_1(h1), test=not self.train))
        _out = F.relu(self.deconv_2(h2))
        return _out

    def set_mode(self, _train):
        self.train = _train


class Discriminator(chainer.Chain):
    """Discriminator for Chainese characters generation"""

    def __init__(self, name=None):
        _layers = {
                'fc_y': L.Linear(in_size=3740, out_size=256),
                'conv_1': L.Convolution2D(in_channels=1, out_channels=256, ksize=(5, 5), stride=2),
                'conv_2': L.Convolution2D(in_channels=256, out_channels=320, ksize=(5, 5), stride=2),
                'bn_1': L.BatchNormalization(size=320),
                'fc_1': L.Linear(in_size=None, out_size=1024),
                'bn_2': L.BatchNormalization(size=1024),
                'fc_2': L.Linear(in_size=1024, out_size=1)
                }
        if name is None:
            name = "Generator"
        self.name = name
        self.train = True
        super(Discriminator, self).__init__(**_layers)

    def __call__(self, x, y):
        if y.dtype == numpy.int:
            y = y.astype(numpy.float32)
        h1 = F.leaky_relu(self.conv_1(x))
        h2 = F.leaky_relu(self.bn_1(self.conv_2(h1), test=not self.train))
        h3 = F.leaky_relu(self.bn_2(self.fc_1(h2), test=not self.train))
        _out = F.leaky_relu(self.fc_2(h3))
        return _out
