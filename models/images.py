#!/usr/bin/env python
# coding: utf-8
import numpy

import chainer
from chainer import functions as F
from chainer import links as L


class Generator(chainer.Chain):
    """Discriminator for scene generation"""

    def __init__(self, name=None):
        self._layers = {}
        self._layers['fc_1'] = L.Linear(in_size=1024, out_size=7 * 7 * 256)
        self._layers['deconv_1'] = L.Deconvolution2D(
            in_channels=256, out_channels=256, ksize=(3, 3), stride=2)
        self._layers['bn_1'] = L.BatchNormalization(size=256)
        self._layers['deconv_2'] = L.Deconvolution2D(
            in_channels=256, out_channels=256, ksize=(3, 3), stride=1)
        self._layers['bn_2'] = L.BatchNormalization(size=256)
        self._layers['deconv_3'] = L.Deconvolution(
            in_channels=256, out_channels=256, ksize=(3, 3), stride=2)
        self._layers['bn_3'] = L.BatchNormalization(size=256)
        self._layers['deconv_4'] = L.Deconvolution(
            in_channels=256, out_channels=256, ksize=(3, 3), stride=1)
        self._layers['bn_4'] = L.BatchNormalization(size=256)
        self._layers['deconv_5'] = L.Deconvolution(
            in_channels=256, out_channels=128, ksize=(3, 3), stride=2)
        self._layers['bn_5'] = L.BatchNormalization(size=64)
        self._layers['deconv_6'] = L.Deconvolution(
            in_channels=128, out_channels=64, kszie=(3, 3), stride=2)
        self._layers['bn_6'] = L.BatchNormalization(size=64)
        self._layers['deconv_7'] = L.Deconvolution(
            in_channels=64, out_channels=3, ksize=(3, 3), stride=2)
        if name is None:
            name = "Generator"
        self.name = name
        self.trian = True
        super(Generator, self).__init__(**self._layers)

    def __call__(self, z=None):
        if z is None:
            z = numpy.random.uniform(low=-1.0, high=1.0, size=1024)
        h1 = F.relu(self.fc_1(z))
        h2 = F.relu(self.bn_1(self.deconv_1(h1), test=not self.train))
        h3 = F.relu(self.bn_2(self.deconv_2(h2), test=not self.train))
        h4 = F.relu(self.bn_3(self.deconv_3(h3), test=not self.train))
        h5 = F.relu(self.bn_4(self.deconv_4(h4), test=not self.train))
        h6 = F.relu(self.bn_5(self.deconv_5(h5), test=not self.train))
        h7 = F.relu(self.bn_6(self.deconv_6(h6), test=not self.train))
        _out = F.relu(self.deconv_7(h7))
        return _out

    def train(self, _train):
        self.train = _train


class Discriminator(chainer.Chain):
    """Discriminator for scene generation"""

    def __init__(self, name=None):
        self._layers = {}
        self._layers['conv_1'] = L.Convolution2D(
            in_channels=3, out_channels=64, ksize=(5, 5), stride=2)
        self._layers['bn_1'] = L.BatchNormalization(size=64)
        self._layers['conv_2'] = L.Convolution2D(
            in_channels=64, out_channels=128, ksize=(5, 5), stride=2)
        self._layers['bn_2'] = L.BatchNormalization(size=128)
        self._layers['conv_3'] = L.Convolution2D(
            in_channels=128, out_channels=256, ksize=(5, 5), stride=2)
        self._layers['bn_3'] = L.BatchNormalization(size=256)
        self._layers['conv_4'] = L.Convolution2D(
            in_channels=256, out_chaneels=512, ksize=(5, 5), stride=2)
        self._layers['bn_4'] = L.BatchNormalization(size=512)
        self._layers['fc_1'] = L.Linear(in_size=None, out_size=1)
        if name is None:
            name = "Discriminator"
        self.name = name
        self.train = True
        super(Discriminator, self).__init__(**self._layers)

    def __call__(self, x):
        h1 = F.leaky_relu(self.bn_1(self.conv_1(x), test=not self.train))
        h2 = F.leaky_relu(self.bn_2(self.conv_2(h1), test=not self.train))
        h3 = F.leaky_relu(self.bn_3(self.conv_3(h2), test=not self.train))
        h4 = F.leaky_relu(self.bn_4(self.conv_4(h3), test=not self.train))
        _out = F.leaky_relu(self.fc_1(h4))
        return _out

    def set_mode(self, _train):
        self.train = _train
