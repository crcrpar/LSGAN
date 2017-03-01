#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function

import chainer
from chainer import functions as F


class LSGANUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        self._a, self._b, sefl._c = kwargs.pop('params')
        super(LSGANUpdater, self).__init__(*args, **kwargs)

    def loss_dis(self, dis, y_real, y_fake):
        batchsize = y_real.data.shape[0]
        loss = (y_real - self._b) ** 2 + (y_fake - self._a) ** 2
        loss /= float(2 * batchsize)
        chaienr.report({'loss': loss}, dis)
        return loss

    def loss_gen(self, gen, y_real, y_fake):
        batchsize = y_real.data.shape[0]
        loss = (y_real - self._c) ** 2 + (y_fake - self._c) ** 2
        loss /= float(2 * batchsize)
        chainer.report({'loss': loss}, gen)
        return loss

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimiazer('dis')

        batch = self.get_iterator('main').next()
        batch, y = self.converter(batch, self.device)
        x_real = chainer.Variable(self.converter(batch, self.device)) / 255.
        xp = chainer.cuda.get_array_module(x_real.data)

        batchsize = len(batch)

        y_real = self.dis(x_real)
        x_fake = self.gen(y)
        y_fake = self.dis(y_fake)

        gen_optimizer.update(self.loss_gen, self.gen, y_real, y_fake)
        dis_optimizer.update(self.loss_dis, self.dis, y_fake)
