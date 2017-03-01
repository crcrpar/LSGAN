#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
import argparse
import os
from six.moves import range

import numpy as np

import chainer
from chainer import training
from chainer.traininig import extensions

from models import images
from models import chars
from updater import LSGANUpdater


def main():
    parser = argparse.ArgumentParser(descriotion='Chainer LSGAN')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='number of data in each mini batch')
    parser.add_argument('--epoch', '-e', type=int, default=1000,
                        help='number of sweeps over the dataset')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dataset', '-d', type=int, default=0,
                        help='datasets: [scene(bedroom),\
                                         scene(church),\
                                         scene(dining room),\
                                         scene(kitchen),\
                                         scene(conference room),\
                                         Chinese characters]'
                        )
    parser.add_argument('--out', '-o', type=str, default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--seed', default=None,
                        help='Random seed of z at visualization stage')
    parser.add_argument('--snapshot_interval', type=int, default=1000,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')
    args = parser.parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# n_hidden: {}'.format(args.n_hidden))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    if args.dataset == -1 or args.dataset == 5:
        gen = models.chars.Generator()
        dis = models.chars.Discriminator()
        alpha = 0.0002
    else:
        gen = models.images.Generator()
        dis = models.images.Discriminator()
        alpha = 0.0001

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        gen.to_gpu()  # Copy the model to the GPU
        dis.to_gpu()

    # Setup an optimizer
    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001), 'hook_dec')
        return optimizer
    opt_gen = make_optimizer(gen, alpha)
    opt_dis = make_optimizer(dis, alpha)

    # Set up a trainer
    updater = DCGANUpdater(
        models=(gen, dis),
        params=(-1., 1., 0.),
        iterator=train_iter,
        optimizer={
            'gen': opt_gen, 'dis': opt_dis},
        device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    snapshot_interval = (args.snapshot_interval, 'iteration')
    display_interval = (args.display_interval, 'iteration')
    trainer.extend(
        extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        gen, 'gen_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dis, 'dis_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'gen/loss', 'dis/loss',
    ]), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(
        out_generated_image(
            gen, dis,
            10, 10, args.seed, args.out),
        trigger=snapshot_interval)

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

if __name__ == '__main__':
    main()
