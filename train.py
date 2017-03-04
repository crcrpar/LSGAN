#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
import os
import datetime
import json

import yaml
import numpy as np
import chainer
from chainer import training
from chainer.traininig import extensions

import models
from updater import LSGANUpdater
from visualizer import out_generated_image
from dump_mosels import dump_mosels


def setup_models(dataset, batch_size):
    if dataset == 'chinese':
        gen = models.chars.Generator()
        dis = models.chars.Discriminator()
        alpha = 0.0002
    else:
        gen = models.images.Generator()
        dis = models.images.Discriminator()
        alpha = 0.0001
    return gen, dis, alpha


def setup_Adam(model, alpha=0.0002, beta1=0.5, weight_decay=True):
    optimizer = chainer.optimizers.Adam(alpha, beta1=beta1)
    optimizer.setup(model)
    if weight_decay:
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001), 'hook_decay')
    return optimizer


def main():
    with open('settings.yml', 'r') as f:
        conf = yaml.load(f)
    print('### settings')
    print(json.dumps(conf, indent=2))

    # Set up neural networks to train
    gen, dis, alpha = setup_models(conf['dataset'], conf['batch_size'])
    opt_gen, opt_dis = setup_Adam(gen, alpha), setup_Adam(dis, alpha)
    if conf['gpu'] >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device(conf['gpu']).use()
        gen.to_gpu()
        dis.to_gpu()

    # Set up iterators
    if not os.isdir(conf['dataset_dir'], conf['dataset']):
        msg = 'No dataset found'
        raise Exception(msg)
    dataset_dir = os.path.join(conf['dataset_dif'], conf['dataset'])
    file_list = os.listdir(dataset_dir)
    train_iter = chainer.datasets.ImageDataset(
        paths=file_list, root=dataset_dir)

    # Set up a trainer
    updater = LSGANUpdater(
        models=(gen, dis),
        params=(-1., 1., 0.),
        iterator=train_iter,
        optimizer={'gen': opt_gen, 'dis': opt_dis},
        device=conf['gpu'])
    timestamp = datetime.datetime.now().strftime('%Y%m%d')
    out_dir = os.path.join(conf['out'], conf['dataset'], timestamp)
    trainer = training.Trainer(updater, (conf['epoch'], 'epoch'), out=out_dir)
    snapshot_interval = (conf['snapshot_interval'], 'iteration')
    display_interval = (conf['display_interval'], 'iteration')
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
    trainer.extend(out_generated_image(gen, dis, 10, 10, conf[
                   'seed'], conf['out']), trigger=snapshot_interval)
    trainer.extend(dump_mosels(gen, dis, out_dir), trigger=snapshot_interval)
    if conf['resume']:
        # Resume from a snapshot
        chainer.serializers.load_npz(conf['resume'], trainer)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
