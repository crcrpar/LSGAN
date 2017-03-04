import os

import numpy
from PIL import Image

import chainer


def out_generated_image(gen, dis, rows, cols, seed, dst):
    @chainer.training.make_extension()
    def make_image(trainer):
        numpy.random.seed(seed)
        n_images = rows * cols
        xp = gen.xp
        x = gen(test=True, bs=n_images)
        if xp == chainer.cuda.cupy:
            x = chainer.cuda.to_cpu(x)
        numpy.random.seed()
        x = numpy.asarray(numpy.clip(x * 255, 0.0, 255.0), dtype=numpy.uint8)
        _, _, H, W = x.shape
        x = x.reshape((rows, cols, 3, H, W))
        x = x.transpose(0, 3, 1, 4, 2)
        x = x.reshape((rows * H, cols * W, 3))
        preview_dir = '{}/preview'.format(dst)
        preview_path = os.path.join(
            preview_dir, 'image{:0>8}.png'.format(trainer.updater.iteration))
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        Image.fromarray(x).save(preview_path)
    return make_image
