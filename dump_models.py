import os

import chainer


def dump_mosels(gen, dis, dst):
    @chainer.training.make_extension()
    def save_models(traier):
        xp = gen.xp
        if xp == chainer.cuda.cupy:
            gen.to_cpu()
            dis.to_cpu()
        save_dir = '{}/cpu_models'.format(dst)
        gen_save_path = os.path.join(save_dir, 'gen_{:0>8}.npz'.format(trainer.updater.epoch))
        dis_save_path = os.path.join(save_dir, 'dis_{:0>8}.npz'.format(trainer.updater.epoch))
        chainer.serializers.save_npz(gen_save_path, gen)
        chainer.serializers.save_npz(dis_save_path, dis)
    return save_models
