# encoding: utf-8
"""
@author:  liaoxingyu
@contact: xyliao1993@qq.com 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
from os import path as osp
from pprint import pprint

import mxnet as mx
import numpy as np
from mxboard import SummaryWriter
from mxnet import gluon
from mxnet.gluon.data import DataLoader

from config import opt
from datasets import data_manager
from datasets.data_provider import ImageData
from datasets.samplers import RandomIdentitySampler
from models import get_baseline_model
from trainers import reidEvaluator, reidTrainer
from utils.loss import TripletLoss
from utils.serialization import Logger
from utils.serialization import save_checkpoint
from utils.transforms import TrainTransform, TestTransform


def train(**kwargs):
    opt._parse(kwargs)

    # set random seed and cudnn benchmark

    sys.stdout = Logger(osp.join(opt.save_dir, 'log_train.txt'))

    print('=========user config==========')
    pprint(opt._state_dict())
    print('============end===============')

    print('initializing dataset {}'.format(opt.dataset))
    dataset = data_manager.init_dataset(name=opt.dataset)

    summary_writer = SummaryWriter(osp.join(opt.save_dir, 'tensorboard_log'))

    if 'triplet' in opt.model_name:
        trainloader = DataLoader(
            ImageData(dataset.train, TrainTransform(opt.height, opt.width)),
            sampler=RandomIdentitySampler(dataset.train, opt.num_instances),
            batch_size=opt.train_batch, num_workers=opt.workers, last_batch='discard'
        )
    else:
        trainloader = DataLoader(
            ImageData(dataset.train, TrainTransform(opt.height, opt.width)),
            batch_size=opt.train_batch, shuffle=True, num_workers=opt.workers,
        )

    queryloader = DataLoader(
        ImageData(dataset.query, TestTransform(opt.height, opt.width)),
        batch_size=opt.test_batch, num_workers=opt.workers,
    )

    galleryloader = DataLoader(
        ImageData(dataset.gallery, TestTransform(opt.height, opt.width)),
        batch_size=opt.test_batch, num_workers=opt.workers,
    )

    print('initializing model ...')
    model = get_baseline_model(dataset.num_train_pids, mx.gpu(0))

    print('model size: {:.5f}M'.format(sum(p.data().size
                                           for p in model.collect_params().values()) / 1e6))

    xent_criterion = gluon.loss.SoftmaxCrossEntropyLoss()
    tri_criterion = TripletLoss(opt.margin)

    def cls_criterion(cls_scores, feat, targets):
        cls_loss = xent_criterion(cls_scores, targets)
        return cls_loss

    def triplet_criterion(cls_scores, feat, targets):
        triplet_loss, dist_ap, dist_an = tri_criterion(feat, targets)
        return triplet_loss

    def cls_tri_criterion(cls_scores, feat, targets):
        cls_loss = xent_criterion(cls_scores, targets)
        triplet_loss, dist_ap, dist_an = tri_criterion(feat, targets)
        loss = cls_loss + triplet_loss
        return loss

    # get optimizer
    optimizer = gluon.Trainer(model.collect_params(), opt.optim, {'learning_rate': opt.lr, 'wd': opt.weight_decay})

    def adjust_lr(optimizer, ep):
        if ep < 20:
            lr = 1e-4 * (ep + 1) / 2
        elif ep < 80:
            lr = 1e-3 * opt.num_gpu
        elif ep < 180:
            lr = 1e-4 * opt.num_gpu
        elif ep < 300:
            lr = 1e-5 * opt.num_gpu
        elif ep < 320:
            lr = 1e-5 * 0.1 ** ((ep - 320) / 80) * opt.num_gpu
        elif ep < 400:
            lr = 1e-6
        elif ep < 480:
            lr = 1e-4 * opt.num_gpu
        else:
            lr = 1e-5 * opt.num_gpu

        optimizer.set_learning_rate(lr)

    start_epoch = opt.start_epoch

    # get trainer and evaluator
    use_criterion = None
    if opt.model_name == 'softmax':
        use_criterion = cls_criterion
    elif opt.model_name == 'softmax_triplet':
        use_criterion = cls_tri_criterion
    elif opt.model_name == 'triplet':
        use_criterion = triplet_criterion

    reid_trainer = reidTrainer(opt, model, optimizer, use_criterion, summary_writer, mx.gpu(0))
    reid_evaluator = reidEvaluator(model, mx.gpu(0))

    # start training
    best_rank1 = -np.inf
    best_epoch = 0

    for epoch in range(start_epoch, opt.max_epoch):
        if opt.step_size > 0:
            adjust_lr(optimizer, epoch + 1)
        reid_trainer.train(epoch, trainloader)

        # skip if not save model
        if opt.eval_step > 0 and (epoch + 1) % opt.eval_step == 0 or (epoch + 1) == opt.max_epoch:
            rank1 = reid_evaluator.evaluate(queryloader, galleryloader)
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

            state_dict = {'model': model,
                          'epoch': epoch}
            save_checkpoint(state_dict, is_best=is_best, save_dir=opt.save_dir,
                            filename='checkpoint_ep' + str(epoch + 1) + '.params')

    print(
        'Best rank-1 {:.1%}, achived at epoch {}'.format(best_rank1, best_epoch))


def test(**kwargs):
    opt._parse(kwargs)

    sys.stdout = Logger(osp.join(opt.save_dir, 'log_test.txt'))
    ctx = mx.gpu(0)
    print('initializing dataset {}'.format(opt.dataset))
    dataset = data_manager.init_dataset(name=opt.dataset)

    queryloader = DataLoader(
        ImageData(dataset.query, TestTransform(opt.height, opt.width)),
        batch_size=opt.test_batch, num_workers=opt.workers,
    )

    galleryloader = DataLoader(
        ImageData(dataset.gallery, TestTransform(opt.height, opt.width)),
        batch_size=opt.test_batch, num_workers=opt.workers,
    )

    print('loading model ...')
    model = get_baseline_model(dataset.num_train_pids, ctx)
    model.load_parameters('/home/test2/mxnet-ckpt/market1501_softmax_triplet_gluon/model_best.params', ctx)
    print('model size: {:.5f}M'.format(sum(p.data().size
                                           for p in model.collect_params().values()) / 1e6))

    reid_evaluator = reidEvaluator(model, ctx)
    reid_evaluator.evaluate(queryloader, galleryloader)


if __name__ == '__main__':
    import fire

    fire.Fire()
