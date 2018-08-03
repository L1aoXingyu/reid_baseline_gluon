# encoding: utf-8
"""
@author:  liaoxingyu
@contact: xyliao1993@qq.com 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import time

from utils.meters import AverageMeter


class BaseTrainer(object):
    def __init__(self, opt, model, optimzier, criterion, summary_writer, ctx):
        self.opt = opt
        self.model = model
        self.optimizer = optimzier
        self.criterion = criterion
        self.summary_writer = summary_writer
        self.ctx = ctx

        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.losses = AverageMeter()

    def train(self, epoch, data_loader):

        self.batch_time.reset()
        self.data_time.reset()
        self.losses.reset()

        start = time.time()
        for i, inputs in enumerate(data_loader):
            self.data_time.update(time.time() - start)

            # model optimizer
            loss = self._train_step(inputs)

            self.losses.update(loss)
            self.batch_time.update(time.time() - start)

            # tensorboard
            global_step = epoch * len(data_loader) + i

            self.summary_writer.add_scalar('loss', loss, global_step)
            self.summary_writer.add_scalar('lr', self.optimizer.learning_rate, global_step)

            start = time.time()

            if (i + 1) % self.opt.print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Batch Time {:.3f} ({:.3f})\t'
                      'Data Time {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, len(data_loader),
                              self.batch_time.val, self.batch_time.mean,
                              self.data_time.val, self.data_time.mean,
                              self.losses.val, self.losses.mean))

        print('Epoch: [{}]\tEpoch Time {:.3f} s\tLoss {:.3f}\t'
              'Lr {:.2e}'
              .format(epoch, self.batch_time.sum, self.losses.mean, self.optimizer.learning_rate))
        print()

    def _train_step(self, inputs):
        raise NotImplementedError
