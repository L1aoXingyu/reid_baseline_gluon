# encoding: utf-8
"""
@author:  liaoxingyu
@contact: xyliao1993@qq.com 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from mxnet import autograd

from bases.base_trainer import BaseTrainer


class reidTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer, criterion, summary_writer, ctx):
        super().__init__(opt, model, optimizer, criterion, summary_writer, ctx)

    def _train_step(self, inputs):
        imgs, pids, _ = inputs
        bs = imgs.shape[0]
        data = imgs.as_in_context(self.ctx)
        target = pids.as_in_context(self.ctx)

        with autograd.record():
            score, feat = self.model(data)
            loss = self.criterion(score, feat, target)

        loss.backward()
        self.optimizer.step(bs)
        return loss.mean().asscalar()
