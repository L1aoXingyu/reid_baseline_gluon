# encoding: utf-8
"""
@author:  liaoxingyu
@contact: xyliao1993@qq.com 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from bases.base_evaluator import BaseEvaluator


class reidEvaluator(BaseEvaluator):
    def __init__(self, model, ctx):
        super().__init__(model, ctx)

    def _eval_step(self, inputs):
        imgs, pids, camids = inputs
        imgs = imgs.as_in_context(self.ctx)
        pids = pids.asnumpy()
        camids = camids.asnumpy()

        _, feature = self.model(imgs)
        return feature, pids, camids
