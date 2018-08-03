# encoding: utf-8
"""
@author:    liaoxingyu
@contact:   xyliao1993@qq.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .baseline_model import ResNetBuilder


def get_baseline_model(num_classes, ctx, model_path='/home/test2/.mxnet/models/resnet50_v1-c940b1a0.params'):
    model = ResNetBuilder(num_classes)
    model.initialize(ctx=ctx)
    model.base.load_parameters(model_path, ctx, ignore_extra=True)
    # state_dict = nd.load(model_path)
    # params = model.base._collect_params_with_prefix()
    # for i in params:
    #     params[i]._load_init(state_dict[i], ctx)
    model.hybridize()
    return model
