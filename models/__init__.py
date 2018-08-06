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


def get_baseline_model(num_classes, ctx, model_path=None):
    model = ResNetBuilder(num_classes)
    model.initialize(ctx=ctx)
    if model_path is not None:
        model.base.load_parameters(model_path, ctx, ignore_extra=True)
    # state_dict = nd.load(model_path)
    # params = model.base._collect_params_with_prefix()
    # for i in params:
    #     params[i]._load_init(state_dict[i], ctx)
    model.hybridize(static_alloc=True, static_shape=True)
    return model
