# encoding: utf-8
"""
@author:  liaoxingyu
@contact: xyliao1993@qq.com 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# import torch
from mxnet import nd
import numpy as np

# from torch import nn


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (nd.norm(x, axis=axis, keepdim=True) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.shape[0], y.shape[0]
    xx = nd.power(x, 2).sum(axis=1, keepdims=True).broadcast_to((m, n))
    yy = nd.power(y, 2).sum(axis=1, keepdims=True).broadcast_to((n, m)).T
    dist = xx + yy
    dist = dist - 2 * nd.dot(x, y.T)
    dist = dist.clip(a_min=1e-12, a_max=1e12).sqrt()  # for numerical stability
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.shape) == 2
    assert dist_mat.shape[0] == dist_mat.shape[1]
    N = dist_mat.shape[0]

    # shape [N, N]
    is_pos = nd.equal(labels.broadcast_to((N, N)), labels.broadcast_to((N, N)).T).astype('float32')
    is_neg = nd.not_equal(labels.broadcast_to((N, N)), labels.broadcast_to((N, N)).T).astype('float32')
    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_pos = dist_mat * is_pos
    dist_ap = nd.max(dist_pos, axis=1)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_neg = dist_mat * is_neg + nd.max(dist_mat, axis=1, keepdims=True) * is_pos
    dist_an = nd.min(dist_neg, axis=1)
    # shape [N]

    # if return_inds:
    #     # shape [N, N]
    #     ind = (labels.new().resize_as_(labels)
    #            .copy_(torch.arange(0, N).long())
    #            .unsqueeze(0).expand(N, N))
    #     # shape [N, 1]
    #     p_inds = torch.gather(
    #         ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
    #     n_inds = torch.gather(
    #         ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
    #     # shape [N]
    #     p_inds = p_inds.squeeze(1)
    #     n_inds = n_inds.squeeze(1)
    #     return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None):
        self.margin = margin

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(
            dist_mat, labels)
        loss = nd.relu(dist_ap - dist_an + self.margin)
        return loss, dist_ap, dist_an

# class CrossEntropyLabelSmooth(nn.Module):
#     """Cross entropy loss with label smoothing regularizer.
#     Reference:
#     Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
#     Equation: y = (1 - epsilon) * y + epsilon / K.
#     Args:
#         num_classes (int): number of classes.
#         epsilon (float): weight.
#     """
#
#     def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
#         super(CrossEntropyLabelSmooth, self).__init__()
#         self.num_classes = num_classes
#         self.epsilon = epsilon
#         self.use_gpu = use_gpu
#         self.logsoftmax = nn.LogSoftmax(dim=1)
#
#     def forward(self, inputs, targets):
#         """
#         Args:
#             inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
#             targets: ground truth labels with shape (num_classes)
#         """
#         log_probs = self.logsoftmax(inputs)
#         targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
#         if self.use_gpu: targets = targets.cuda()
#         targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
#         loss = (- targets * log_probs).mean(0).sum()
#         return loss
