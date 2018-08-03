# encoding: utf-8
"""
@author:  liaoxingyu
@contact: xyliao1993@qq.com 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from mxnet import nd


class BaseEvaluator(object):
    def __init__(self, model, ctx):
        self.model = model
        self.ctx = ctx

    def evaluate(self, queryloader, galleryloader, ranks=[1, 5, 10, 20]):
        qf, q_pids, q_camids = [], [], []
        for batch_idx, inputs in enumerate(queryloader):
            feature, pids, camids = self._eval_step(inputs)
            qf.append(feature)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = nd.concat(*qf, dim=0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.shape[0], qf.shape[1]))

        gf, g_pids, g_camids = [], [], []
        for batch_idx, inputs in enumerate(galleryloader):
            feature, pids, camids = self._eval_step(inputs)
            gf.append(feature)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = nd.concat(*gf, dim=0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.shape[0], gf.shape[1]))

        print("Computing distance matrix")

        m, n = qf.shape[0], gf.shape[0]
        distmat = nd.power(qf, 2).sum(axis=1, keepdims=True).broadcast_to((m, n)) + \
                  nd.power(gf, 2).sum(axis=1, keepdims=True).broadcast_to((n, m)).T
        distmat = distmat - 2 * nd.dot(qf, gf.T)
        distmat = distmat.asnumpy()

        print("Computing CMC and mAP")
        cmc, mAP = self.eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        print("Results ----------")
        print("mAP: {:.1%}".format(mAP))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
        print("------------------")

        return cmc[0]

    def eval_func(self, distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
        """Evaluation with market1501 metric
            Key: for each query identity, its gallery images from the same camera view are discarded.
            """
        num_q, num_g = distmat.shape
        if num_g < max_rank:
            max_rank = num_g
            print("Note: number of gallery samples is quite small, got {}".format(num_g))
        indices = np.argsort(distmat, axis=1)
        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

        # compute cmc curve for each query
        all_cmc = []
        all_AP = []
        num_valid_q = 0.  # number of valid query
        for q_idx in range(num_q):
            # get query pid and camid
            q_pid = q_pids[q_idx]
            q_camid = q_camids[q_idx]

            # remove gallery samples that have the same pid and camid with query
            order = indices[q_idx]
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            keep = np.invert(remove)

            # compute cmc curve
            # binary vector, positions with value 1 are correct matches
            orig_cmc = matches[q_idx][keep]
            if not np.any(orig_cmc):
                # this condition is true when query identity does not appear in gallery
                continue

            cmc = orig_cmc.cumsum()
            cmc[cmc > 1] = 1

            all_cmc.append(cmc[:max_rank])
            num_valid_q += 1.

            # compute average precision
            # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
            num_rel = orig_cmc.sum()
            tmp_cmc = orig_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)

        assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)

        return all_cmc, mAP

    def _eval_step(self, inputs):
        raise NotImplementedError
