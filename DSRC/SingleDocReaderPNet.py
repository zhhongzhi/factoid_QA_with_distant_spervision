# -*- encoding: utf-8 -*-
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import logging
import math

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

from torch.autograd import Variable
from utils import AverageMeter
from DSRC.rnn_readerPNet import RnnDocReader

# Modification:
#   - change the logger name
#   - save & load optimizer state dict
#   - change the dimension of inputs (for POS and NER features)
# Origin: https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/drqa

logger = logging.getLogger(__name__)


class DocReaderModel(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    def __init__(self, opt, char_embedding, state_dict=None):
        # Book-keeping.
        self.opt = opt
        self.updates = state_dict['updates'] if state_dict else 0
        self.train_loss = AverageMeter()

        # Building network.
        self.network = RnnDocReader(opt, char_embedding=char_embedding)
        if state_dict:
            new_state = set(self.network.state_dict().keys())
            for k in list(state_dict['network'].keys()):
                if k not in new_state:
                    del state_dict['network'][k]
            self.network.load_state_dict(state_dict['network'])

        # Building optimizer.
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if opt.optimizer == 'sgd':
            self.optimizer = optim.SGD(parameters, opt.learning_rate,
                                       momentum=opt.momentum,
                                       weight_decay=opt.weight_decay)
        elif opt.optimizer == 'adamax':
            self.optimizer = optim.Adamax(parameters, opt.learning_rate,
                                          weight_decay=opt.weight_decay)
        else:
            raise RuntimeError('Unsupported optimizer: %s' % opt.optimizer)
        if state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        num_params = sum(p.data.numel()
                         for p in parameters)  #  if p.data.data_ptr() != self.network.embedding.weight.data.data_ptr()
        print(("{} parameters".format(num_params)))
        self.loss_fun = WNLL()

    def zero_loss(self):
        self.train_loss = AverageMeter()

    def update(self, ex):
        # Train mode
        self.network.train()  # 设置为训练模式

        inputs = [Variable(torch.from_numpy(e).long().cuda(async=True)) for e in ex[:5]]  # , device_id=self.opt.device_id
        x1_c, x1_f, x1_mask, x2_c, x2_mask = inputs
        x1_f = x1_f.float()
        x1_mask = x1_mask.byte()
        x2_mask = x2_mask.byte()
        inputs = x1_c, x1_f, x1_mask, x2_c, x2_mask

        # weights is always None in the experiment.
        weights = ex[-1]
        if weights is not None:
            weights = Variable(torch.from_numpy(ex[-1]).float().cuda(async=True)) # , device_id=self.opt.device_id

        # Run forward
        s, e = self.network(*inputs)
        g_s = ex[5]
        g_e = ex[6]
        g_s = Variable(torch.from_numpy(g_s).long().cuda(async=True)) # , device_id=self.opt.device_id
        g_e = Variable(torch.from_numpy(g_e).long().cuda(async=True)) # , device_id=self.opt.device_id

        bs, p_len, _ = s.size()
        s = s.view(bs, p_len)
        e = e.view(bs, p_len)

        # loss = F.cross_entropy(s, g_s, ignore_index=-1) + F.cross_entropy(e, g_e, ignore_index=-1)
        loss = self.loss_fun(s, g_s.data.long(), weights) + self.loss_fun(e, g_e.data.long(), weights)
        # Compute loss and accuracies
        self.train_loss.update(loss.data[0], len(ex[0]))

        # Clear gradients and run backward
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm(self.network.parameters(), self.opt.grad_clipping)

        # Update parameters
        self.optimizer.step()
        self.updates += 1

        # Reset any partially fixed parameters (e.g. rare words)
        self.reset_parameters()

    def predict_on_valid_or_test(self, dt, valid_or_test ='valid'):
        # do valid on the whole valid set
        # 把valid data 分batch，
        q_c_idxs, q_masks, p_c_idxs, p_masks, qe_comm_ft, b_pos, e_pos, p_tokens = \
                dt.valid_dat if valid_or_test == 'valid' else dt.test_dat
        all_s, all_e = self.predict_all_in_batch(p_c_idxs, qe_comm_ft, p_masks, q_c_idxs, q_masks)
        lbs = self.old_eval(all_s, all_e, p_tokens=p_tokens)
        return lbs

    def get_eval_acc(self, b_pos, e_pos, all_s, all_e):
        pos_num = 0
        for g_b, g_e, p_b, p_e in zip(b_pos, e_pos, all_s, all_e):
            if g_b == p_b and g_e == p_e:
                pos_num += 1
        return pos_num * 1.0 / len(b_pos)

    def old_eval(self, all_s, all_e, p_tokens):
        all_lbs = self.point_rse2lbs(all_e=all_e, all_s=all_s, p_tokens=p_tokens)
        return all_lbs

    def predict_on_valid_or_test_constrained(self, dt, valid_or_test ='valid', return_p_res=False):
        q_c_idxs, q_masks, p_c_idxs, p_masks, qe_comm_ft, b_pos, e_pos, p_tokens = \
                dt.valid_dat if valid_or_test == 'valid' else dt.test_dat
        all_s, all_e = self.predict_all_in_batch(p_c_idxs, qe_comm_ft,  p_masks, q_c_idxs, q_masks, need_prob_ditri=True)
        all_s, all_e = self.constrained_infer(all_s, all_e)
        lbs = self.old_eval(all_s, all_e, p_tokens=p_tokens)
        return lbs

    def constrained_infer(self, all_s, all_e, need_score=False):
        s_poses, e_poses = [], []

        all_s_pos_vs, all_s_pos = torch.max(all_s, dim=1)
        all_e_pos_vs, all_e_pos = torch.max(all_e, dim=1)
        all_s_pos, all_e_pos = all_s_pos.data.cpu().numpy(), all_e_pos.data.cpu().numpy()
        all_s_pos_vs, all_e_pos_vs = all_s_pos_vs.data.cpu().numpy(), all_e_pos_vs.data.cpu().numpy()
        infer_num = 0
        scores = []
        for idx, (p_s, p_e, glb_s, glb_e) in enumerate(zip(all_s, all_e, all_s_pos, all_e_pos)):
            # if glb_s <= glb_e and glb_s+10 >= glb_e:
            L = 10   # originally this value is set to 10, changed
            if glb_s <= glb_e and glb_s+ L>= glb_e:
                s_poses.append(glb_s)
                e_poses.append(glb_e)
                scores.append(float(all_s_pos_vs[idx] + all_e_pos_vs[idx]))
                continue
            infer_num += 1
            pair_score = {}
            s_vs, cdt_s_poses = torch.topk(p_s, k=5)
            s_vs = s_vs.data.cpu().numpy()
            cdt_s_poses = cdt_s_poses.data.cpu().numpy()
            p_e = p_e.data.cpu().numpy()
            for s_v, cdt_s in zip(s_vs, cdt_s_poses):
                part_p_e = list(p_e[cdt_s: cdt_s+L])
                e_v = max(part_p_e)
                e = cdt_s + part_p_e.index(e_v)
                pair_score[(cdt_s, e)] = float(s_v) + float(e_v)
            max_score = max(pair_score.values())
            s, e = [pair for pair, score in pair_score.items() if score == max_score][0]
            s_poses.append(s)
            e_poses.append(e)
            scores.append(max_score)
        if need_score:
            return s_poses, e_poses, scores
        return s_poses, e_poses

    def point_rse2lbs(self, all_s, all_e, p_tokens):
        """
        transfer the output result to sequence labeling result for evaluation
        :param all_s:
        :param all_e:
        :param p_tokens:
        :return:
        """
        if not isinstance(all_s, list):
            all_s = all_s.data.cpu().numpy()
            all_e = all_e.data.cpu().numpy()
        all_lbs = []
        for s_idx, e_idx, st_token in zip(all_s, all_e, p_tokens):
            for idx, token in enumerate(st_token):
                if idx == s_idx and e_idx >= s_idx:
                    all_lbs.append(0)
                elif s_idx < idx < e_idx:
                    all_lbs.append(1)
                else:
                    all_lbs.append(2)
        return all_lbs

    def predict_batch(self, ex, gpu=True, give_att=False):
        self.network.eval()
        inputs = [Variable(torch.from_numpy(e).long(), volatile=True) for e in ex[:5]]
        if gpu:
            inputs = [i.cuda(async=True) for i in inputs]
        x1_c, x1_f, x1_mask, x2_c, x2_mask = inputs
        x1_f = x1_f.float()
        x1_mask = x1_mask.byte()
        x2_mask = x2_mask.byte()
        inputs = x1_c, x1_f, x1_mask, x2_c, x2_mask, give_att
        self.network.eval()
        if give_att:
            s, e, att_emb, att_hign = self.network(*inputs)
        else:
            s, e = self.network(*inputs)
        s = F.softmax(s.view(x1_c.size(0), x1_c.size(1)))
        e = F.softmax(e.view(x1_c.size(0), x1_c.size(1)))
        if give_att:
            return s, e, att_emb, att_hign
        return s, e

    def predict_all_in_batch(self, p_c_idxs, qe_comm_ft, p_masks, q_c_idxs, q_masks, need_prob_ditri=False, gpu=True):
        batch_num = math.ceil(len(qe_comm_ft)/self.opt.valid_batch_size)
        all_s, all_e = [], []
        for batch_idx in range(int(batch_num)+1):
            batch_end = min((batch_idx + 1) * self.opt.valid_batch_size, len(qe_comm_ft))
            indexes = np.arange(batch_idx*self.opt.valid_batch_size, batch_end)
            if len(indexes) == 0:
                break
            inputs = [i[indexes] for i in (p_c_idxs, qe_comm_ft, p_masks, q_c_idxs, q_masks)]
            s, e = self.predict_batch(inputs, gpu=gpu)
            all_s.append(s)
            all_e.append(e)
        all_s = torch.cat(all_s, dim=0)
        all_e = torch.cat(all_e, dim=0)
        bs, p_len = all_s.size()
        all_s = all_s.view(bs, p_len)
        all_e = all_e.view(bs, p_len)
        if need_prob_ditri:
            return all_s, all_e
        _, all_s_pos = torch.max(all_s, dim=1)
        _, all_e_pos = torch.max(all_e, dim=1)
        return all_s_pos, all_e_pos

    def reset_parameters(self):
        # Reset fixed embeddings to original value
        if self.opt.tune_partial > 0:
            offset = self.opt.tune_partial + 2
            if offset < self.network.embedding.weight.data.size(0):
                self.network.embedding.weight.data[offset:] \
                    = self.network.fixed_embedding

    def save(self, filename, epoch):
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'updates': self.updates
            },
            'config': self.opt,
            'epoch': epoch
        }
        try:
            torch.save(params, filename)
            logger.info('model saved to {}'.format(filename))
        except BaseException:
            logger.warn('[ WARN: Saving failed... continuing anyway. ]')

    def cuda(self):
        self.network.cuda()


class WNLL(nn.Module):
    #def __init(self, p, y):
     #   super(WNLL, self).__init__()
        #self.p = p
        #self.y = y
    def forward(self, p, y, weights):
        tmp = F.softmax(p)[torch.arange(0, p.size(0)).long().cuda(), y]
        tmp = torch.log(tmp)
        if weights is not None:
            tmp = tmp * weights
        return -1 * torch.mean(tmp)
