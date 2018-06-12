# -*- encoding: utf-8 -*-
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import torch
import torch.nn as nn
import layers
import torch.nn.functional as F

# Modification: add 'pos' and 'ner' features.
# Origin: https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/drqa


def normalize_emb_(data):
    print(data.size(), data[:10].norm(2, 1))
    norms = data.norm(2, 1) + 1e-8
    if norms.dim() == 1:
        norms = norms.unsqueeze(1)
    data.div_(norms.expand_as(data))
    print(data.size(), data[:10].norm(2, 1))


class RnnDocReader(nn.Module):
    """Network for the Document Reader module of DrQA."""

    def __init__(self, opt, char_embedding, padding_idx=0, normalize_emb=False):
        super(RnnDocReader, self).__init__()
        # Store config
        self.opt = opt

        char_embedding = torch.FloatTensor(char_embedding)
        self.char_embedding = nn.Embedding(char_embedding.size(0), char_embedding.size(1), padding_idx=padding_idx)
        self.char_embedding.weight.data = char_embedding

        # Projection for attention weighted question
        if opt.use_qemb:
            self.qemb_match = layers.SeqAttnMatch(opt.char_emb_dim * 2)
            self.qemb_match_ds = layers.SeqAttnMatch(opt.char_emb_dim * 2)

        # Input size to RNN: word emb + question emb + manual features
        doc_input_size = opt.char_emb_dim * 2 + opt.num_features
        if opt.use_qemb:
            doc_input_size += opt.char_emb_dim * 2
        if opt.use_interaction:
            self.qhiden_match = layers.SeqAttnMatch(opt.hidden_size*2)
            self.qhiden_match_ds = layers.SeqAttnMatch(opt.hidden_size*2)

        self.char_rnn = layers.StackedBRNN(
            input_size=opt.char_emb_dim,
            hidden_size=opt.char_emb_dim,
            num_layers=1,
            dropout_rate=opt.dropout_rnn,
            dropout_output=opt.dropout_rnn_output,
            concat_layers=opt.concat_rnn_layers,
            use_tanh=True, #
            bidirectional=True
        )
        # RNN document encoder
        self.doc_rnn = layers.StackedBRNN(
            input_size=doc_input_size,
            hidden_size=opt.hidden_size,
            num_layers=opt.doc_layers,
            dropout_rate=opt.dropout_rnn,
            dropout_output=opt.dropout_rnn_output,
            concat_layers=opt.concat_rnn_layers,
            res_net=opt.res_net
        )
        question_input_size = opt.char_emb_dim * 2
        # RNN question encoder
        self.question_rnn = layers.StackedBRNN(
            input_size=question_input_size,
            hidden_size=opt.hidden_size,
            num_layers=opt.question_layers,
            dropout_rate=opt.dropout_rnn,
            dropout_output=opt.dropout_rnn_output,
            concat_layers=opt.concat_rnn_layers,
            res_net=opt.res_net
        )

        # Output sizes of rnn encoders
        doc_hidden_size = 2 * opt.hidden_size
        question_hidden_size = 2 * opt.hidden_size
        if opt.concat_rnn_layers:
            doc_hidden_size *= opt.doc_layers
            question_hidden_size *= opt.question_layers

        match_in_dim = opt.hidden_size * 2 * 2
        # Bilinear attention for span start/end
        self.s_linear = nn.Linear(match_in_dim, 1)
        self.e_linear = nn.Linear(match_in_dim + 1, 1)

    def forward(self, x1_c, x1_f, x1_mask, x2_c, x2_mask, give_att=False):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_pos = document POS tags             [batch * len_d]
        x1_ner = document entity tags          [batch * len_d]
        x1_mask = document padding mask        [batch * len_d]   # 加一个吧
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q]
        """
        # Embed both document and question

        question_rnn = self.question_rnn
        qemb_match = self.qemb_match
        qhiden_match = self.qhiden_match

        batch_size = x2_c.size(0)
        char_num = x1_c.size(2)
        x1_seq_len = x1_c.size(1)
        x2_seq_len = x2_c.size(1)

        x1_c = x1_c.view(batch_size*x1_seq_len, char_num)
        x2_c = x2_c.view(batch_size*x2_seq_len, char_num)   # bs * seq_len, char_num

        x1_c_emb = self.char_embedding(x1_c)
        x2_c_emb = self.char_embedding(x2_c)

        x1_c_emb_processed = self.char_rnn(x1_c_emb)  # bs * seq_len, char_num, emb_dim
        x2_c_emb_processed = self.char_rnn(x2_c_emb)  # bs * seq_len, char_num, emb_dim
        c_emb_dim_processed = x1_c_emb_processed.size(2)
        x1_emb = F.max_pool1d(x1_c_emb_processed.transpose(1, 2), kernel_size=char_num).contiguous().view(batch_size, x1_seq_len, c_emb_dim_processed)
        x2_emb = F.max_pool1d(x2_c_emb_processed.transpose(1, 2), kernel_size=char_num).contiguous().view(batch_size, x2_seq_len, c_emb_dim_processed)

        x1_f = x1_f.unsqueeze(2)
        if self.opt.dropout_emb > 0:
            x1_emb = nn.functional.dropout(x1_emb, p=self.opt.dropout_emb, training=self.training)
            x2_emb = nn.functional.dropout(x2_emb, p=self.opt.dropout_emb, training=self.training)

        drnn_input_list = [x1_emb, x1_f]
        # Add attention-weighted question representation
        if self.opt.use_qemb:
            if give_att:
                x2_weighted_emb, att_emb = qemb_match(x1_emb, x2_emb, x2_mask, need_attention=give_att)
            else:
                x2_weighted_emb = qemb_match(x1_emb, x2_emb, x2_mask)
            drnn_input_list.append(x2_weighted_emb)
        drnn_input = torch.cat(drnn_input_list, 2)

        # Encode document with RNN
        doc_hiddens = self.doc_rnn(drnn_input)

        # Encode question with RNN + merge hiddens
        question_hiddens = question_rnn(x2_emb)

        if give_att:
            question_hidden_expanded, att_high = qhiden_match(doc_hiddens, question_hiddens, x2_mask, need_attention=give_att)
        else:
            question_hidden_expanded = qhiden_match(doc_hiddens, question_hiddens, x2_mask)


        match_in = torch.cat([doc_hiddens, question_hidden_expanded], 2)

        # fts = self.match_rnn(match_in)
        s = self.s_linear(match_in)
        match_in_e = torch.cat([match_in, s], dim=2)
        e = self.e_linear(match_in_e)
        if give_att:
            return s, e, att_emb, att_high

        return s, e

        # crf_out = self.crf(fts)
        # crf_out = crf_out.view(self.opt.batch_size, psg_len, self.target_size, self.target_size)
        # return crf_out, fts
        # loss = crit.forward(scores, tg_v, mask_v)

    def get_word_emb(self, x1_c):
        """

        :param xs: bs * len * char_idxs
        :return:
        """
        self.eval()
        batch_size = x1_c.size(0)
        char_num = x1_c.size(2)
        x1_seq_len = x1_c.size(1)

        x1_c = x1_c.view(batch_size*x1_seq_len, char_num)

        x1_c_emb = self.char_embedding(x1_c)

        x1_c_emb_processed = self.char_rnn(x1_c_emb)  # bs * seq_len, char_num, emb_dim
        c_emb_dim_processed = x1_c_emb_processed.size(2)
        x1_emb = F.max_pool1d(x1_c_emb_processed.transpose(1, 2), kernel_size=char_num).contiguous().view(batch_size, x1_seq_len, c_emb_dim_processed)
        return x1_emb

def nan_num(v):
    nan_pos = v!=v
    print(torch.sum(nan_pos.long()))
