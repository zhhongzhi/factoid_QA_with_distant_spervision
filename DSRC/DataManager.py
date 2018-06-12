# -*- encoding: utf-8 -*-
"""
    @author: hongzhi
    @time: 
    @des: 应该现在就划分好训练集和测试集和验证集了，回去写好这个文件，再新抓取的数据都当做训练集吧，如何防止训练集和测试集混叠呢？
    后面新抓取的数据，增量式添加吗？ok
"""
import json
import os
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import random
import jieba

from word_vocab import Vocab, CharVocab
import dat.file_pathes as dt_p
from dat.reformat_dat import RawWebQaReader


class DataManager:
    # 标起始位置的
    def __init__(self, params, log, no_need_dat=False, only_test_data=False, debug=False):
        self.params = params
        self.log = log
        self.char_vocab = CharVocab(params.char_vocab_file, params.char_emb_dim)
        self.debug = debug
        self.orig_train_num = None
        self.word_lengths = defaultdict(int)
        self.ds_qs = self.load_ds_questions_paraphrased(dt_p.paraphrased_train_dat_f)
        if not no_need_dat:  # 快速加载
            self.ds_smp_vs = None
            self.valid_dat = self.read_data(which='valid')
            self.test_dat = self.read_data(which='test')
            if not only_test_data:
                if debug:
                    self.params.ds_smp_num = 10000
                train_dat = self.read_data(which='train')
                self.q_c_idxs, self.q_masks, self.p_c_idxs, self.p_masks, self.qe_comm_ft, \
                    self.b_pos, self.e_pos, self.p_tokens = train_dat
                self.train_smp_num = len(self.qe_comm_ft)  # 这些是在读远监督的数据，怎么读原始的数据呢？
                self.ds_smp_idxs = np.arange(self.orig_train_num, self.train_smp_num)

                print(len(self.ds_smp_idxs), 'len of dis data')
    def load_both_dat(self, which):
        """

        :param which: train test or valid
        :return:
        """
        if which == 'train':
            dt_file = dt_p.web_qa_train_dat
        elif which == 'valid':
            dt_file = dt_p.web_qa_valid_dat
        else:
            dt_file = dt_p.web_qa_test_dat
        if self.params.kept_train_rate > 0.001 or which != 'train':
            smps = self.load_webqa_lns(dt_file)
            self.orig_train_num = int(self.params.kept_train_rate * len(smps))
            if which == 'train':
                smps = smps[:int(self.params.kept_train_rate * len(smps))]
        else:
            smps = []
        self.log.info('[{}] data sample num [{}]'.format(which, len(smps)))

        if self.params.with_dis_dat and which == 'train':
            # 远监督的数据不要用太多吧？
            if self.params.with_web_weight:
                print('loading dis data')
                ds_smps, self.ds_smp_vs = self.load_ds_lns_with_weights(dt_p.dis_questions_150m_html_f)

                print(len(ds_smps))
            else:
                ds_smps = self.load_ds_lns(dt_p.dis_questions_150m_html_f)
                # self.ds_smp_vs = self.calculate_simlarity(smps, ds_smps)
            smps += ds_smps
        if self.params.with_paraphrase_dat and which == 'train':
            ds_smps = self.load_ds_lns_paraphrased(dt_p.paraphrased_train_dat_f)
            print(len(ds_smps))
            if self.params.with_dis_dat:
                drop_num = min(len(ds_smps[:self.params.paraphrase_num]), self.params.ds_smp_num)
                # smps = smps[: -1 * self.params.paraphrase_num]
                smps = smps[: -1 * drop_num]
            smps += ds_smps[:self.params.paraphrase_num]
            # if self.ds_smp_vs is not None:
            #     self.ds_smp_vs /= self.ds_smp_vs.max()
            #     self.ds_smp_vs *= self.params.max_weight
            #
            # if self.params.smp_ds:
            #     self.ds_smp_vs /= self.ds_smp_vs.sum()

            # print('smp_weights', self.ds_smp_vs)
        print(which, len(smps))
        # q, st, s_pos_, e_pos_, ans 每个sample都用这种格式
        return smps

    def recover_rel_from_q(self, q):
        if u'的' in q:
            idx = q.index(u'的')
            rel = u''.join(q[idx + 1: -1])
            return rel
        else:
            return ''

    def calculate_simlarity(self, smps, dis_smps):
        # 这里有各种的similarity measurement 的方法  但是没有基于 search的结果弄的，在生成数据的时候弄好吧
        in_domain_q_v = defaultdict(float)
        in_domain_ans_v = defaultdict(float)
        for q, st, s_pos, e_pos, ans in smps:
            for w in q:
                if w in [u'的', u'?', u'？']:
                    continue
                in_domain_q_v[w] += 1
            for w in ans[0]:
                in_domain_ans_v[w] += 1
        # now 归一化
        in_domain_q_v = self.l2_norm(in_domain_q_v)
        # in_domain_ans_v = self.l2_norm(in_domain_ans_v)
        vs = []
        rels = []
        for q, st, s_pos, e_pos, ans in dis_smps:
            rel = self.recover_rel_from_q(q)
            rels.append(rel)
        rel_cnt = Counter(rels)
        rel_cnt = {k: 1 / (v + 100.0) for k, v in rel_cnt.items()}
        sum_cnt = sum(rel_cnt.values())
        rel_cnt = {k: v / sum_cnt for k, v in rel_cnt.items()}

        for rel, (q, st, s_pos, e_pos, ans) in zip(rels, dis_smps):
            q_vec = defaultdict(float)
            ans_vec = defaultdict(float)
            for w in q:
                if w in [u'的', u'?', u'？']:
                    continue
                q_vec[w] += 1
            for w in jieba.cut(ans):
                ans_vec[w] += 1
            q_vec = self.l2_norm(q_vec)
            # ans_vec = self.l2_norm(ans_vec)
            v_q = self.dot(q_vec, in_domain_q_v)
            # v_ans = self.dot(ans_vec, in_domain_ans_v)
            vs.append(rel_cnt[rel])
        # sum_vs = sum(vs) / len(dis_smps)
        # max_v = max(vs) / 5.0
        sum_vs = sum(vs)
        # vs = [v / max_v for v in vs]  #
        vs = [v / sum_vs for v in vs]  #
        print(np.array(vs))
        return np.array(vs)

    def l2_norm(self, k_v_vec):
        l2_q_v = sum([v**2 for v in k_v_vec.values()])
        k_v_vec = {k: v/l2_q_v for k, v in k_v_vec.items()}
        return k_v_vec

    def dot(self, k_v_vec1, k_v_vec2):
        """

        :param k_v_vec1: this is smaller
        :param k_v_vec2:
        :return:
        """
        v = sum([v1 * k_v_vec2.get(k, 0.0) for k, v1 in k_v_vec1.items()])
        return v

    def load_ds_lns(self, dat_file):
        """
        加上限制，entity要出现
        :param dat_file:
        :return:
        """
        inf = open(dat_file, 'r')
        if self.params.ds_smp_num < 100000:
            lns = [inf.readline() for i in range(self.params.ds_smp_num * 2)]
        else:
            lns = inf.readlines()[:self.params.ds_smp_num * 2]
            # indexes = np.array(np.random.randint(low=0, high=len(lns), size=self.params.ds_smp_num * 2))   # 随机选一些
            # lns = [lns[idx] for idx in indexes]

        ln_infos = [json.loads(ln) for ln in lns]
        smps = []
        for ln_info in ln_infos:
            q, st, s_pos_, e_pos_, value = ln_info['question'], ln_info['evidence'], ln_info['start_pos'], \
                                           ln_info['end_pos'], ln_info['answer']
            if u''.join(q) in self.ds_qs:
                continue
            if e_pos_ >= self.params.p_max_len:
                continue
            title = ln_info['title']
            # if ent in st
            common_w_num = len(set(jieba.lcut(title)).union(set(st)))
            if common_w_num == 0:
                continue
            if self.params.special_q_flag:
                q = [self.char_vocab.DSStart] + q + [self.char_vocab.DSEnd]   # add special token
            # q = [self.char_vocab.DSStart] +  [self.char_vocab.DSEnd]
            smps.append((q, st, s_pos_, e_pos_, value))
            if len(smps) == self.params.ds_smp_num:
                break
        self.log.info('config distant training data num [{}], distant training data kept [{}]'.format(len(smps), len(smps)))
        return smps

    def load_ds_lns_with_weights(self, dat_file):
        """
        加上限制，entity要出现
        :param dat_file:
        :return:
        """
        triple_weights = json.load(open(dt_p.triple_weight_by_search_f))
        inf = open(dat_file, 'r')
        lns = inf.readlines()
        # if self.params.select_train_smps:
        #     lns = inf.readlines()
        # elif self.params.ds_smp_num < 100000:
        #     lns = [inf.readline() for i in range(self.params.ds_smp_num * 2)]
        # else:
        #     lns = inf.readlines()

        ln_infos = [json.loads(ln) for ln in lns]
        smp_idxs = list(range(len(lns)))
        if not self.params.select_train_smps:
            random.shuffle(smp_idxs)
        smps = []
        weights = []
        # for ln_info in ln_infos:
        for smp_idx in smp_idxs:
            ln_info = ln_infos[smp_idx]
            q, st, s_pos_, e_pos_, value = ln_info['question'], ln_info['evidence'], ln_info['start_pos'], \
                                           ln_info['end_pos'], ln_info['answer']
            if u''.join(q) in self.ds_qs:  # 去掉被paraphrased的
                continue
            if e_pos_ >= self.params.p_max_len:
                continue
            title = ln_info['title']
            key = ln_info['key']

            triple_str = u'\t'.join([title, key, value, ''])

            if triple_str not in triple_weights:
                continue
            weight = triple_weights[triple_str]
            weights.append(weight)
            common_w_num = len(set(jieba.lcut(title)).union(set(st)))
            if common_w_num == 0:
                continue
            if self.params.special_q_flag:
                q = [self.char_vocab.DSStart] + q + [self.char_vocab.DSEnd]   # add special token
            # q = [self.char_vocab.DSStart] +  [self.char_vocab.DSEnd]
            smps.append((q, st, s_pos_, e_pos_, value))

            if len(smps) == self.params.ds_smp_num and not self.params.select_train_smps:
                break
        if self.params.select_train_smps:
            weight_th = sorted(weights, reverse=True)[self.params.ds_smp_num]
            kept_smp_idxs = []
            for idx, weight in enumerate(weights):
                if weight > weight_th:
                    kept_smp_idxs.append(idx)
                if len(kept_smp_idxs) == self.params.ds_smp_num:
                    break
            for idx, weight in enumerate(weights):
                if len(kept_smp_idxs) == self.params.ds_smp_num:
                    break
                if weight == weight_th:
                    kept_smp_idxs.append(idx)

            smps = [smps[idx] for idx in kept_smp_idxs]
            self.log.info('distant select weight th: [{}]'.format(weight_th))
        self.log.info('config distant training data num [{}], distant training data kept [{}]'.format(len(smps), len(smps)))
        return smps, np.array(weights)

    def load_ds_lns_with_weightsrelease(self, dat_file):
        """
        加上限制，entity要出现
        :param dat_file:
        :return:
        """
        triple_weights = json.load(open(dt_p.triple_weight_by_search_f))
        inf = open(dat_file, 'r')
        lns = inf.readlines()
        ln_infos = [json.loads(ln) for ln in lns]
        smp_idxs = list(range(len(lns)))
        if not self.params.select_train_smps:
            random.shuffle(smp_idxs)
        smps = []
        weights = []
        for smp_idx in smp_idxs:
            ln_info = ln_infos[smp_idx]
            q, st, s_pos_, e_pos_, value = ln_info['question'], ln_info['evidence'], ln_info['start_pos'], \
                                           ln_info['end_pos'], ln_info['answer']
            if u''.join(q) in self.ds_qs:  # 去掉被paraphrased的
                continue
            if e_pos_ >= self.params.p_max_len:
                continue
            title = ln_info['title']
            key = ln_info['key']

            triple_str = u'\t'.join([title, key, value, ''])

            if triple_str not in triple_weights:
                continue
            weight = triple_weights[triple_str]
            weights.append(weight)
            common_w_num = len(set(jieba.lcut(title)).union(set(st)))
            if common_w_num == 0:
                continue
            if self.params.special_q_flag:
                q = [self.char_vocab.DSStart] + q + [self.char_vocab.DSEnd]   # add special token
            # q = [self.char_vocab.DSStart] +  [self.char_vocab.DSEnd]
            smps.append((q, st, s_pos_, e_pos_, value))

            if len(smps) == self.params.ds_smp_num and not self.params.select_train_smps:
                break
        # print(len(smps))
        # assert 1==2
        if self.params.select_train_smps:
            weight_th = sorted(weights, reverse=True)[self.params.ds_smp_num]
            kept_smp_idxs = []
            for idx, weight in enumerate(weights):
                if weight > weight_th:
                    kept_smp_idxs.append(idx)
                if len(kept_smp_idxs) == self.params.ds_smp_num:
                    break
            for idx, weight in enumerate(weights):
                if len(kept_smp_idxs) == self.params.ds_smp_num:
                    break
                if weight == weight_th:
                    kept_smp_idxs.append(idx)

            smps = [smps[idx] for idx in kept_smp_idxs]
            self.log.info('distant select weight th: [{}]'.format(weight_th))
        self.log.info('config distant training data num [{}], distant training data kept [{}]'.format(len(smps), len(smps)))
        return smps, np.array(weights)

    def load_ds_questions_paraphrased(self, dat_file):
        """
        加上限制，entity要出现
        :param dat_file:
        :return:
        """
        inf = open(dat_file, 'r')
        lns = inf.readlines()
        ln_infos = [json.loads(ln) for ln in lns]
        qs = []
        smps_paraphrased = []
        for ln_info in ln_infos:
            q_paraphrased = ln_info['question']
            q, st, s_pos_, e_pos_, value = ln_info['question_original'], ln_info['evidence'], ln_info['start_pos'], \
                                           ln_info['end_pos'], ln_info['answer']
            qs.append(u''.join(q))
        return set(qs)

    def load_ds_lns_paraphrased(self, dat_file):
        """
        加上限制，entity要出现
        :param dat_file:
        :return:
        """
        inf = open(dat_file, 'r')
        lns = inf.readlines()
        ln_infos = [json.loads(ln) for ln in lns]
        smps = []
        smps_paraphrased = []
        for ln_info in ln_infos:
            q_paraphrased = ln_info['question']
            q, st, s_pos_, e_pos_, value = ln_info['question_original'], ln_info['evidence'], ln_info['start_pos'], \
                                           ln_info['end_pos'], ln_info['answer']
            if e_pos_ >= self.params.p_max_len:
                continue
            title = ln_info['title']
            # if ent in st
            common_w_num = len(set(jieba.lcut(title)).union(set(st)))
            if common_w_num == 0:
                continue
            q = [self.char_vocab.DSStartP] + q + [self.char_vocab.DSEndP]
            smps.append((q, st, s_pos_, e_pos_, value))
            # print(q_paraphrased)
            smps_paraphrased.append((q_paraphrased, st, s_pos_, e_pos_, value))
        self.log.info('config distant training data num [{}], paraphrased distant training data kept [{}]'.format(len(lns), len(smps)))
        # return smps + smps_paraphrased
        return smps_paraphrased
        # return smps

    def load_webqa_lns(self, file_path):
        all_infos = RawWebQaReader.read_file(file_path, use_refined_flag=True, debug=self.debug)
        smps = []
        for q_infos in all_infos:
            q = q_infos['question_tokens']
            for evd in q_infos['evidences']:
                if evd['type'] != 'positive' and 'training' in file_path:
                    continue
                st = evd['evidence_tokens']
                if 'train' in file_path:
                    s_pos = evd['begin']
                    e_pos = evd['end']
                else:
                    s_pos = -1
                    e_pos = -1
                if e_pos >= self.params.p_max_len:
                    continue
                ans = evd['golden_answers']
                smps.append((q, st, s_pos, e_pos, ans))
        return smps

    def read_data(self, which):
        p_tokens, ans_tokens, b_pos, e_pos, flags = [], [], [], [], []
        q_c_idxs, p_c_idxs = [], []
        qe_comm_ft = []
        len_q_idxs, len_p_idxs = [], []
        smps = self.load_both_dat(which=which)
        for smp in smps:
            q, st, s_pos_, e_pos_, value = smp
            q_c_idxs.append(self.get_char_idxs(q, max_seq_len=self.params.q_max_len))
            p_c_idxs.append(self.get_char_idxs(st, max_seq_len=self.params.p_max_len))
            b_pos.append(s_pos_)
            e_pos.append(e_pos_)

            ans_tokens.append(value)
            q_ws_set = set(q)
            qe_comm_ft.append([1 if p_w in q_ws_set else 0 for p_w in st])
            len_q_idxs.append([0] * len(q))
            len_p_idxs.append([0] * len(st))
            p_tokens.append(st)
        _, q_masks = self.padding_sequence(len_q_idxs, max_len=self.params.q_max_len)
        _, p_masks = self.padding_sequence(len_p_idxs, max_len=self.params.p_max_len)
        qe_comm_ft, _ = self.padding_sequence(qe_comm_ft, max_len=self.params.p_max_len)
        p_c_idxs = np.array(p_c_idxs)
        q_c_idxs = np.array(q_c_idxs)
        b_pos, e_pos = [np.array(i) for i in (b_pos, e_pos)]
        return q_c_idxs, q_masks, p_c_idxs, p_masks, qe_comm_ft, b_pos, e_pos, p_tokens

    def get_char_idxs(self, seq_ws, max_seq_len):
        char_idxs = []
        pad_char_idx = self.char_vocab.word2id(self.char_vocab.PAD_TOKEN)
        if len(seq_ws) > max_seq_len:
            seq_ws = seq_ws[:max_seq_len]
        else:
            seq_ws = seq_ws + [''] * (max_seq_len - len(seq_ws))
        for w in seq_ws:
            self.word_lengths[len(w)] += 1
            w_char_idxs = self.char_vocab.seqword2id(w)
            if len(w_char_idxs) < self.params.char_max_len:
                w_char_idxs = [pad_char_idx] * (self.params.char_max_len - len(w_char_idxs)) + w_char_idxs
            elif len(w_char_idxs) > self.params.char_max_len:
                w_char_idxs = w_char_idxs[:self.params.char_max_len]
            char_idxs.append(w_char_idxs)
        return char_idxs

    def padding_sequence(self, idxs, max_len):
        new_idxs = []
        masks = []
        for idx_smp in idxs:
            to_add = max_len - len(idx_smp)
            if to_add >= 0:
                new_idxs.append(idx_smp + [0] * to_add)
                masks.append([0] * len(idx_smp) + [1] * to_add)
            else:
                new_idxs.append(idx_smp[:max_len])
                masks.append([0] * max_len)
        return np.array(new_idxs), np.array(masks)

    def get_train_batch(self, batch_size, ds_flag):
        weights = None
        if ds_flag:
            if self.params.smp_ds:
                indexes = np.random.choice(self.ds_smp_idxs, size=batch_size, replace=False, p=self.ds_smp_vs)
            else:
                indexes = np.array(np.random.randint(low=self.orig_train_num, high=self.train_smp_num, size=batch_size))
                # print((self.orig_train_num, self.train_smp_num))
            # if self.ds_smp_vs is not None:
            #     weights = self.ds_smp_vs[indexes-self.orig_train_num]
        else:
            indexes = np.array(np.random.randint(self.orig_train_num, size=batch_size))
        # indexes = np.array(np.random.randint(self.train_smp_num, size=batch_size))
        # indexes = np.array(np.random.randint(self.orig_train_num, size=batch_size))

        q_masks_batch = self.q_masks[indexes]
        p_masks_batch = self.p_masks[indexes]
        qe_comm_fts_batch = self.qe_comm_ft[indexes]
        q_c_idxs_batch = self.q_c_idxs[indexes]
        b_batch = self.b_pos[indexes]
        e_batch = self.e_pos[indexes]
        p_c_idxs_batch = self.p_c_idxs[indexes]
        weights = None
        return p_c_idxs_batch, qe_comm_fts_batch, p_masks_batch, q_c_idxs_batch, \
               q_masks_batch, b_batch, e_batch, weights

    def get_train_batch_new(self, batch_size, paraphrased):
        weights = None
        if paraphrased:
            indexes = np.array(np.random.randint(low=int(self.train_smp_num / 2), high=self.train_smp_num, size=batch_size))
        else:
            indexes = np.array(np.random.randint(low=0, high=int(self.train_smp_num / 2), size=batch_size))
        q_masks_batch = self.q_masks[indexes]
        p_masks_batch = self.p_masks[indexes]
        qe_comm_fts_batch = self.qe_comm_ft[indexes]
        q_c_idxs_batch = self.q_c_idxs[indexes]
        b_batch = self.b_pos[indexes]
        e_batch = self.e_pos[indexes]
        p_c_idxs_batch = self.p_c_idxs[indexes]
        return p_c_idxs_batch, qe_comm_fts_batch, p_masks_batch, q_c_idxs_batch, \
               q_masks_batch, b_batch, e_batch, weights

    def trans_to_xs(self, questions, evidences):
        """
        分词之后的
        :param questions:
        :param evidences:
        :return:
        """
        q_max_len = min(max([len(q) for q in questions]), self.params.q_max_len)
        doc_max_len = min(max([len(doc) for doc in evidences]), self.params.p_max_len)
        q_xs, doc_xs = [], []
        qe_comm_fts = []
        len_p_idxs, len_q_idxs = [], []
        for q, evi in zip(questions, evidences):
            q_x = self.get_char_idxs(q, max_seq_len=q_max_len)
            q_xs.append(q_x)
            doc_xs.append(self.get_char_idxs(evi, max_seq_len=doc_max_len))
            q_ws = set(q)
            qe_comm_fts.append([1 for w in evi if w in q_ws])
            len_q_idxs.append([0] * len(q))
            len_p_idxs.append([0] * len(evi))
        qe_comm_fts, _ = self.padding_sequence(qe_comm_fts, max_len=doc_max_len)
        _, p_masks = self.padding_sequence(len_p_idxs, max_len=doc_max_len)
        _, q_masks = self.padding_sequence(len_q_idxs, max_len=q_max_len)
        doc_xs, qe_comm_fts, p_masks, q_xs, q_masks = [np.array(i) for i in (doc_xs, qe_comm_fts, p_masks, q_xs, q_masks)]
        return doc_xs, qe_comm_fts, p_masks, q_xs, q_masks

    def length_of_distant_data(self):

        # ds_smps, self.ds_smp_vs = self.load_ds_lns_with_weights(dt_p.paraphrased_train_dat_f)
        ds_smps, self.ds_smp_vs = self.load_ds_lns_with_weights(dt_p.dis_questions_150m_html_f)

        # if self.params.with_paraphrase_dat and which == 'train':
        #     ds_smps = self.load_ds_lns_paraphrased(dt_p.paraphrased_train_dat_f)
        #     smps = smps[: -1 * self.params.paraphrase_num]
        #     smps += ds_smps[:self.params.paraphrase_num]
        q_len = Counter()
        ans_len = Counter()
        st_len = Counter()
        for q, st, s_pos_, e_pos_, ans in ds_smps:
            q_len[len(q)] += 1
            st_len[len(st)] += 1
            ans_len[len(jieba.lcut(ans))] += 1
        print(q_len)
        print(st_len)
        print(ans_len)


if __name__ == '__main__':
    DataManager()
    pass
