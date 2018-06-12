# -*- encoding: utf-8 -*-
"""
    @author: hongzhi
    @time: 
    @des: 
"""
import gzip
import json
import pickle

from collections import Counter

import dat.file_pathes as dt


class RawWebQaReader:
    def __init__(self):
        self.train_dat = self.read_file(dt.web_qa_train_dat)
        self.test_dat = self.read_file(dt.web_qa_ir_test_dat)
        self.test_dat = self.read_file(dt.web_qa_test_dat)
        self.valid_dat = self.read_file(dt.web_qa_valid_dat)

    @staticmethod
    def read_file(file_path, use_refined_flag=False, debug=False):
        f = gzip.open(file_path, 'rb')  #
        if 'training' in file_path and debug:
            lns = [f.readline() for i in range(1000)]
        else:
            lns = f.readlines()
        oo, tt = 0, 0
        all_infos = []
        all_refined_lbs = []
        if use_refined_flag and 'training' in file_path:
            all_refined_lbs = pickle.load(open(dt.refined_train_lbs, 'rb'))
        refine_idx = 0
        for ln in lns:
            q_infos = {'evidences': []}
            raw_q_infos = json.loads(ln)
            q_infos['question_tokens'] = raw_q_infos['question_tokens']
            for raw_evd in raw_q_infos['evidences']:
                evd = {k: raw_evd[k] for k in ['evidence_tokens', 'type', 'golden_answers', 'q-e.comm_features']}
                if 'training' in file_path:
                    if raw_evd['type'] == 'positive':
                        tt += 1
                        lbs = raw_evd['golden_labels']
                        cnt = Counter(lbs)
                        if cnt.get('b', 0) > 1:   # 这里还多虑掉了一部分训练数据 将来加进来，很烦的哈  训练数据而已的   如果模型结果好，应该也不在意这一点点的
                            oo += 1
                            if not use_refined_flag:
                                continue
                            else:
                                lbs = all_refined_lbs[refine_idx]
                                refine_idx += 1
                        evd['begin'] = lbs.index('b')
                        end = evd['begin'] + 1
                        while end < len(lbs) and lbs[end] == 'i':
                            end += 1
                        evd['end'] = end
                q_infos['evidences'].append(evd)
            all_infos.append(q_infos)
        # assert refine_idx == len(all_refined_lbs)
        print('question num:{}, total_pos_num:{}, more than one b:{}, refined: {}'.format(len(all_infos), tt, oo, refine_idx))
        return all_infos

    @staticmethod
    def read_evidences_need_refine(file_path=dt.web_qa_train_dat):
        f = gzip.open(file_path, 'rb')  #
        lns = f.readlines()
        oo, tt = 0, 0
        all_infos = []
        for ln in lns: #[:1000]
            q_infos = {'evidences': []}
            raw_q_infos = json.loads(ln)
            q_infos['question_tokens'] = raw_q_infos['question_tokens']
            for raw_evd in raw_q_infos['evidences']:
                evd = {k: raw_evd[k] for k in ['evidence_tokens', 'type', 'golden_answers', 'q-e.comm_features', 'golden_labels']}
                if raw_evd['type'] == 'positive':
                    tt += 1
                    lbs = raw_evd['golden_labels']
                    cnt = Counter(lbs)
                    if cnt.get('b', 0) > 1:
                        oo += 1
                        q_infos['evidences'].append(evd)
                    else:
                        continue
            if len(q_infos):
                all_infos.append(q_infos)
        return all_infos

# build a vocab map them to idx then i can train it.   point net loss function 怎么写？


def write_all_questions():
    all_infos = RawWebQaReader.read_file(dt.web_qa_train_dat)
    questions = []
    for info in all_infos:
        q = info['question_tokens']
        q = u''.join(q) + u'\n'
        questions.append(q)
    of = open('web_qa_questions', 'w')
    of.writelines(questions)
    of.close()


if __name__ == '__main__':
    # RawWebQaReader()
    # test_dat = RawWebQaReader.read_file(dt.web_qa_test_dat)
    # fo = open('test.txt', 'w', encoding='utf-8')
    # lns = []
    # for q_info in test_dat:
    #     lns.append(u' '.join(q_info['question_tokens'])+'\n')
    #     for evi in q_info['evidences']:
    #         # lns.append('----'*10+'\n')
    #         lns.append(u' '.join(evi['golden_answers'][0])+'\n')
    #         lns.append(u' '.join(evi['evidence_tokens'])+'\n')
    #     lns.append('==='*10+'\n')
    # fo.writelines(lns)
    write_all_questions()
    pass
# question num:36181, total_pos_num:139199, more than one b:42971  还有这么多没用的数据

