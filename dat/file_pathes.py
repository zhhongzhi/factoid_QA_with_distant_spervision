# -*- encoding: utf-8 -*-
"""
    @author: hongzhi
    @time: 
    @des: 
"""
import os
import pickle

dp = os.path.abspath(os.path.dirname(__file__)) + '/'


# ###################################################################
test_data_ent_names_f = dp + 'test_data_ent_names.h5'
test_data_ent_names_see_f = dp + 'test_data_ent_names.txt'

web_qa_mother_dp = '{}/WebQA.v1.0/'.format(dp)
web_qa_emb_dp = web_qa_mother_dp + 'embedding/'
web_qa_dp = web_qa_mother_dp + 'data/'

web_qa_train_dat = web_qa_dp + 'training.json.gz'
web_qa_test_dat = web_qa_dp + 'test.ann.json.gz'
web_qa_ir_test_dat = web_qa_dp + 'test.ir.json.gz'
web_qa_valid_dat = web_qa_dp + 'validation.ann.json.gz'
web_qa_ir_valid_dat = web_qa_dp + 'validation.ir.json.gz'
refined_train_lbs = web_qa_dp + 'refined_lbs.pkl'


webqa_train_qs_f = web_qa_dp + 'webqa_train_questions.txt'
webqa_valid_qs_f = web_qa_dp + 'webqa_valid_questions.txt'
webqa_test_qs_f = web_qa_dp + 'webqa_test_questions.txt'

clf_train_f = dp + 'clf_train.txt'
clf_valid_f = dp + 'clf_valid.txt'
clf_test_f = dp + 'clf_test.txt'
clf_test_see_f = dp + 'clf_test_see.txt'


clf_train_f_with_pt = dp + 'clf_train_with_pt.txt'   # with pt 是啥意思？  手写了几个生成问题的pattern 模板
clf_valid_f_with_pt = dp + 'clf_valid_with_pt.txt'
clf_test_f_with_pt = dp + 'clf_test_with_pt.txt'
clf_test_see_f_with_pt = dp + 'clf_test_see_with_pt.txt'

chars_f = dp + 'chars.txt'

distant_paraphrase_f = dp + 'paraphrase_questions.txt'
distant_paraphrase_f = dp + 'paraphrase_questions_new.txt'
distant_paraphrase_f = dp + 'paraphrase_questions_new0111.txt'
dis_questions_150m_html_f = dp + 'questions_dis_data_150htmls_using_abstext.txt'
paraphrased_train_dat_f = dp + 'paraphrased_train_dat.txt'
paraphrased_train_dat_f = dp + 'paraphrased_train_dat_new.txt'
paraphrased_train_dat_f = dp + 'paraphrased_train_dat_new0111.txt'
paraphrased_train_dat_f = dp + 'new_mined_paraphrase0123.txt'
paraphrased_train_dat_f = dp + 'new_mined_paraphrase0124.txt'
# 用哪一个text呢？知道上带有的？也可以 easy

#######################################################################
triples_f = dp + 'triples.txt'

triple_weight_by_search_f = dp + 'triple_weight_by_search.txt'
# triple_weight_by_search_f = dp + 'triple_weight_by_search_3_7.txt'
# triple_weight_by_search_f = dp + 'triple_weight_by_search_10_0.txt'

distant_train_f = ''

if __name__ == '__main__':
    # 先把原来的模型跑起来。
    pass
