# -*- encoding: utf-8 -*-
import logging
import sys
import os
import json
import random

import torch

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curdir)
sys.path.append(os.path.split(curdir)[0])
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(1)

from DSRC.DataManager import DataManager
from DSRC.parameters import SingleModelParameters
from DSRC.SingleDocReaderPNet import DocReaderModel
from Evaluation import eval_it


class Train:
    def __init__(self, params=SingleModelParameters(), no_need_dat=False, dt=None, only_test_data=False, need_test_raw_info=False):
        self.params = params
        self.log = self.loger()
        self.log.info(json.dumps(dict(params), indent=True))

        if dt is not None:
            self.dt = dt
        else:
            self.dt = DataManager(self.params, log=self.log, no_need_dat=no_need_dat, only_test_data=only_test_data)
        if self.params.resume_flag:
            self.model = self.resume()
        else:
            model_dir = self.params.model_dir
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            self.model = DocReaderModel(opt=self.params,
                                        char_embedding=self.dt.char_vocab.vs, state_dict=None)
        self.model.cuda()
        self.best_valid_f1 = 0

    def resume(self):
        self.log.info('[loading previous model...]')
        checkpoint = torch.load(self.params.pretrained_model)
        opt = checkpoint['config']
        state_dict = checkpoint['state_dict']
        model = DocReaderModel(opt=opt,
                               char_embedding=self.dt.char_vocab.vs, state_dict=state_dict)
        return model

    def valid_it(self, bt):
        for valid_or_test in ('valid', 'test'):
            self.log.info('==={}==='.format(valid_or_test))
            lbs = self.model.predict_on_valid_or_test_constrained(self.dt, valid_or_test)
            # lbs_constrained = self.model.predict_on_valid_or_test(self.dt, valid_or_test)
            for lbs_ in [lbs]: #, lbs_constrained]:
                res, fussy_res = eval_it(p_lbs=lbs_, valid_or_test=valid_or_test)
                if valid_or_test == 'valid':
                    valid_f1 = fussy_res.get_f1()
                    if valid_f1 > self.best_valid_f1:
                        self.log.info('new best valid f1 found')
                        self.best_valid_f1 = valid_f1
                        if bt > 30000 or 3e4 <= bt <3e4+2000:
                            model_file = os.path.join(self.params.model_dir, 'checkpoint_epoch_{}.pt'.format(bt))
                            self.log.info('save model at: {}'.format(model_file))
                            self.model.save(model_file, bt)

                self.log.info(res.get_metrics_str())
                self.log.info(fussy_res.get_metrics_str())   # 好想替换掉这种evaluation的方法啊！

    def model_train(self):
        self.log.info('sample number of distant supervision learning: {}'.format(len(self.dt.q_c_idxs)))

        tune_smp_num = len(self.dt.q_c_idxs)
        mini_batch_num = int(tune_smp_num / self.params.batch_size)
        if mini_batch_num > 1000:
            mini_batch_num = 1000
        for bt in range(self.params.batch_num):
            # self.model.update(self.dt.get_train_batch(batch_size=self.params.batch_size, ds_flag=True), ds_flag=False)
            # if bt % 3 == 0:

            if self.params.with_dis_dat or self.params.with_paraphrase_dat:
                self.model.update(self.dt.get_train_batch(batch_size=self.params.batch_size, ds_flag=True))
            if self.params.kept_train_rate > 0.001:   # fine tune 才对吧？   因此，在一定的epoch之后用original domain的数据进行训练吧？fine

                    self.model.update(self.dt.get_train_batch(batch_size=self.params.batch_size, ds_flag=False))
            if bt % mini_batch_num == 0:   # 到底怎么办好一点呢？  fine tune 或许在数据量小的时候结果会好一点吧？
                self.log.info('updates[{0:6}] train loss[{1:.5f}]]'.format(self.model.updates, self.model.train_loss.avg))
                self.valid_it(self.model.updates)
                self.model.zero_loss()

    def model_train_fine_tune(self):
        self.log.info('pretrained result')
        self.valid_it(0)

        self.log.info('sample number of WebQA: {}'.format(len(self.dt.q_c_idxs)))
        self.log.info('start fine tune'.format(len(self.dt.q_c_idxs)))
        tune_smp_num = len(self.dt.q_c_idxs)
        # mini_batch_num = int(tune_smp_num / self.params.batch_size)
        mini_batch_num = min(int(self.dt.orig_train_num / self.params.batch_size), 1000)

        for bt in range(self.params.batch_num):
            if bt % 5 == 0:# and bt > 38000:
                self.model.update(self.dt.get_train_batch(batch_size=self.params.batch_size, ds_flag=True))
            self.model.update(self.dt.get_train_batch(batch_size=self.params.batch_size, ds_flag=False))
            # if bt % mini_batch_num == 0:   # 到底怎么办好一点呢？  fine tune 或许在数据量小的时候结果会好一点吧？
            if bt % mini_batch_num == 0:   # 到底怎么办好一点呢？  fine tune 或许在数据量小的时候结果会好一点吧？
                self.log.info('updates[{0:6}] train loss[{1:.5f}]]'.format(self.model.updates, self.model.train_loss.avg))

            if bt % mini_batch_num == 0:
                self.valid_it(bt)

    def loger(self):
        # setup logger
        log = logging.getLogger(__name__)
        log.setLevel(logging.DEBUG)
        fh = logging.FileHandler(self.params.log_file)
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        log.addHandler(fh)
        log.addHandler(ch)
        return log

if __name__ == '__main__':
    Train().model_train()
    # Train().model_train_fine_tune()
    # t = Train(no_need_dat=True)
    # t.dt.length_of_distant_data()
    # Train().valid_it()

    # params=SingleModelParameters()
    #
    # log = logging.getLogger(__name__)
    #
    # dt = DataManager(params, log=log, no_need_dat=True, only_test_data=False)
    # import dat.file_pathes as dt_p
    # dt.load_ds_lns_with_weightsrelease(dat_file=dt_p.dis_questions_150m_html_f)