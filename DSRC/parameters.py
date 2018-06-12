# -*- encoding: utf-8 -*-
import math

import dat.file_pathes as dt


class SingleModelParameters:
    def __init__(self):

        self.q_max_len = 20
        self.p_max_len = 80
        self.char_max_len = 3

        self.q_pre_dim = 200
        self.p_pre_dim = self.q_pre_dim

        self.match_lstm_dim = 150   # fts dim
        self.ft_cnn_size = 100

        # about training
        self.batch_num = 10000000
        self.batch_size = 64
        self.valid_batch_size = 128
        self.kernel_num = 100
        self.kernel_sizes = [1, 3, 5]

        self.tune_partial = 0  # 不需要
        self.emb_dim = 64
        self.embedding_dim = self.emb_dim
        self.char_emb_dim = 64
        self.char_vocab_file = dt.chars_f

        # model structure relevant
        self.hidden_size = 100
        self.doc_layers = 9
        self.question_layers = 9
        self.match_layers = 1   # match lstm
        self.char_rep_layers = 1   # match lstm
        self.concat_rnn_layers = False
        self.question_merge = 'self_attn'
        self.num_features = 1  #
        self.use_qemb = True
        self.concat_rnn_layers = False
        self.use_interaction = True
        self.res_net = False

        # 正则
        self.dropout_rnn = 0.15
        self.dropout_rnn_output = True
        self.dropout_emb = 0.5

        # 优化器参数
        self.optimizer = 'adamax'
        self.learning_rate = 0.001
        self.weight_decay = 0
        self.momentum = 0
        self.grad_clipping = 20
        self.reduce_lr = 0.0

        train_idx = 5

        # 文件保存
        self.model_dir = './models/rl_models{}/'.format(train_idx)
        self.log_file = './logs/rl_models0{}.log'.format(train_idx)
        self.special_q_flag = True

        self.cuda = True
        self.resume_flag = False
        self.pretrained_model = '/home/hongzhi/wp/DSRC/ForPaper/models/rl_models16/checkpoint_epoch_300000.pt'  # why this epoch?

        self.max_weight = 1.0
        self.smp_ds = False

        self.with_web_weight = True
        self.with_paraphrase_dat = False

        # purely trained on the generated data # results showed in Figure 5. Factoid QA via distant supervision.

        # DSBasic
        if train_idx == 0:
            self.with_dis_dat = False     # whether to use all the paraphrased questions
            self.kept_train_rate = 10e-6  # no labeled kept
            self.select_train_smps = False

            self.ds_smp_num = 20 * 1000 # the sample number varies from 20k to 105k

        # DS + SS
        if train_idx == 1:
            self.with_dis_dat = False
            self.kept_train_rate = 10e-6
            self.select_train_smps = True

            self.ds_smp_num = 20 * 1000  # the sample number varies from 20k to 105k

        # DS+SS+DP
        if train_idx == 2:
            self.with_dis_dat = True
            self.kept_train_rate = 10e-6
            self.select_train_smps = True

            self.ds_smp_num = 20 * 1000  # the sample number varies from 20k to 105k

        # configurations for results showed in Figure 6. Improved factoid QA with distant supervision.
        if train_idx == 3: # SLBasic
            self.with_dis_dat = False
            self.select_train_smps = False
            self.ds_smp_num = 0

            self.kept_train_rate = 0.10 # varies from 0.01 to 1

        if train_idx == 4:  # DS + SL
            self.with_dis_dat = False
            self.select_train_smps = False
            self.ds_smp_num = 0
            self.resume_flag = True
            self.pretrained_model = './models/checkpoint_epoch_300000.pt'
            # This model is pre-trained with 320k DSBasic data (train_idx=0);
            # configurations like DS+SS+SL DS+SS+DP+SL could be configured similarly

            self.kept_train_rate = 0.5 # 0.5 or 1 is reported

        if train_idx == 5:  # SL+  Table 5 and Figure 7
            self.with_dis_dat = True
            self.select_train_smps = False

            self.ds_smp_num = 320 * 1000  # more data should generate better result
            self.kept_train_rate = 1.0

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value
