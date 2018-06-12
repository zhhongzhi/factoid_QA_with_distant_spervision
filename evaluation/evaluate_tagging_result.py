import sys
import os
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path += [curdir]
import argparse

from evaluation.tagging_evaluation_util import get_tagging_results
from evaluation.evaluation_stats_util import F1Stats
from evaluation.raw_result_parser import iter_results
from evaluation.ioutil import open_file

class Options:
    def __init__(self):
        self.raw_prediction = 'example/raw_prediction_example.txt'
        self.test_file = 'example/test_file_example.json.gz'
        self.schema = 'BIO2'
        self.fuzzy = False
        self.fuzzy = True


def get_val_result(options, labels):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('raw_prediction', default='raw_prediction_example.txt')
    # parser.add_argument('test_file', default='test_file_example.json.gz')
    # parser.add_argument('-s', '--schema', default='BIO2',
    #         choices=['BO', 'BO2', 'BIO', 'BIO2', 'BIO3'])
    # parser.add_argument('-o', '--output', default='-')
    # parser.add_argument('-f', '--fuzzy', action='store_true',
    #                     help='fuzzy evaluation')
    # options = parser.parse_args()
    stats = F1Stats(options.fuzzy, need_every_score=options.need_every_score)
    for q_tokens, e_tokens, tags, golden_answers in \
            iter_results(labels, options.test_file, options.schema):
        if q_tokens is None: continue # one question has been processed
        pred_answers = get_tagging_results(e_tokens, tags)
        stats.update(golden_answers, pred_answers)
    # print((stats.get_metrics_str()))
    return stats


if __name__ == '__main__':
    options = Options()
    get_val_result(options, labels=options.raw_prediction)
# chunk_f1=0.413521 chunk_precision=0.436303 chunk_recall=0.393000 true_chunks=4000 result_chunks=3603 correct_chunks=1572
# chunk_f1=0.460345 chunk_precision=0.485706 chunk_recall=0.437500 true_chunks=4000 result_chunks=3603 correct_chunks=1750
