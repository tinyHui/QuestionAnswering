from multiprocessing import Pool
from functools import partial
from collections import defaultdict
from preprocess.data import ReVerbPairs
from train import CCA_FILE
from preprocess.feats import FEATURE_OPTS, data2feats
from CCA import distance
import argparse
import pickle as pkl
import logging
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
INF_FREQ = 1000
OUTPUT_FILE_TOP1 = './data/reverb-test-with_dist.top1.%s.txt'
OUTPUT_FILE_TOP5 = './data/reverb-test-with_dist.top5.%s.txt'
OUTPUT_FILE_TOP10 = './data/reverb-test-with_dist.top10.%s.txt'


def loader(feat, Q_k, A_k):
    indx, (_, (crt_q_v, crt_a_v, _, _)) = feat
    proj_q = crt_q_v.dot(Q_k)
    proj_a = crt_a_v.dot(A_k)
    return indx, distance(proj_q, proj_a)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Define training process.')
    parser.add_argument('--feature', type=str, default='bow', help="Feature option: %s" % (", ".join(FEATURE_OPTS)))
    parser.add_argument('--freq', type=int, default=INF_FREQ, help='Information print out frequency')

    args = parser.parse_args()
    feature = args.feature
    INF_FREQ = args.freq

    OUTPUT_FILE_TOP1 = OUTPUT_FILE_TOP1 % feature
    OUTPUT_FILE_TOP10 = OUTPUT_FILE_TOP10 % feature

    for f in [OUTPUT_FILE_TOP1, OUTPUT_FILE_TOP5, OUTPUT_FILE_TOP10]:
        if os.path.exists(f):
            os.remove(f)

    logging.info("loading CCA model")
    # load CCA model
    with open(CCA_FILE % feature, 'rb') as f:
        Q_k, A_k = pkl.load(f)

    logging.info("calculating distance")
    data = ReVerbPairs(usage='test', mode='index')
    feats = data2feats(data, feature)

    with Pool(processes=30) as pool:
        result = pool.map(partial(distance, Q_k=Q_k, A_k=A_k), enumerate(feats))
    # sort by index, make sure the result in the same order with the file
    result = sorted(result, key=lambda x: x[0])
    del data, feats, Q_k, A_k

    logging.info("combining with text file")
    line_num = 0
    output_tuple = defaultdict(tuple)
    for line in open('./data/labels.txt', 'r'):
        _, q, a = line.strip().split('\t')
        pred = result[line_num]
        output_tuple[q].append(pred, a)
        line_num += 1

    f1 = open(OUTPUT_FILE_TOP1, 'a')
    f5 = open(OUTPUT_FILE_TOP5, 'a')
    f10 = open(OUTPUT_FILE_TOP10, 'a')

    for q in output_tuple.keys():
        # sort by distance, accent
        tmp_set = sorted(output_tuple[q], key=lambda x: x[0])
        if len(tmp_set) == 0: continue

        # only keep the best one
        for pred, a in tmp_set[0]:
            output_line = "{}\t{}\t{}".format(q, pred, a)
        # keep top 5
        for pred, a in tmp_set[:5]:
            output_line = "{}\t{}\t{}".format(q, pred, a)
        # keep top 10
        for pred, a in tmp_set[:10]:
            output_line = "{}\t{}\t{}".format(q, pred, a)

    f1.close()
    f5.close()
    f10.close()
