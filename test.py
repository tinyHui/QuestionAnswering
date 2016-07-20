from collections import UserList
from collections import defaultdict
from train import CCA_FILE
from preprocess.feats import FEATURE_OPTS, feats_loader
from CCA import distance
from sys import stdout
import argparse
import pickle as pkl
import logging
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
OUTPUT_FILE_TOP1 = './data/reverb-test-with_dist.top1.%s.txt'
OUTPUT_FILE_TOP5 = './data/reverb-test-with_dist.top5.%s.txt'
OUTPUT_FILE_TOP10 = './data/reverb-test-with_dist.top10.%s.txt'


def loader(feat, Q_k, A_k):
    _, (_, crt_q_v, crt_a_v, _) = feat
    proj_q = crt_q_v.dot(Q_k)
    proj_a = crt_a_v.dot(A_k)
    return distance(proj_q, proj_a)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Define training process.')
    parser.add_argument('--feature', type=str, default='unigram', help="Feature option: %s" % (", ".join(FEATURE_OPTS)))
    # parser.add_argument('--full_rank', action='store_true', default=False,
    #                     help='Use full rank for selecting answer')
    # parser.add_argument('--rerank', action='store_true', default=False,
    #                     help='Use rerank for selecting answer')

    args = parser.parse_args()
    feature = args.feature
    # assert args.full_rank ^ args.rerank, 'must specify full rank or rerank'
    # full_rank = args.full_rank

    OUTPUT_FILE_TOP1 = OUTPUT_FILE_TOP1 % feature
    OUTPUT_FILE_TOP5 = OUTPUT_FILE_TOP5 % feature
    OUTPUT_FILE_TOP10 = OUTPUT_FILE_TOP10 % feature

    for f in [OUTPUT_FILE_TOP1, OUTPUT_FILE_TOP5, OUTPUT_FILE_TOP10]:
        if os.path.exists(f):
            os.remove(f)

    logging.info("using feature: %s" % feature)
    logging.info("loading CCA model")
    # load CCA model
    with open(CCA_FILE % feature, 'rb') as f:
        Q_k, A_k = pkl.load(f)

    logging.info("calculating distance")
    feats = feats_loader(feature, usage='test')

    result = UserList()
    length = len(feats)
    for i, feat in enumerate(feats):
        stdout.write("\rTesting: %d/%d" % (i+1, length))
        stdout.flush()
        result.append(loader(feat, Q_k=Q_k, A_k=A_k))
    stdout.write("\n")

    logging.info("combining with text file")
    line_num = 0
    output_tuple = defaultdict(list)
    for line in open('./data/labels.txt', 'r'):
        _, q, a = line.strip().split('\t')
        pred = result[line_num]
        output_tuple[q].append((pred, a))
        line_num += 1

    f1 = open(OUTPUT_FILE_TOP1, 'a')
    f5 = open(OUTPUT_FILE_TOP5, 'a')
    f10 = open(OUTPUT_FILE_TOP10, 'a')

    for q in output_tuple.keys():
        # sort by distance, accent
        tmp_set = sorted(output_tuple[q], key=lambda x: x[0])
        if len(tmp_set) == 0: continue

        # only keep the best one
        pred, a = tmp_set[0]
        output_line = "{}\t{}\t{}\n".format(q, pred, a)
        f1.write(output_line)
        # keep top 5
        for pred, a in tmp_set[:5]:
            output_line = "{}\t{}\t{}\n".format(q, pred, a)
            f5.write(output_line)
        # keep top 10
        for pred, a in tmp_set[:10]:
            output_line = "{}\t{}\t{}\n".format(q, pred, a)
            f10.write(output_line)

    f1.close()
    f5.close()
    f10.close()
