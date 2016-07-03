from multiprocessing import Pool
from functools import partial
from preprocess.data import ReVerbPairs
from train import CCA_FILE
from preprocess.feats import FEATURE_OPTS, data2feats
from CCA import distance
import argparse
import pickle as pkl
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
INF_FREQ = 1000
OUTPUT_FILE = './data/reverb-test-with_dist.txt'


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

    logging.info("loading CCA model")
    # load CCA model
    with open(CCA_FILE % feature, 'rb') as f:
        Q_k, A_k = pkl.load(f)

    logging.info("calculating distance")
    data = ReVerbPairs(usage='test', mode='index')
    feats = data2feats(data, feature)

    with Pool(processes=8) as pool:
        result = pool.map(partial(distance, Q_k=Q_k, A_k=A_k), enumerate(feats))
    # sort by index
    result = sorted(result, key=lambda x:x[0])

    with open(OUTPUT_FILE, 'a') as f:
        line_num = 0
        for line in open('./data/labels.txt', 'r'):
            _, q, a = line.strip().split('\t')
            pred = result[line_num]
            output_line = "{}\t{}\t{}".format(q, pred, a)
            f.write(output_line)
            line_num += 1

