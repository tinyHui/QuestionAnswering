from collections import defaultdict
from sys import stdout
from multiprocessing import Process, Manager,Queue
from queue import Empty
from preprocess.feats import FEATURE_OPTS, feats_loader
from CCA import distance
import argparse
import pickle as pkl
import numpy as np
import logging
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
OUTPUT_FILE_TOP1 = './result/reverb-test-with_dist.top1.%s.txt'
OUTPUT_FILE_TOP5 = './result/reverb-test-with_dist.top5.%s.txt'
OUTPUT_FILE_TOP10 = './result/reverb-test-with_dist.top10.%s.txt'
PROCESS_NUM = 20


def loader(feat, results, length, Q_k=None, A_k=None, use_paraphrase_map=False, Q1_k=None, Q2_k=None):
    while True:
        try:
            indx, (d, f) = feat
            feat = f(d)
            _, crt_q_v, crt_a_v, _ = feat
            if use_paraphrase_map:
                crt_q_v1 = crt_q_v.dot(Q1_k)
                crt_q_v2 = crt_q_v.dot(Q2_k)
                crt_q_v = np.hstack((crt_q_v1, crt_q_v2))
            if Q_k is not None and A_k is not None:
                proj_q = crt_q_v.dot(Q_k)
                proj_a = crt_a_v.dot(A_k)
                dist = distance(proj_q, proj_a)
            else:
                dist = distance(crt_q_v, crt_a_v)
            results[indx] = dist
        except Empty:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Define test process.')
    parser.add_argument('--CCA_stage', type=int,
                        help='Define CCA stage number')
    parser.add_argument('--para_map_file', type=str,
                        help='Define location for CCA model trained by paraphrase question')
    parser.add_argument('--feature', nargs=2, default=[],
                        help="Take 2 args, feature and model file. Feature option: %s" % (", ".join(FEATURE_OPTS)))
    # parser.add_argument('--full_rank', action='store_true', default=False,
    #                     help='Use full rank for selecting answer')
    # parser.add_argument('--rerank', action='store_true', default=False,
    #                     help='Use rerank for selecting answer')
    parser.add_argument('--worker', type=int, default=PROCESS_NUM, help='Process number')

    args = parser.parse_args()
    feature = args.feature[0]
    qa_model_file = args.feature[1]

    # 1/2 stage CCA
    cca_stage = args.CCA_stage
    assert cca_stage in [0, 1, 2], "can only use 1 stage CCA or 2 stage CCA or no CCA"
    use_paraphrase_map = False
    if cca_stage == 2:
        use_paraphrase_map = True
        para_map_file = args.para_map_file
        assert feature in ['avg', 'holographic'], "%s is not supported by 2 stage CCA"

    # assert args.full_rank ^ args.rerank, 'must specify full rank or rerank'
    # full_rank = args.full_rank
    PROCESS_NUM = args.worker

    OUTPUT_FILE_TOP1 = OUTPUT_FILE_TOP1 % feature
    OUTPUT_FILE_TOP5 = OUTPUT_FILE_TOP5 % feature
    OUTPUT_FILE_TOP10 = OUTPUT_FILE_TOP10 % feature

    for f in [OUTPUT_FILE_TOP1, OUTPUT_FILE_TOP5, OUTPUT_FILE_TOP10]:
        if os.path.exists(f):
            os.remove(f)

    logging.info("using feature: %s" % feature)

    Q_k = None
    A_k = None
    if cca_stage > 0:
        logging.info("loading CCA model")
        # load CCA model
        with open(qa_model_file, 'rb') as f:
            Q_k, A_k = pkl.load(f)

    Q1_k = None
    Q2_k = None
    if cca_stage == 2:
        with open(para_map_file, 'rb') as f:
            Q1_k, Q2_k = pkl.load(f)

    logging.info("calculating distance")
    _, _, feats = feats_loader(feature, usage='test')
    length = len(feats)

    # calculate the distance
    result = {}
    for feat in enumerate(feats):
        loader(feat, result, length, Q_k, A_k, use_paraphrase_map, Q1_k, Q2_k)

    # sort by index, low to high
    stdout.write("\n")
    logging.info("sort result in order")
    result = []
    for i in range(len(feats)):
        result.append(result_list_share[i])

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
