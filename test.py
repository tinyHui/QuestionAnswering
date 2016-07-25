from collections import defaultdict
from collections import UserList
from sys import stdout
from multiprocessing import Process, Manager,Queue
from queue import Empty
from itertools import islice
from preprocess.feats import FEATURE_OPTS, feats_loader
from CCA import distance
from functools import partial
import argparse
import pickle as pkl
import logging
import os
from time import sleep

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
OUTPUT_FILE_TOP1 = './result/reverb-test-with_dist.top1.%s.txt'
OUTPUT_FILE_TOP5 = './result/reverb-test-with_dist.top5.%s.txt'
OUTPUT_FILE_TOP10 = './result/reverb-test-with_dist.top10.%s.txt'
PROCESS_NUM = 20


def loader(feats_queue, Q_k, A_k, results, length):
    while True:
        try:
            indx, (d, f) = feats_queue.get(timeout=5)
            stdout.write("\rTesting: %d/%d" % (indx+1, length))
            stdout.flush()
            feat = f(d)
            _, crt_q_v, crt_a_v, _ = feat
            proj_q = crt_q_v.dot(Q_k)
            proj_a = crt_a_v.dot(A_k)
            dist = distance(proj_q, proj_a)
            results[indx] = dist
        except Empty:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Define test process.')
    parser.add_argument('--feature', nargs=2, default=[],
                        help="Take 2 args, feature and model file. Feature option: %s" % (", ".join(FEATURE_OPTS)))
    # parser.add_argument('--full_rank', action='store_true', default=False,
    #                     help='Use full rank for selecting answer')
    # parser.add_argument('--rerank', action='store_true', default=False,
    #                     help='Use rerank for selecting answer')
    parser.add_argument('--worker', type=int, default=PROCESS_NUM, help='Process number')

    args = parser.parse_args()
    feature = args.feature[0]
    model_file = args.feature[1]
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
    logging.info("loading CCA model")
    # load CCA model
    with open(model_file, 'rb') as f:
        Q_k, A_k = pkl.load(f)

    logging.info("calculating distance")
    feats = feats_loader(feature, usage='test')
    length = len(feats)

    # multiprocess to calculate the distance
    manager = Manager()
    feats_queue = Queue(maxsize=length)
    result_list_share = manager.dict()

    p_list = [Process(target=loader, args=(feats_queue, Q_k, A_k, result_list_share, length))
              for _ in range(PROCESS_NUM)]

    for p in p_list:
        p.daemon = True
        p.start()

    for i, feat in enumerate(feats):
        feats_queue.put((i, feat))

    for p in p_list:
        p.join()

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
