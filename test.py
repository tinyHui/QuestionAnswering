from text2index import VOC_DICT_FILE
from preprocess.data import QAs
from train import CCA_FILE
from preprocess.feats import FEATURE_OPTS, data2feats
from collections import defaultdict
from CCA import find_answer
from multiprocessing import Pool, Manager
from functools import partial
import argparse
import pickle as pkl
import logging
import numpy as np


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
INF_FREQ = 10


def projection(t, P, no_indx):
    i, v = t
    v_proj = np.dot(v, P)
    if no_indx:
        return v_proj
    else:
        return i, v_proj


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Define training process.')
    parser.add_argument('--feature', type=str, default='bow', help="Feature option: %s" % (", ".join(FEATURE_OPTS)))
    parser.add_argument('--freq', type=int, default=INF_FREQ, help='Information print out frequency')

    args = parser.parse_args()
    feature = args.feature
    INF_FREQ = args.freq

    logging.info("loading vocabulary index")
    with open(VOC_DICT_FILE, 'rb') as f:
        voc_dict = pkl.load(f)

    logging.info("loading CCA model")
    # load CCA model
    with open(CCA_FILE % feature, 'rb') as f:
        U, V = pkl.load(f)

    logging.info("constructing testing data")
    data = QAs(usage='test', mode='index', voc_dict=voc_dict)
    feats = data2feats(data, feature)
    length = len(feats)

    question_indx = 1
    answer_indx = 1
    prev_q = None
    crt_q = None
    q_a_map = defaultdict(list)
    Qs = []
    As = []
    for t in feats:
        if answer_indx % INF_FREQ == 0 or answer_indx == length:
            logging.info("loading: %d/%d" % (answer_indx, length))

        (crt_q, crt_a), (crt_q_v, crt_a_v) = t

        # question are sorted by alphabet
        # no need to add repeat questions
        if crt_q != prev_q:
            question_indx += 1
            Qs.append((question_indx, crt_q_v))

        # bind answer with its index
        As.append((answer_indx, crt_a_v))
        # current answer is one of the correct answer
        q_a_map[question_indx].append(answer_indx)

        prev_q = crt_q
        answer_indx += 1

    logging.info("generating project vector using trained CCA model")
    with Pool(processes=8) as p:
        Qs = p.map(partial(projection, P=U, no_indx=False), Qs)
        As = p.map(partial(projection, P=V.T, no_indx=True), As)

    logging.info("testing")
    correct_num = 0
    indx = 1
    for question_indx, q in Qs:
        pred = find_answer(q, As)
        # if the found answer is one of the potential answer of the question
        if pred in q_a_map[question_indx]:
            # correct
            correct_num += 1
        logging.warning("tested: %d/%d, get %d correct" % (indx, length, correct_num))
        indx += 1

    # output result
    accuracy = float(correct_num) / len(QAs)
    print("The model get %d/%d correct, precision: %f" % (correct_num, len(QAs), accuracy))



