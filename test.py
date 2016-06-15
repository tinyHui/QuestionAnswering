from text2index import VOC_DICT_FILE
from preprocess.data import QAs
from train import CCA_FILE
from preprocess.feats import FEATURE_OPTS, data2feats
from collections import defaultdict
from CCA import find_answer
import argparse
import pickle as pkl
import logging
import numpy as np


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
INF_FREQ = 300


def projection(t, P, no_indx, q):
    i, v = t
    v_proj = np.dot(v, P)
    if no_indx:
        q.append(v_proj)
    else:
        q.append(i, v_proj)


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

    question_indx = 0
    answer_indx = 0
    prev_q = None
    crt_q = None
    q_a_map = defaultdict(list)
    Qs = []
    As = []
    for t in feats:
        indx = answer_indx + 1
        if indx % INF_FREQ == 0 or indx == length:
            logging.info("loading: %d/%d" % (indx, length))

        (crt_q, crt_a), (crt_q_v, crt_a_v) = t

        # question are sorted by alphabet
        # no need to add repeat questions
        if crt_q != prev_q:
            question_indx += 1
            Qs.append(crt_q_v)

        # bind answer with its index
        As.append(crt_a_v)
        # current answer is one of the correct answer
        q_a_map[question_indx].append(answer_indx)

        prev_q = crt_q
        answer_indx += 1

    q_num = len(Qs)
    a_num = len(As)
    logging.info("found %d questions, %d answers" % (q_num, a_num))

    logging.info("generating project vector using trained CCA model")
    Qs_proj = np.tensordot(Qs, U, axes=1)
    As_proj = np.tensordot(As, V.T, axes=1)
    del Qs, As

    logging.info("testing")
    correct_num = 0
    for question_indx, q in enumerate(Qs_proj):
        pred = find_answer(q, As_proj)
        # if the found answer is one of the potential answer of the question
        if pred in q_a_map[question_indx]:
            # correct
            correct_num += 1
        if indx % 5 == 0 or indx == length:
            logging.info("tested: %d/%d, get %d correct" % (question_indx + 1, q_num, correct_num))

    # output result
    accuracy = float(correct_num) / len(data)
    print("The model get %d/%d correct, precision: %f" % (correct_num, len(data), accuracy))



