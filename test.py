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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Define training process.')
    parser.add_argument('--feature', type=str, default='bow', help="Feature option: %s" % (", ".join(FEATURE_OPTS)))
    parser.add_argument('--freq', type=int, default=300, help='Information print out frequency')

    args = parser.parse_args()
    feature = args.feature
    INF_FREQ = args.freq

    with open(VOC_DICT_FILE, 'rb') as f:
        voc_dict = pkl.load(f)

    data = QAs(usage='test', mode='index', voc_dict=voc_dict)
    feats = None
    Qs = []
    As = []

    feats = data2feats(data, feature)

    logging.info("loading CCA model")
    # load CCA model
    with open(CCA_FILE % feature, 'rb') as f:
        U, V = pkl.load(f)

    logging.info("constructing testing data")
    length = len(feats)
    question_indx = 1
    answer_indx = 1
    prev_q = None
    crt_q = None
    q_a_map = defaultdict(list)
    for feat in feats:
        if answer_indx % INF_FREQ == 0 or answer_indx == length:
            logging.info("loading: %d/%d" % (answer_indx, length))
        # project question use CCA
        crt_q = np.dot(feat[0], U)
        # question are sorted by alphabet
        # no need to add repeat questions
        if crt_q != prev_q:
            question_indx += 1
            Qs.append(crt_q)

        prev_q = crt_q
        # project answer use CCA
        crt_a = np.dot(feat[1], V.T)
        # bind answer with its index
        As.append((answer_indx, crt_a))
        # current answer is one of the correct answer
        q_a_map[question_indx].append(answer_indx)

        answer_indx += 1

    logging.info("testing")
    correct_num = 0
    for answer_indx, q in enumerate(Qs):
        pred = find_answer(q, As)
        # if the found answer is one of the potential answer of the question
        if pred in q_a_map[q]:
            # correct
            correct_num += 1
        logging.warning("tested: %d/%d, get %d correct" % (answer_indx + 1, length, correct_num))

    # output result
    accuracy = float(correct_num) / len(QAs)
    print("The model get %d/%d correct, precision: %f" % (correct_num, len(QAs), accuracy))



