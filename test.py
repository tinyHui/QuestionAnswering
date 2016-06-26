from word2index import VOC_DICT_FILE
from preprocess.data import ReVerbPairs
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
        Q_k, A_k = pkl.load(f)

    logging.info("constructing testing data")
    data = ReVerbPairs(usage='test', mode='index', voc_dict=voc_dict)
    feats = data2feats(data, feature)
    length = len(feats)

    question_indx = -1      # question index will always add one in the beginning
    answer_indx = 0
    prev_q = None
    crt_q = None
    q_a_map_list = defaultdict(list)      # all potential answers related to the question
    q_a_map_correct = defaultdict(list)        # the correct answer for this question
    Qs = []
    As = []
    for t in feats:
        indx = answer_indx + 1
        if indx % INF_FREQ == 0 or indx == length:
            logging.info("loading: %d/%d" % (indx, length))

        (crt_q, crt_a, label), (crt_q_v, crt_a_v, _) = t

        # question are sorted by alphabet
        # no need to add repeat questions
        if prev_q != crt_q:
            question_indx += 1
            Qs.append(crt_q_v)

        if label == 1:
            # current answer is the correct answer
            q_a_map_correct[question_indx].append(answer_indx)

        q_a_map_list[question_indx].append(answer_indx)
        # bind answer with its index
        As.append(crt_a_v)

        prev_q = crt_q
        answer_indx += 1

    q_num = len(Qs)
    a_num = len(As)
    logging.info("found %d questions, %d answers" % (q_num, a_num))

    logging.info("generating project vector using trained CCA model")
    proj_Qs = np.tensordot(Qs, Q_k, axes=1)
    proj_As = np.tensordot(As, A_k, axes=1)
    del Qs, As

    logging.info("testing")
    correct_num = 0
    one_candidate = 0
    no_correct = 0
    more_correct = 0
    for question_indx, q in enumerate(proj_Qs):
        # answer index is stored in accent order
        answer_indx_list = q_a_map_list[question_indx]
        if len(q_a_map_correct[question_indx]) == 0:
            no_correct += 1
            continue
        if len(q_a_map_correct[question_indx]) > 1:
            more_correct += 1
        # only have one candidate answer
        elif len(answer_indx_list) == 1:
            pred = find_answer(q, [proj_As[answer_indx_list[0]]])
            one_candidate += 1
        else:
            pred = find_answer(q, proj_As[answer_indx_list[0]:answer_indx_list[-1]])
        # add the offset
        pred += answer_indx_list[0]
        # if the found answer is one of the potential answer of the question
        if pred in q_a_map_correct[question_indx]:
            # correct
            correct_num += 1
        if question_indx % 5 == 0 or question_indx + 1 == q_num:
            logging.info("tested: %d/%d, get %d correct"
                         % (question_indx + 1, q_num, correct_num))

    # output result
    accuracy = float(correct_num) / q_num
    accuracy_fix = float(correct_num - one_candidate) / (q_num - one_candidate - no_correct)
    print("%d questions have only one answer" % one_candidate)
    print("%d questions don't have correct answer" % no_correct)
    print("%d questions have more than one correct answer" % more_correct)
    print("The model get %d/%d correct, precision: %f, fixed precision: %f"
          % (correct_num, q_num, accuracy, accuracy_fix))
