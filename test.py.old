from word2index import UNIGRAM_DICT_FILE
from preprocess.data import ReVerbPairs
from train import CCA_FILE
from preprocess.feats import FEATURE_OPTS, data2feats
from collections import defaultdict, UserList
from CCA import find_answer
import argparse
import pickle as pkl
import logging
import numpy as np

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
INF_FREQ = 1000


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

    logging.info("loading CCA model")
    # load CCA model
    with open(CCA_FILE % feature, 'rb') as f:
        Q_k, A_k = pkl.load(f)

    logging.info("constructing testing data")
    data = ReVerbPairs(usage='test', mode='index')
    feats = data2feats(data, feature)
    length = len(feats)

    crt_q_indx = -1
    answer_indx = 0
    crt_q = None
    prev_q = None
    q_a_map_list = defaultdict(list)      # all potential answers related to the question
    q_a_map_correct = defaultdict(list)        # the correct answer for this question
    q_cluster_map = {}                      # map question index to cluster id
    Qs = UserList()
    As = UserList()
    for t in feats:
        indx = answer_indx + 1
        if indx % INF_FREQ == 0 or indx == length:
            logging.info("loading: %d/%d" % (indx, length))

        (crt_q, crt_a, crt_q_cluster_id, label), (crt_q_v, crt_a_v, _, _) = t

        # question are sorted by alphabet
        # no need to add repeat questions
        if prev_q != crt_q:
            crt_q_indx += 1
            Qs.append(crt_q_v)
            q_cluster_map[crt_q_indx] = crt_q_cluster_id

        if label == 1:
            # current answer is the correct answer
            q_a_map_correct[crt_q_cluster_id].append(answer_indx)

        q_a_map_list[crt_q_cluster_id].append(answer_indx)
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
    for crt_q_indx, q in enumerate(proj_Qs):
        # get cluster id for this question
        crt_q_cluster_id = q_cluster_map[crt_q_indx]
        # answer index is stored in accent order
        answer_indx_list = q_a_map_list[crt_q_cluster_id]
        if len(q_a_map_correct[crt_q_cluster_id]) == 0:
            no_correct += 1
            continue

        if len(answer_indx_list) == 1:
            # only have one candidate answer
            pred = q_a_map_correct[crt_q_cluster_id][0]
            one_candidate += 1
        else:
            # more than one candidate answers
            pred = find_answer(q, proj_As[answer_indx_list[0]:answer_indx_list[-1]])
        # add the offset
        pred += answer_indx_list[0]
        # if the found answer is one of the potential answer of the question
        if pred in q_a_map_correct[crt_q_cluster_id]:
            # correct
            correct_num += 1
        if (crt_q_indx + 1) % 5 == 0 or crt_q_indx + 1 == q_num:
            logging.info("tested: %d/%d, get %d correct"
                         % (crt_q_indx + 1, q_num, correct_num))

    # output result
    accuracy = float(correct_num) / q_num
    accuracy_fix = float(correct_num - one_candidate) / (q_num - one_candidate - no_correct)
    print("%d questions have only one answer" % one_candidate)
    print("%d questions don't have correct answer" % no_correct)
    print("%d questions have more than one correct answer" % more_correct)
    print("The model get %d/%d correct, precision: %f, fixed precision: %f"
          % (correct_num, q_num, accuracy, accuracy_fix))
