from preprocess.feats import FEATURE_OPTS, feats_loader
from scipy.sparse import csr_matrix
from scipy.sparse import vstack as sparse_vstack
import numpy as np
from multiprocessing import Queue, Process
from queue import Empty
from time import sleep
import argparse
import logging
import pickle as pkl

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
Q_MATRIX = "./bin/Q.%s.pkl"
A_MATRIX = "./bin/A.%s.pkl"

PARA_1_MATRIX = "./bin/Para1.%s.pkl"
PARA_2_MATRIX = "./bin/Para2.%s.pkl"

INF_FREQ = 2000  # information message frequency
PROCESS_NUM = 15
MAX_HOLD = INF_FREQ / 4
MAX_PART_H = 500000


def produce_feats(feats, feats_queue):
    logging.info("Start producer")
    for feat in feats:
        feats_queue.put(feat)
        while feats_queue.qsize() > 100000:
            sleep(0.01)
    logging.info("Stop producer")


def generate_part_dense(feats_queue, qa_queue, q_indx, a_indx):
    logging.info("Start part consumer")
    Qs = []
    As = []
    hold = 0
    while True:
        try:
            feat = feats_queue.get(timeout=5)

            Qs.append(feat[q_indx])
            As.append(feat[a_indx])
            hold += 1
            if hold == MAX_HOLD:
                qa_queue.put((np.asarray(Qs, dtype='float64'), np.asarray(As, dtype='float64')))
                hold = 0
                Qs = []
                As = []
        except Empty:
            if Qs:
                qa_queue.put((np.asarray(Qs, dtype='float64'), np.asarray(As, dtype='float64')))
            break
    logging.info("Stop part consumer")


def generate_part_sparse(feats_queue, qa_queue, q_indx, a_indx):
    logging.info("Start part consumer")
    Qs = []
    As = []
    hold = 0
    while True:
        try:
            feat = feats_queue.get(timeout=5)

            Qs.append(feat[q_indx])
            As.append(feat[a_indx])
            hold += 1
            if hold == MAX_HOLD:
                qa_queue.put((csr_matrix(Qs, dtype='float64'), csr_matrix(As, dtype='float64')))
                hold = 0
                Qs = []
                As = []
        except Empty:
            if Qs:
                qa_queue.put((csr_matrix(Qs, dtype='float64'), csr_matrix(As, dtype='float64')))
            break
    logging.info("Stop part consumer")


def generate_dense(qa_queue, length, l_matrix_fname, r_matrix_fname):
    logging.info("Start consumer")
    Qs = []
    As = []
    part_file_count = 0
    part_count = 0
    overall_count = 0
    while True:
        try:
            Qs_temp, As_temp = qa_queue.get(timeout=120)
            part_count += Qs_temp.shape[0]
            overall_count += Qs_temp.shape[0]
            Qs.append(Qs_temp)
            As.append(As_temp)
            if overall_count == length:
                raise Empty
            if overall_count % INF_FREQ == 0:
                logging.info("loading: %d/%d, %.2f%%" % (overall_count, length, overall_count / length * 100))

            if part_count >= MAX_PART_H:
                part_Qs = np.vstack(Qs[:MAX_PART_H])
                Qs = Qs[MAX_PART_H:]
                with open("{}.part{}".format(l_matrix_fname, part_file_count), 'wb') as f:
                    pkl.dump(part_Qs, f, protocol=4)

                part_As = np.vstack(As[:MAX_PART_H])
                As = As[MAX_PART_H:]
                with open("{}.part{}".format(r_matrix_fname, part_file_count), 'wb') as f:
                    pkl.dump(part_As, f, protocol=4)

                part_count -= MAX_PART_H
                part_file_count += 1

        except Empty:
            logging.info("loading: %d/%d, %.2f%%" % (overall_count, length, overall_count / length * 100))
            with open("{}.part{}".format(l_matrix_fname, part_file_count), 'wb') as f:
                pkl.dump(np.vstack(Qs), f, protocol=4)
            with open("{}.part{}".format(r_matrix_fname, part_file_count), 'wb') as f:
                pkl.dump(np.vstack(As), f, protocol=4)
            break

    logging.info("Stop consumer")
    logging.info("saving the list version of features")


def generate_sparse(qa_queue, length, l_matrix_fname, r_matrix_fname):
    logging.info("Start consumer")
    Qs = None
    As = None
    count = 0
    while True:
        try:
            Qs_temp, As_temp = qa_queue.get(timeout=120)
            count += Qs_temp.shape[0]
            if Qs is None:
                Qs = Qs_temp
                As = As_temp
            else:
                Qs = sparse_vstack((Qs, Qs_temp))
                As = sparse_vstack((As, As_temp))
            if count == length:
                raise Empty
            if count % INF_FREQ == 0:
                logging.info("loading: %d/%d, %.2f%%" % (count, length, count / length * 100))
        except Empty:
            logging.info("loading: %d/%d, %.2f%%" % (count, length, count / length * 100))
            break

    logging.info("Stop consumer")
    with open(l_matrix_fname, 'wb') as f:
        pkl.dump(Qs, f, protocol=4)
    with open(r_matrix_fname, 'wb') as f:
        pkl.dump(As, f, protocol=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Define training process.')
    parser.add_argument('--feature', type=str, default='unigram', help="Feature option: %s" % (", ".join(FEATURE_OPTS)))
    parser.add_argument('--freq', type=int, default=INF_FREQ, help='Information print out frequency')
    parser.add_argument('--sparse', action='store_true', default=False,
                        help='Use sparse matrix for C_AA, C_AB and C_BB')
    parser.add_argument('--CCA_stage', type=int, default=1,
                        help='Define CCA stage number, set as -1 for train paraphrase CCA')
    parser.add_argument('--worker', type=int, default=PROCESS_NUM, help='Process number')

    args = parser.parse_args()
    feature = args.feature
    INF_FREQ = args.freq
    sparse = args.sparse
    PROCESS_NUM = args.worker
    MAX_HOLD = INF_FREQ // 4

    mid_fix = feature

    # 1/2 stage CCA
    cca_stage = args.CCA_stage
    assert cca_stage in [1, 2, -1], "can only use 1 stage CCA or 2 stage CCA"
    use_paraphrase_map = False
    if cca_stage == -1:
        train_two_stage_cca = True
        # dump file name
        l_matrix_fname = PARA_1_MATRIX % mid_fix
        r_matrix_fname = PARA_2_MATRIX % mid_fix
    else:
        train_two_stage_cca = False
        # dump file name
        l_matrix_fname = Q_MATRIX % mid_fix
        r_matrix_fname = A_MATRIX % mid_fix

    logging.info("constructing train data")
    if sparse:
        generate_part = generate_part_sparse
        generate = generate_sparse
    else:
        generate_part = generate_part_dense
        generate = generate_dense

    feats_queue = Queue()
    qa_queue = Queue()
    q_indx, a_indx, feats = feats_loader(feature, usage='train', train_two_stage_cca=train_two_stage_cca)
    length = len(feats)

    p_producer = Process(target=produce_feats, args=(feats, feats_queue))
    p_producer.daemon = True
    p_producer.start()

    p_list = [Process(target=generate_part, args=(feats_queue, qa_queue, q_indx, a_indx))
              for _ in range(PROCESS_NUM)]
    for p in p_list:
        p.daemon = True
        p.start()

    generate(qa_queue, length, l_matrix_fname, r_matrix_fname)

    for p in p_list:
        p.join()

    logging.info("Done")
