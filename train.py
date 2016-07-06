from preprocess.data import ReVerbPairs
from preprocess.feats import FEATURE_OPTS, data2feats
from scipy.sparse import csr_matrix
from scipy.sparse import vstack as sparse_vstack
import numpy as np
from multiprocessing import Queue, Process
from CCA import train
import argparse
import pickle as pkl
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
QA_PAIR_FILE = "./bin/QsAs.%s.pkl"
CCA_FILE = "./bin/CCA_model_%s.pkl"
INF_FREQ = 1000  # information message frequency
PROCESS_NUM = 30


def generate_part_dense(feature_set, qa_queue, count_queue):
    last_i = 0
    i = 1
    Qs = None
    As = None
    for feature in feature_set:
        _, feat = feature

        if isinstance(Qs, type(None)):
            Qs = feat[0]
            As = feat[1]
        else:
            Qs = np.vstack((Qs, feat[0]))
            As = np.vstack((As, feat[1]))

        if i % INF_FREQ == 0 or i == len(feature_set):
            qa_queue.put((Qs, As))
            count_queue.put(i - last_i)
            # reset Qs and As
            Qs = None
            As = None
            last_i = i
        i += 1


def generate_part_sparse(feature_set, qa_queue, count_queue):
    last_i = 0
    i = 1
    Qs = None
    As = None
    for feature in feature_set:
        _, feat = feature

        if isinstance(Qs, type(None)):
            Qs = csr_matrix(feat[0], dtype='float64')
            As = csr_matrix(feat[1], dtype='float64')
        else:
            Qs = sparse_vstack((Qs, feat[0]))
            As = sparse_vstack((As, feat[1]))

        if i % INF_FREQ == 0 or i == len(feature_set):
            qa_queue.put((Qs, As))
            count_queue.put(i - last_i)
            # reset Qs and As
            Qs = None
            As = None
            last_i = i
        i += 1


def generate_dense(qa_queue, count_queue):
    Qs = None
    As = None
    count = 0
    while True:
        if count == length:
            break
        # pending until have result
        QA_temp = qa_queue.get()
        new_line_num = count_queue.get()
        Qs_temp, As_temp = QA_temp
        if isinstance(Qs, type(None)):
            Qs = np.array(Qs_temp, dtype='float64')
            As = np.array(As_temp, dtype='float64')
        else:
            Qs = np.vstack((Qs, Qs_temp))
            As = np.vstack((As, As_temp))

        count += new_line_num
        logging.info("loading: %d/%d" % (count, length))

    return Qs, As


def generate_sparse(qa_queue, count_queue):
    Qs = None
    As = None
    count = 0
    while True:
        if count == length:
            break
        # pending until have result
        QA_temp = qa_queue.get()
        new_line_num = count_queue.get()
        Qs_temp, As_temp = QA_temp
        if isinstance(Qs, type(None)):
            Qs = csr_matrix(Qs_temp, dtype='float64')
            As = csr_matrix(As_temp, dtype='float64')
        else:
            Qs = sparse_vstack((Qs, Qs_temp))
            As = sparse_vstack((As, As_temp))

        count += new_line_num
        logging.info("loading: %d/%d" % (count, length))

    return Qs, As


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Define training process.')
    parser.add_argument('--feature', type=str, default='bow', help="Feature option: %s" % (", ".join(FEATURE_OPTS)))
    parser.add_argument('--freq', type=int, default=INF_FREQ, help='Information print out frequency')
    parser.add_argument('--sparse', action='store_true', default=False,
                        help='Use sparse matrix for C_AA, C_AB and C_BB')
    parser.add_argument('--svds', type=int, default=-1, help='Define k value for svds, otherwise use full svd')
    parser.add_argument('--load_QA', type=str, default=None, help='Use pre-constructed QA pairs')

    args = parser.parse_args()
    feature = args.feature
    INF_FREQ = args.freq
    k = args.svds
    sparse = args.sparse
    full_svd = k == -1
    qa_pair_file = args.load_QA

    if qa_pair_file:
        logging.info("constructing train data")
        data_list = [ReVerbPairs(usage='train', part=i, mode='index') for i in range(PROCESS_NUM)]
        feats_list = [data2feats(data, feature) for data in data_list]
        pair_num = sum([len(data) for data in data_list])
        length = sum([len(feats) for feats in feats_list])

        if sparse:
            generate_part = generate_part_sparse
            generate = generate_sparse
        else:
            generate_part = generate_part_dense
            generate = generate_dense

        qa_queue = Queue()
        count_queue = Queue()
        for i in range(PROCESS_NUM):
            p = Process(target=generate_part, args=(feats_list[i], qa_queue, count_queue))
            p.start()

        Qs, As = generate(qa_queue, count_queue)
        with open(QA_PAIR_FILE % feature, 'wb') as f:
            pkl.dump((Qs, As), f, protocol=4)

    else:
        logging.info("loading pre-constructed data")
        assert qa_pair_file is not None
        with open(qa_pair_file, 'rb') as f:
            Qs, As = pkl.load(f)
            pair_num = Qs.shape[0]

    if sparse:
        logging.info("using sparse matrix")
    else:
        logging.info("using dense matrix")

    if full_svd:
        logging.info("running CCA, using full SVD")
    else:
        logging.info("running CCA, using SVDs, k=%d" % k)

    Q_k, A_k = train(Qs, As, sample_num=pair_num, full_svd=full_svd, k=k, sparse=sparse)

    logging.info("dumping model into binary file")
    # dump to disk for reuse
    with open(CCA_FILE % feature, 'wb') as f:
        pkl.dump((Q_k, A_k), f, protocol=4)

