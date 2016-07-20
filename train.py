from preprocess.feats import FEATURE_OPTS, feats_loader
from scipy.sparse import csr_matrix
import numpy as np
from multiprocessing import Queue, Process
from queue import Empty
from CCA import xcov, decompose
import argparse
import pickle as pkl
import logging
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
QA_PAIR_FILE = "./bin/Raw.%s.pkl"
XCOV_FILE = "./bin/XCOV.%s.pkl"
CCA_FILE = "./bin/CCA.%s.%s.pkl"
PARA_MAP_FILE = "./bin/ParaMap.pkl"
INF_FREQ = 5000  # information message frequency
PROCESS_NUM = 15
MAX_HOLD = 5000


def generate_part_dense(feats_queue, qa_queue):
    Qs = []
    As = []
    hold = 0
    while True:
        try:
            data_feat = feats_queue.get(timeout=5)
            _, feat = data_feat

            Qs.append(feat[0])
            As.append(feat[1])
            hold += 1
            if hold == MAX_HOLD:
                qa_queue.put((np.asarray(Qs), np.asarray(As)))
                hold = 0
                Qs = []
                As = []
        except Empty:
            if Qs:
                qa_queue.put((np.asarray(Qs), np.asarray(As)))
            break


def generate_part_sparse(feats_queue, qa_queue):
    Qs = []
    As = []
    hold = 0
    while True:
        try:
            data_feat = feats_queue.get(timeout=5)
            _, feat = data_feat

            Qs.append(feat[0])
            As.append(feat[1])
            hold += 1
            if hold == MAX_HOLD:
                qa_queue.put((csr_matrix(Qs, dtype='float32'), csr_matrix(As, dtype='float32')))
                hold = 0
                Qs = []
                As = []
        except Empty:
            if Qs:
                qa_queue.put((np.asarray(Qs), np.asarray(As)))
            break


def generate_dense(qa_queue, length):
    Qs = []
    As = []
    count = 0
    while True:
        try:
            Qs_temp, As_temp = qa_queue.get()
            count += Qs_temp.shape[0]
            Qs.append(Qs_temp)
            As.append(As_temp)
            if count == length:
                raise Empty
            if count % INF_FREQ == 0:
                logging.info("loading: %d/%d, %.2f%%" % (count, length, count / length * 100))
        except Empty:
            pass

    logging.info("loading: %d/%d, %.2f%%" % (count, length, count / length * 100))
    return np.asarray(Qs), np.asarray(As)


def generate_sparse(qa_queue, length):
    Qs = []
    As = []
    count = 0
    while True:
        try:
            Qs_temp, As_temp = qa_queue.get()
            count += Qs_temp.shape[0]
            Qs.append(Qs_temp)
            As.append(As_temp)
            if count == length:
                raise Empty
            if count % INF_FREQ == 0:
                logging.info("loading: %d/%d, %.2f%%" % (count, length, count / length * 100))
        except Empty:
            pass

    logging.info("loading: %d/%d, %.2f%%" % (count, length, count / length * 100))
    return csr_matrix(Qs, dtype='float32'), csr_matrix(Qs, dtype='float32')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Define training process.')
    parser.add_argument('--feature', type=str, default='unigram', help="Feature option: %s" % (", ".join(FEATURE_OPTS)))
    parser.add_argument('--freq', type=int, default=INF_FREQ, help='Information print out frequency')
    parser.add_argument('--sparse', action='store_true', default=False,
                        help='Use sparse matrix for C_AA, C_AB and C_BB')
    parser.add_argument('--svds', type=int, default=-1, help='Define k value for svds, otherwise use full svd')
    parser.add_argument('--CCA_stage', type=int, default=1,
                        help='Use 2 stage CCA, set as -1 for train paraphrase CCA')
    parser.add_argument('--reuse', default=[], nargs=2, help='Reuse pre-trained data, two arguments: stage file')
    parser.add_argument('--worker', type=int, default=PROCESS_NUM, help='Process number')

    args = parser.parse_args()
    feature = args.feature
    INF_FREQ = args.freq
    k = args.svds
    sparse = args.sparse
    full_svd = k == -1
    PROCESS_NUM = args.worker

    # middle-fix for dump binary file name
    mid_fix = feature

    # 1/2 stage CCA
    cca_stage = args.CCA_stage
    assert cca_stage in [1, 2, -1], "can only use 1 stage CCA or 2 stage CCA"
    use_paraphrase_map = False
    if cca_stage == -1:
        mid_fix = '2stage'
        train_two_stage_cca = True
    else:
        train_two_stage_cca = False
        if cca_stage == 2:
            use_paraphrase_map = True
            assert os.path.exists(PARA_MAP_FILE), "%s not exist" % PARA_MAP_FILE

    # reuse trained model
    reuse_stage = 0
    file = None
    reuse_arg = args.reuse
    reuse_stages = ['features', 'xcov']
    if reuse_arg:
        reuse_stage_name, file = reuse_arg
        reuse_stage = reuse_stages.index(reuse_stage_name) + 1

    # dump file name
    QA_PAIR_FILE = QA_PAIR_FILE % mid_fix
    XCOV_FILE = XCOV_FILE % mid_fix
    if train_two_stage_cca:
        MODULE_FILE = PARA_MAP_FILE
    else:
        if use_paraphrase_map:
            MODULE_FILE = CCA_FILE % ("with_para", mid_fix)
        else:
            MODULE_FILE = CCA_FILE % ("no_para", mid_fix)

    if reuse_stage == 0:
        # not reuse half-finished pretrained model
        logging.info("constructing train data")
        if sparse:
            generate_part = generate_part_sparse
            generate = generate_sparse
        else:
            generate_part = generate_part_dense
            generate = generate_dense

        feats_queue = Queue()
        qa_queue = Queue()
        feats = feats_loader(feature, usage='train', train_two_stage_cca=train_two_stage_cca)
        length = len(feats)

        p_list = [Process(target=generate_part, args=(feats_queue, qa_queue))
                  for i in range(PROCESS_NUM)]

        for p in p_list:
            p.daemon = True
            p.start()

        for _, feat in enumerate(feats):
            feats_queue.put(feat)

        Qs, As = generate(qa_queue, length)
        with open(QA_PAIR_FILE, 'wb') as f:
            pkl.dump((Qs, As), f, protocol=4)

        for p in p_list:
            p.join()

    if reuse_stage == 1:
        # reuse QA pair data
        logging.info("loading pre-constructed data")
        assert file is not None
        with open(file, 'rb') as f:
            Qs, As = pkl.load(f)

    if sparse:
        logging.info("using sparse matrix")
    else:
        logging.info("using dense matrix")

    if use_paraphrase_map:
        logging.info("load paraphrase questions data")
        with open(PARA_MAP_FILE, 'rb') as f:
            paraphrase_map_U, paraphrase_map_V = pkl.load(f)
        logging.info("project question matrix use paraphrase questions")
        Qs1 = Qs.dot(paraphrase_map_U)
        Qs2 = Qs.dot(paraphrase_map_V)
        Qs = np.hstack((Qs1, Qs2))

    if 0 <= reuse_stage <= 1:
        c_qq_sqrt, c_aa_sqrt, result = xcov(Qs, As, sparse=sparse)
        with open(XCOV_FILE, 'wb') as f:
            pkl.dump((c_qq_sqrt, c_aa_sqrt, result), f, protocol=4)

    if reuse_stage == 2:
        logging.info("loading pre-trained xcov")
        assert file is not None
        with open(file, 'rb') as f:
            c_qq_sqrt, c_aa_sqrt, result = pkl.load(f)

    if full_svd:
        logging.info("running CCA, using full SVD")
    else:
        logging.info("running CCA, using SVDs, k=%d" % k)

    Q_k, A_k = decompose(c_qq_sqrt, c_aa_sqrt, result, full_svd=full_svd, k=k)

    with open(MODULE_FILE, 'wb') as f:
        pkl.dump((Q_k, A_k), f, protocol=4)

    logging.info("Done")

