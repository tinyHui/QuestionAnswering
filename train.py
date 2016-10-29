import numpy as np
import pickle as pkl
import logging
from scipy.linalg import sqrtm as dense_sqrt
from scipy.linalg import inv as dense_inv
from scipy.sparse.linalg import svds
from preprocess.data import ReVerbPairs
from word2vec import EMBEDDING_SIZE
from preprocess.feats import FEATURE_OPTS
from text2feature import Q_MATRIX, A_MATRIX, PARA_1_MATRIX, PARA_2_MATRIX
import argparse
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def load(fname):
    logging.info("Loadding %s" % fname)
    with open(fname, 'rb') as f:
        m = pkl.load(f)
    return m


def dump(obj, fname):
    logging.info("Dumping %s" % fname)
    with open(fname, 'wb') as f:
        pkl.dump(obj, f, protocol=4)


def segment_dump(m, base_fname):
    SEG_HEIGHT = 500000
    h, _ = m.shape

    start = 0
    part_count = 0
    while start < h:
        part = m[start:start+SEG_HEIGHT]
        fname = "{}.part{}".format(base_fname, part_count)
        dump(part, fname)
        start += SEG_HEIGHT
        part_count += 1


def segment_generator(base_fname):
    fname_list = []
    main_fpath = os.path.dirname(base_fname)
    main_fname = os.path.basename(base_fname)
    for fname in os.listdir(main_fpath):
        if fname.startswith(main_fname):
            fname_list.append(fname)

    file_number = len(fname_list)
    logging.info("Found %d files" % file_number)
    for part_count in range(file_number - 1):
        fname = "{}.part{}".format(base_fname, part_count)
        with open(fname, 'rb') as f:
            logging.info("loading: %s, %d/%d" % (fname, part_count+1, len(fname_list)))
            m = pkl.load(f)
        yield m


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Define training process.')
    parser.add_argument('--feature', type=str, default='unigram', help="Feature option: %s" % (", ".join(FEATURE_OPTS)))
    parser.add_argument('--stage', type=str, help="Tain paraphrase data/1-stage CCA/2-stage CCA")

    args = parser.parse_args()
    feature = args.feature
    stage_name = args.stage

    if stage_name == "paraphrase":
        stage = -1
    elif stage_name == "1stage":
        stage = 1
    elif stage_name == "2stage":
        stage = 2
    else:
        raise ValueError("stage can be only set as paraphrase/1stage/2stage")

    if stage == -1:
        L_SEG_MATRIX = PARA_1_MATRIX % feature
        R_SEG_MATRIX = PARA_2_MATRIX % feature
        MODULE_FNAME = "./bin/ParaMap.%s.pkl" % feature
    else:
        L_SEG_MATRIX = Q_MATRIX % feature
        R_SEG_MATRIX = A_MATRIX % feature
        MODULE_FNAME = "./bin/CCA.%s.pkl" % feature

    sample_num = len(ReVerbPairs(usage='train'))

    if stage == 2:
        ParaMap_MATRIX = "./bin/ParaMap.pkl"
        para_1, para_2 = load(ParaMap_MATRIX)                   # R^300 x 300
        _, para_1_width = para_1.shape
        _, para_2_width = para_2.shape
        para_width = para_1_width + para_2_width

        c_qq = np.zeros((para_width, para_width))               # R^600 x 600
        c_qa = np.zeros((para_width, EMBEDDING_SIZE))           # R^600 x 300

    else:
        c_qq = np.zeros((EMBEDDING_SIZE, EMBEDDING_SIZE))
        c_qa = np.zeros((EMBEDDING_SIZE, EMBEDDING_SIZE))
        para_1 = np.eye(EMBEDDING_SIZE)
        para_2 = np.eye(EMBEDDING_SIZE)

    c_aa = np.zeros((EMBEDDING_SIZE, EMBEDDING_SIZE))           # R^300 x 300

    for seg_Q, seg_A in zip(segment_generator(L_SEG_MATRIX),
                            segment_generator(R_SEG_MATRIX)):
        seg_Q = seg_Q.astype('float64')
        seg_A = seg_A.astype('float64')
        logging.info("product with Para map 1")
        seg_Q_Para_map_1 = seg_Q.dot(para_1)                    # R^10000 x 300
        logging.info("product with Para map 2")
        seg_Q_Para_map_2 = seg_Q.dot(para_2)                    # R^10000 x 300
        logging.info("Join together")
        seg_Q_Para_map = np.hstack((seg_Q_Para_map_1, seg_Q_Para_map_2))     # R^10000 x 600
        del seg_Q_Para_map_1, seg_Q_Para_map_2

        logging.info("refresh c_qq, c_aa, c_qa")
        c_qq += seg_Q_Para_map.T.dot(seg_Q_Para_map)
        c_aa += seg_A.T.dot(seg_A)
        c_qa += seg_Q_Para_map.T.dot(seg_A)

    logging.info("keep only diagonal")
    c_qq = np.diag(np.diag(c_qq))                           # R^600 x 600, keep only diagnal values
    c_aa = np.diag(np.diag(c_aa))                           # R^300 x 300, keep only diagnal values
    c_qa /= sample_num                                      # R^600 x 300

    logging.info("doing square root and invert for C_AA")
    c_qq_sqrt = dense_inv(dense_sqrt(c_qq)) / sample_num    # R^600 x 600
    logging.info("doing square root and invert for C_BB")
    c_aa_sqrt = dense_inv(dense_sqrt(c_aa)) / sample_num    # R^300 x 300
    logging.info("C_AA * C_AB * C_BB")
    result = c_qq_sqrt.dot(c_qa).dot(c_aa_sqrt)             # R^600 x 300

    k = 100
    logging.info("decompose on cross covariant matrix \in R^%d x %d" % (result.shape[0], result.shape[1]))
    full_svd = True
    if full_svd:
        U, s, V = np.linalg.svd(result, full_matrices=False)    # U \in R^600 x
                                                                # V \in R^k x 300
    else:
        U, s, V = svds(result, k=k)

    Q_k = c_qq_sqrt.dot(U)                                      # R^600 x k
    A_k = c_aa_sqrt.dot(V.T)                                    # R^300 x k
    dump((Q_k, A_k), MODULE_FNAME)
