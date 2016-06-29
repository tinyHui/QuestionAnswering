from word2index import VOC_DICT_FILE
from preprocess.data import ReVerbPairs
from preprocess.feats import FEATURE_OPTS, data2feats
from scipy.sparse import csr_matrix, vstack
from collections import UserList
from CCA import train
import argparse
import pickle as pkl
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
CCA_FILE = "./bin/CCA_model_%s.pkl"
INF_FREQ = 1000  # information message frequency

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Define training process.')
    parser.add_argument('--feature', type=str, default='bow', help="Feature option: %s" % (", ".join(FEATURE_OPTS)))
    parser.add_argument('--freq', type=int, default=INF_FREQ, help='Information print out frequency')
    parser.add_argument('--svds', type=int, default=-1, help='Define k value for svds, otherwise use full svd')
    parser.add_argument('--diag_only', action='store_true', default=False,
                        help='Use only diagonal value for C_AA and C_BB')

    args = parser.parse_args()
    feature = args.feature
    INF_FREQ = args.freq
    k = args.svds
    full_svd = k == -1
    diag_only = args.diag_only

    logging.info("loading vocabulary index")
    with open(VOC_DICT_FILE, 'rb') as f:
        voc_dict = pkl.load(f)

    data = ReVerbPairs(usage='train', mode='index', voc_dict=voc_dict)

    feats = data2feats(data, feature)
    pair_num = len(data)
    Qs = None
    As = None
    Qs_temp = UserList()
    As_temp = UserList()

    logging.info("constructing train data")
    length = len(feats)
    i = 1
    for t in feats:
        _, feat = t

        if i % INF_FREQ == 0 or i == length:
            logging.info("loading: %d/%d" % (i, length))
            del Qs_temp[:], As_temp[:]
            if Qs is None:
                Qs = csr_matrix(Qs_temp, dtype='float64')
                As = csr_matrix(Qs_temp, dtype='float64')
            else:
                Qs = vstack((Qs, Qs_temp))
                As = vstack((As, As_temp))

        Qs_temp.append(feat[0])
        As_temp.append(feat[1])

        i += 1

    if not full_svd:
        logging.info("running CCA, using SVDs, k=%d" % k)
    else:
        logging.info("running CCA, using full SVD")

    Q_k, A_k = train(Qs, As, sample_num=pair_num, diag_only=diag_only, full_svd=full_svd, k=k)

    logging.info("dumping model into binary file")
    # dump to disk for reuse
    with open(CCA_FILE % feature, 'wb') as f:
        pkl.dump((Q_k, A_k), f, protocol=4)

