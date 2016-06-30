from word2index import VOC_DICT_FILE
from preprocess.data import ReVerbPairs
from preprocess.feats import FEATURE_OPTS, data2feats
from scipy.sparse import csr_matrix, vstack
from multiprocessing import Queue, Process
from CCA import train
import argparse
import pickle as pkl
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
CCA_FILE = "./bin/CCA_model_%s.pkl"
INF_FREQ = 1000  # information message frequency
PROCESS_NUM = 20


def generate(feature_set, q):
    i = 1
    Qs = None
    As = None
    for feature in feature_set:
        _, feat = feature

        if isinstance(Qs, type(None)):
            Qs = csr_matrix(feat[0], dtype='float64')
            As = csr_matrix(feat[1], dtype='float64')
        else:
            Qs = vstack((Qs, feat[0]))
            As = vstack((As, feat[1]))

        if i % INF_FREQ == 0:
            q.put((Qs, As))
            # reset Qs and As
            Qs = None
            As = None
        i += 1


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

    logging.info("constructing train data")
    data_list = [ReVerbPairs(usage='train', part=i, mode='index', voc_dict=voc_dict) for i in range(PROCESS_NUM)]
    feats_list = [data2feats(data, feature) for data in data_list]
    pair_num = sum([len(data) for data in data_list])
    length = sum([len(feats) for feats in feats_list])

    Qs = None
    As = None
    pool = []
    temp_qa = Queue()
    for i in range(PROCESS_NUM):
        p = Process(target=generate, args=(feats_list[i], temp_qa))
        p.start()

    count = 0
    while True:
        # pending until have result
        temp = temp_qa.get()
        if temp is None:
            break
        Qs_temp, As_temp = temp
        if isinstance(Qs, type(None)):
            Qs = csr_matrix(Qs_temp, dtype='float64')
            As = csr_matrix(As_temp, dtype='float64')
        else:
            Qs = vstack((Qs, Qs_temp))
            As = vstack((As, As_temp))

        count += INF_FREQ
        logging.info("loading: %d/%d" % (count, length))

    for i in range(PROCESS_NUM):
        p = pool[i]
        p.join()

    # single thread
    # i = 1
    # for t in feats:
    #     _, feat = t
    #
    #     if i % INF_FREQ == 0 or i == length:
    #         logging.info("loading: %d/%d" % (i, length))
    #         if isinstance(Qs, type(None)):
    #             Qs = csr_matrix(Qs_temp, dtype='float64')
    #             As = csr_matrix(As_temp, dtype='float64')
    #         else:
    #             Qs = vstack((Qs, Qs_temp))
    #             As = vstack((As, As_temp))
    #         del Qs_temp[:], As_temp[:]
    #
    #     Qs_temp.append(feat[0])
    #     As_temp.append(feat[1])
    #
    #     i += 1

    if not full_svd:
        logging.info("running CCA, using SVDs, k=%d" % k)
    else:
        logging.info("running CCA, using full SVD")

    Q_k, A_k = train(Qs, As, sample_num=pair_num, diag_only=diag_only, full_svd=full_svd, k=k)

    logging.info("dumping model into binary file")
    # dump to disk for reuse
    with open(CCA_FILE % feature, 'wb') as f:
        pkl.dump((Q_k, A_k), f, protocol=4)

