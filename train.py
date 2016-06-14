from text2index import VOC_DICT_FILE
from preprocess.data import QAs
from preprocess.feats import FEATURE_OPTS, data2feats
from CCA import train
import argparse
import pickle as pkl
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
CCA_FILE = "./bin/CCA_model_%s.pkl"
INF_FREQ = 300

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Define training process.')
    parser.add_argument('--feature', type=str, default='bow', help="Feature option: %s" % (", ".join(FEATURE_OPTS)))
    parser.add_argument('--freq', type=int, default=300, help='Information print out frequency')

    args = parser.parse_args()
    feature = args.feature
    INF_FREQ = args.freq

    with open(VOC_DICT_FILE, 'rb') as f:
        voc_dict = pkl.load(f)

    data = QAs(usage='train', mode='index', voc_dict=voc_dict)
    feats = None
    Qs = []
    As = []

    feats = data2feats(data, feature)

    logging.info("constructing train data")
    length = len(feats)
    i = 1
    for feat in feats:
        if i % INF_FREQ == 0 or i == length:
            logging.info("loading: %d/%d" % (i, length))
        Qs.append(feat[0])
        As.append(feat[1])
        i += 1

    logging.info("running CCA")
    U, V = train(Qs, As)

    logging.info("dumping model into binary file")
    # dump to disk for reuse
    with open(CCA_FILE % feature, 'wb') as f:
        pkl.dump((U, V), f, protocol=4)

