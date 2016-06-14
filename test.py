from text2index import VOC_DICT_FILE
from preprocess.data import QAs
from train import CCA_FILE
from preprocess.feats import FEATURE_OPTS, data2feats
from CCA import CCA
import argparse
import pickle as pkl
import logging

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

    logging.info("constructing testing data")
    length = len(feats)
    i = 1
    for feat in feats:
        if i % INF_FREQ == 0 or i == length:
            logging.warning("loading: %d/%d" % (i, length))
        Qs.append(feat[0])
        As.append(feat[1])
        i += 1

    logging.info("loading CCA model")
    # load CCA model
    with open(CCA_FILE % feature, 'rb') as f:
        model = pkl.load(f)
    assert isinstance(model, CCA)

    logging.info("testing")
    correct_num = 0
    for i, q in enumerate(Qs):
        if i % INF_FREQ == 0 or i + 1 == length:
            logging.warning("tested: %d/%d" % (i + 1, length))
        pred = model.find_answer(q, As)
        if pred == i:
            # correct
            correct_num += 1

    # output result
    accuracy = float(correct_num) / len(QAs)
    print("The model get %d/%d correct, precision: %f" % (correct_num, len(QAs), accuracy))



