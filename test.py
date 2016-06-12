from train import CCA_FILE
from text2index import VOC_DICT_FILE
from preprocess.data import QAs
from siamese_cosine import LSTM_FILE
from preprocess.feats import BoW, LSTM
from CCA import CCA
import argparse
import pickle as pkl
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
INF_FREQ = 300

if __name__ == '__main__':
    feature_opt = ["bow", "lstm"]

    parser = argparse.ArgumentParser(description='Define training process.')
    parser.add_argument('--feature', nargs=1, default='bow', help="Feature option: %s" % (", ".join(feature_opt)))

    args = parser.parse_args()
    feature = args.feature

    with open(VOC_DICT_FILE, 'rb') as f:
        voc_dict = pkl.load(f)

    data = QAs(usage='test', mode='index', voc_dict=voc_dict)
    feats = None
    Qs = []
    As = []

    if feature == feature_opt[0]:
        # bag-of-word
        feats = BoW(data, voc_dict=voc_dict)

    elif feature == feature_opt[1]:
        # sentence embedding by paraphrased sentences
        feats = LSTM(data, lstm_file=LSTM_FILE, voc_dict=voc_dict)

    else:
        raise IndexError("%s is not an available feature" % feature)

    logging.info("constructing testing data")
    length = len(feats)
    for i, feat in enumerate(feats):
        if i % INF_FREQ == 0 or i + 1 == length:
            logging.warning("loading: %d/%d" % (i + 1, length))
        Qs.append(feat[0])
        As.append(feat[1])

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



