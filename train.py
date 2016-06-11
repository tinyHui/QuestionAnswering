from text2index import VOC_DICT_FILE
from preprocess.data import QAs
from preprocess.feats import BoW, LSTM
from siamese_cosine import LSTM_FILE, train_lstm
from CCA import CCA
import argparse
import os
import pickle as pkl
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
CCA_FILE = "CCA_model_%s.pkl"

if __name__ == "__main__":
    feature_opt = ["bow", "lstm"]

    parser = argparse.ArgumentParser(description='Define training process.')
    parser.add_argument('--feature', nargs=1, default='bow', help="Feature option: %s" % (", ".join(feature_opt)))

    args = parser.parse_args()
    feature = args.feature

    with open(VOC_DICT_FILE, 'rb') as f:
        voc_dict = pkl.load(f)

    data = QAs(mode='index', voc_dict=voc_dict)
    feats = None
    Qs = []
    As = []
    model = CCA()

    if feature == feature_opt[0]:
        # bag-of-word
        feats = BoW(data, voc_dict=voc_dict)

    elif feature == feature_opt[1]:
        # sentence embedding by paraphrased sentences
        if not os.path.exists(LSTM_FILE):
            train_lstm(
                max_epochs=100,
                test_size=2,
                saveto=LSTM_FILE,
                reload_model=True
            )
        feats = LSTM(data, lstm_file=LSTM_FILE, voc_dict=voc_dict)

    else:
        raise IndexError("%s is not an available feature" % feature)

    for feat in feats:
        Qs.append(feat[0])
        As.append(feat[1])

    model.train(Qs, As)

    # dump to disk for reuse
    with open(CCA_FILE % feature, 'wb') as f:
        pkl.dump(model, f)

