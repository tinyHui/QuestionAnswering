from text2index import VOC_DICT_FILE
from preprocess.data import QAs
from preprocess.feats import BoW, LSTM, WordEmbedding
from siamese_cosine import LSTM_FILE, train_lstm
from text2embedding import WORD_EMBEDDING_FILE
from CCA import CCA
import argparse
import os
import pickle as pkl
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
CCA_FILE = "CCA_model_%s.pkl"
INF_FREQ = 300

if __name__ == "__main__":
    feature_opt = ['bow', 'lstm', 'we']

    parser = argparse.ArgumentParser(description='Define training process.')
    parser.add_argument('--feature', nargs=1, default='bow', help="Feature option: %s" % (", ".join(feature_opt)))

    args = parser.parse_args()
    feature = args.feature[0]

    with open(VOC_DICT_FILE, 'rb') as f:
        voc_dict = pkl.load(f)

    data = QAs(usage='train', mode='index', voc_dict=voc_dict)
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

    elif feature == feature_opt[2]:
        # word embedding
        feats = WordEmbedding(data, WORD_EMBEDDING_FILE)

    else:
        raise IndexError("%s is not an available feature" % feature)

    logging.info("constructing train data")
    length = len(feats)
    for i, feat in enumerate(feats):
        if i % INF_FREQ == 0 or i + 1 == length:
            logging.warning("loading: %d/%d" % (i + 1, length))
        Qs.append(feat[0])
        As.append(feat[1])

    logging.info("running CCA")
    model.train(Qs, As)

    logging.info("dumping model into binary file")
    # dump to disk for reuse
    with open(CCA_FILE % feature, 'wb') as f:
        pkl.dump(model, f)

