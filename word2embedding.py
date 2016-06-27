from collections import defaultdict
from preprocess.data import BNCembedding
from word2index import VOC_DICT_FILE
import logging
import pickle as pkl
import os
import sys
import numpy as np

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
WORD_EMBEDDING_FILE = './bin/word_embedding.pkl'


if __name__ == "__main__":
    if os.path.exists(WORD_EMBEDDING_FILE):
        logging.info("Word embedding dictionary file exists, skip")
        sys.exit(0)

    logging.info("loading vocabulary index")
    with open(VOC_DICT_FILE, 'rb') as f:
        voc_dict = pkl.load(f)

    logging.info("Generating source data")
    # data is a group of sentences
    src_data = BNCembedding()
    w_emb_map = {}
    for w, emb in src_data:
        w_emb_map[w] = emb

    embedding_dict = defaultdict(np.array)
    for w, indx in voc_dict.items():
        try:
            emb = w_emb_map[w]
        except KeyError:
            continue
        embedding_dict[indx] = np.asarray(emb, dtype='float64')

    logging.info("Saving word embedding dictionary")
    with open(WORD_EMBEDDING_FILE, 'wb') as f:
        pkl.dump(WORD_EMBEDDING_FILE, f)

    logging.info("Free up memory")
    del w_emb_map, embedding_dict
