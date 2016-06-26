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
    embedding_dict = defaultdict(np.array)
    src_data = BNCembedding()
    for w, emb in src_data:
        indx = voc_dict[w]
        embedding_dict[indx] = np.asarray(emb, dtype='float64')

    logging.info("Saving word embedding dictionary")
    with open(WORD_EMBEDDING_FILE, 'wb') as f:
        pkl.dump(WORD_EMBEDDING_FILE, f)

    logging.info("Free up memory")
    del embedding_dict
