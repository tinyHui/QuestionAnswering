from collections import defaultdict
from preprocess.data import BNCembedding
import logging
import pickle as pkl
import os
import sys
import numpy as np

WORD_EMBEDDING_FILE = './bin/word_embedding.pkl'


if __name__ == "__main__":
    if os.path.exists(WORD_EMBEDDING_FILE):
        logging.info("Word embedding dictionary file exists, skip")
        sys.exit(0)

    logging.info("Generating source data")
    # data is a group of sentences
    embedding_dict = defaultdict(np.array)
    src_data = BNCembedding()
    for w, emb in src_data:
        embedding_dict[w] = np.asarray(emb, dtype='float64')

    logging.info("Saving word embedding dictionary")
    with open(WORD_EMBEDDING_FILE, 'wb') as f:
        pkl.dump(WORD_EMBEDDING_FILE, f)

    logging.info("Free up memory")
    del embedding_dict
