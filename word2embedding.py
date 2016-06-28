from collections import defaultdict
from preprocess.data import BNCembedding
from preprocess.feats import EMBEDDING_SIZE
from word2index import VOC_DICT_FILE
import logging
import pickle as pkl
import os
import sys
import numpy as np
from functools import partial

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
WORD_EMBEDDING_FILE = './bin/word_embedding.pkl'


def generate_dictionary(voc_indx_map, w_emb_map):
    embedding_dict = defaultdict(partial(np.zeros, EMBEDDING_SIZE))
    for w, indx in voc_indx_map.items():
        try:
            emb = np.asarray(w_emb_map[w], dtype='float64')
        except KeyError:
            continue
        embedding_dict[indx] = emb
    return embedding_dict


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

    word_emb_hash_group = {}
    for indx, voc_indx_map in voc_dict.items():
        word_emb_hash_group[indx] = generate_dictionary(voc_indx_map, w_emb_map)

    logging.info("Saving word embedding dictionary")
    with open(WORD_EMBEDDING_FILE, 'wb') as f:
        pkl.dump(word_emb_hash_group, f)

    logging.info("Free up memory")
    del w_emb_map, word_emb_hash_group
