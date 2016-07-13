if __name__ == "__main__":
    from word2vec import WORD_EMBEDDING_BIN_FILE, EMBEDDING_SIZE
    from word2index import UNIGRAM_DICT_FILE
    from preprocess.data import WordEmbeddingRaw
    from functools import partial
    from collections import defaultdict
    import pickle as pkl
    import numpy as np
    import os
    import sys

    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    def generate_dictionary(voc_indx_map, w_emb_map):
        embedding_dict = defaultdict(partial(np.zeros, EMBEDDING_SIZE))
        for w, indx in voc_indx_map.items():
            try:
                emb = np.asarray(w_emb_map[w], dtype='float32')
            except KeyError:
                continue
            embedding_dict[indx] = emb
        return embedding_dict

    if os.path.exists(WORD_EMBEDDING_BIN_FILE):
        logging.info("Word embedding dictionary file exists, skip")
        sys.exit(0)

    src_data = WordEmbeddingRaw()

    if src_data.word_is_index:
        # the word embedding text file use raw word
        logging.info("loading vocabulary index")
        with open(UNIGRAM_DICT_FILE, 'rb') as f:
            voc_dict = pkl.load(f)

        logging.info("loading raw embedding text")
        w_emb_map = {}
        for w, emb in src_data:
            w_emb_map[w] = emb

        word_emb_hash_group = {}
        for indx, voc_indx_map in voc_dict.items():
            word_emb_hash_group[indx] = generate_dictionary(voc_indx_map, w_emb_map)

    else:
        # the word embedding text file use word index directly
        word_emb_hash_group = {}
        for w, emb in src_data:
            word_emb_hash_group[w] = np.asarray(emb, dtype='float32')

    logging.info("Saving word embedding dictionary")
    with open(WORD_EMBEDDING_BIN_FILE, 'wb') as f:
        pkl.dump(word_emb_hash_group, f)