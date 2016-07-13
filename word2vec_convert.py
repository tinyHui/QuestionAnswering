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

    if os.path.exists(WORD_EMBEDDING_BIN_FILE):
        logging.info("Word embedding dictionary file exists, skip")
        sys.exit(0)

    src_data = WordEmbeddingRaw()

    line_num = 1
    if not src_data.word_is_index:
        # the word embedding text file use raw word
        logging.info("loading vocabulary index")
        with open(UNIGRAM_DICT_FILE, 'rb') as f:
            voc_dict = pkl.load(f)

        logging.info("converting raw embedding text")
        word_emb_hash_group = {}
        for sent_indx in voc_dict.keys():
            # for unseen words, the embedding is zero \in R^Embedding_size
            word_emb_hash_group[sent_indx] = defaultdict(partial(np.zeros, EMBEDDING_SIZE))

        for w, emb in src_data:
            sys.stdout.write("\rLoad: %d/%d, %.2f%%" % (line_num, len(src_data), line_num/len(src_data)))
            sys.stdout.flush()
            line_num += 1
            for sent_indx in voc_dict.keys():
                try:
                    # find index of the word
                    indx = voc_dict[sent_indx][w]
                    # hash word index to word embedding
                    word_emb_hash_group[sent_indx][indx] = np.asarray(emb, dtype='float32')
                except KeyError:
                    continue
    else:
        # the word embedding text file use word index directly
        logging.info("use word as index")
        word_emb_hash_group = {}
        for w, emb in src_data:
            sys.stdout.write("\rLoad: %d/%d, %.2f%%" % (line_num, len(src_data), line_num/len(src_data)))
            sys.stdout.flush()
            line_num += 1
            word_emb_hash_group[w] = np.asarray(emb, dtype='float32')

    sys.stdout.write("\n")
    logging.info("Saving word embedding dictionary")
    with open(WORD_EMBEDDING_BIN_FILE, 'wb') as f:
        pkl.dump(word_emb_hash_group, f)