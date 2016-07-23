if __name__ == "__main__":
    from word2vec import WORD_EMBEDDING_BIN_FILE
    from preprocess.data import WordEmbeddingRaw
    import pickle as pkl
    import os
    import sys
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    if os.path.exists(WORD_EMBEDDING_BIN_FILE):
        logging.info("Word embedding dictionary file exists, skip")
        sys.exit(0)

    # the word embedding text file use raw word
    src_data = WordEmbeddingRaw()

    logging.info("converting raw embedding text")
    word_emb_hash = {}

    line_num = 1
    for w, emb in src_data:
        sys.stdout.write("\rLoad: %d/%d, %.2f%%" % (line_num, len(src_data), line_num/len(src_data)*100))
        sys.stdout.flush()
        line_num += 1

        word_emb_hash[w] = emb

    sys.stdout.write("\n")
    logging.info("Saving word embedding dictionary")
    with open(WORD_EMBEDDING_BIN_FILE, 'wb') as f:
        pkl.dump(word_emb_hash, f, protocol=4)
