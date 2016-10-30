WORD_EMBEDDING_FILE = './data/embedding.gigapara.txt'
WORD_EMBEDDING_BIN_FILE = './bin/unigram_embedding.pkl'
LOW_FREQ_TOKEN_FILE = './bin/unigram_low_freq_voc.pkl'
EMBEDDING_SIZE = 300


if __name__ == "__main__":
    from preprocess.data import Combine
    from gensim.models import Word2Vec
    import os
    import sys
    import logging

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    if os.path.exists(WORD_EMBEDDING_FILE):
        logging.info("Word embedding text file exists, exit")
        sys.exit(0)

    sentences = Combine()
    # calculate embedding vector
    logging.info("Generating embedding vectors")
    model = Word2Vec(sentences, size=EMBEDDING_SIZE, window=5, min_count=1, workers=40)
    model.save_word2vec_format(WORD_EMBEDDING_FILE, binary=False)
