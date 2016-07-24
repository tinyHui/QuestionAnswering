WORD_EMBEDDING_FILE = './data/embedding.txt'
WORD_EMBEDDING_BIN_FILE = './bin/unigram_embedding.pkl'
EMBEDDING_SIZE = 300


class PairSpliter(object):
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        for line in self.data:
            for i in self.data.sent_indx:
                yield list(map(str, line[i]))


if __name__ == "__main__":
    from gensim.models import Word2Vec
    from preprocess.data import ReVerbPairs
    import os
    import sys
    import logging

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    if os.path.exists(WORD_EMBEDDING_FILE):
        logging.info("Word embedding text file exists, exit")
        sys.exit(0)

    sentences = PairSpliter(ReVerbPairs(usage='train', mode='index'))
    # calculate embedding vector
    logging.info("Generating embedding vectors")
    model = Word2Vec(sentences, size=EMBEDDING_SIZE, window=5, min_count=0, workers=25)
    model.save_word2vec_format(WORD_EMBEDDING_FILE, binary=False)



