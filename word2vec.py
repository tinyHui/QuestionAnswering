UNKNOWN_TOKEN = 'UNKNOWN'
WORD_EMBEDDING_FILE = './data/embedding.wikianswer.txt'
WORD_EMBEDDING_BIN_FILE = './bin/unigram_embedding.pkl'
LOW_FREQ_TOKEN_FILE = './bin/unigram_low_freq_voc.pkl'
EMBEDDING_SIZE = 300


class PairSpliter(object):
    def __init__(self, data, unknown_tokens):
        self.data = data
        self.unknown_tokens = unknown_tokens

    def __iter__(self):
        for line in self.data:
            for i in self.data.sent_indx:
                yield [token if token in self.unknown_tokens else UNKNOWN_TOKEN for token in line[i]]


if __name__ == "__main__":
    from gensim.models import Word2Vec
    from preprocess.data import ParaphraseWikiAnswer
    from collections import defaultdict
    import os
    import sys
    import logging
    import pickle as pkl

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    if os.path.exists(WORD_EMBEDDING_FILE):
        logging.info("Word embedding text file exists, exit")
        sys.exit(0)

    # read through the file, extract all tokens with low frequency
    freq_dict = defaultdict(int)
    data = ParaphraseWikiAnswer(mode='raw_token')
    for line in data:
        for i in data.sent_indx:
            for token in line[i]:
                freq_dict[token] += 1

    unknown_tokens = []
    for token, freq in freq_dict.items():
        if freq < 3:
            unknown_tokens.append(token)

    with open(LOW_FREQ_TOKEN_FILE, 'wb') as f:
        pkl.dump(unknown_tokens, f, protocol=4)

    sentences = PairSpliter(ParaphraseWikiAnswer(mode='raw_token'), unknown_tokens)
    # calculate embedding vector
    logging.info("Generating embedding vectors")
    model = Word2Vec(sentences, size=EMBEDDING_SIZE, window=5, min_count=1, workers=40)
    model.save_word2vec_format(WORD_EMBEDDING_FILE, binary=False)
