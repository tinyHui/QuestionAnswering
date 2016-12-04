WORD_EMBEDDING_FILE = './data/embedding.GigaPara.txt'
WORD_EMBEDDING_BIN_FILE = './bin/unigram_embedding.pkl'
LOW_FREQ_TOKEN_FILE = './bin/unigram_low_freq_voc.pkl'
EMBEDDING_SIZE = 300


if __name__ == "__main__":
    from preprocess.data import GigawordRaw, ParaphraseWikiAnswer
    from gensim.models import Word2Vec
    import os
    import sys
    import logging

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    if os.path.exists(WORD_EMBEDDING_FILE):
        logging.info("Word embedding text file exists, exit")
        sys.exit(0)


    class Raw():
        def __iter__(self):
            sentences = GigawordRaw()
            for sentence in sentences:
                yield sentence

            sentences = ParaphraseWikiAnswer(mode='raw_token')
            for sentence_pair in sentences:
                for i in sentences.sent_indx:
                    yield sentence_pair[i]

    sentences = Raw()
    # calculate embedding vector
    logging.info("Generating embedding vectors")
    model = Word2Vec(sentences, size=EMBEDDING_SIZE, window=5, min_count=1, workers=40)
    model.save_word2vec_format(WORD_EMBEDDING_FILE, binary=False)
