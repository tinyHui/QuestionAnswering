from preprocess import data
from gensim.models import Word2Vec
import configparser
import os
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

config = configparser.ConfigParser()
config.read('setting.ini')
PATH_DATA = os.path.join(config.get('DATA', 'root'), config.get('DATA', 'wikianswers'))
QUESTIONS = os.path.join(PATH_DATA, config.get('DATA', 'questions'))
LEXICON = os.path.join(PATH_DATA, config.get('DATA', 'lexicon'))


if __name__ == "__main__":
    # train word2vec
    if not os.path.exists(LEXICON):
        tokens = data.questions_lemma(QUESTIONS)
        model = Word2Vec(tokens, workers=4, size=300)
        model.save(LEXICON)
    else:
        model = Word2Vec.load(LEXICON)

    print("Obtained %d words" % len(model.vocab.keys()))
