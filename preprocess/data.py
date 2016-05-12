import nltk
from sys import stdout


class questions(object):
    def __init__(self, file):
        self.file = file
        with open(self.file, 'r') as f:
            self.num_lines = sum(1 for _ in f)
        self.line = ""

    def _get(self, key):
        tmp = self.line.split("\t")
        question = tmp[0]
        try:
            tokens = tmp[1].split()
            pos = tmp[2].split()
            lemma = tmp[3].split()
        except IndexError:
            # there is no recorded POS or lemma for current question
            # use nltk to analyse
            tokens = nltk.tokenize.word_tokenize(question)
            tmp = nltk.pos_tag(tokens)
            pos = [y for _, y in tmp]
            lemma = [nltk.stem.porter.PorterStemmer().stem(x) for x, _ in tmp]

        if key == "tokens":
            return tokens
        elif key == "pos":
            return pos
        elif key == "lemma":
            return lemma


class questions_tokens(questions):
    def __iter__(self):
        for line in open(self.file, 'r'):
            self.line = line.strip()
            yield self._get("tokens")


class questions_pos(questions):
    def __iter__(self):
        for line in open(self.file, 'r'):
            self.line = line.strip()
            yield self._get("pos")


class questions_lemma(questions):
    def __iter__(self):
        for line in open(self.file, 'r'):
            self.line = line.strip()
            yield self._get("lemma")


class answers(object):
    pass
