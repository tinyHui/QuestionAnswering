import re
from calendar import month_name, month_abbr
import sqlite3

UNKNOWN_TOKEN = -1


def process_raw(raw):
    # to lower case
    s = raw.lower()
    # replace month name to number
    MONTH_NAME = zip([name.lower() for name in month_name[1:]], [name.lower() for name in month_abbr[1:]])
    for i, (name, abbr) in enumerate(MONTH_NAME):
        s = re.sub(r'\b{}\b|\b{}\b'.format(name, abbr), '%02d' % (i + 1), s)

    # define replace pattern
    DATE = r'(([0]?[1-9]|[1][0-2])[\.\/\- ]([0]?[1-9]|[1|2][0-9]|[3][0|1])[\.\/\- ]([0-9]{4}|[0-9]{2}))|' \
           r'(([0]?[1-9]|[1|2][0-9]|[3][0|1])[\.\/\- ]([0]?[1-9]|[1][0-2])[\.\/\- ]([0-9]{4}|[0-9]{2}))'
    TIME = r'[0-2]?[1-9]:[0-5][0-9][ \-]?(am|pm)?'
    MONEY = r'\$[ \-]?\d+(\,\d+)?\.?\d+'
    PRESENT = r'[-+]?\d+(\,\d+)?(\.\d+)?[ \-]?\%'
    NUMBER = r'[-+]?\d+(\,\d+)?(\.\d+)?'
    EMAIL = r'[_a-z0-9-]+(\.[_a-z0-9-]+)*@[a-z0-9-]+' \
            r'(\.[a-z0-9-]+)*\.(([0-9]{1,3})|([a-z]{2,3})|(aero|coop|info|museum|name))'
    # replace all matched phrase to TOKEN name
    RE_SET = [(DATE, 'DATE'), (TIME, 'TIME'), (MONEY, 'MONEY'), (PRESENT, 'PRESENT'), (NUMBER, 'NUM'), (EMAIL, 'EMAIL')]
    for p, t in RE_SET:
        s = re.sub(p, t, s)
    return s


def word2index(w, voc_dict):
    try:
        return voc_dict[w]
    except KeyError:
        # unseen token
        return UNKNOWN_TOKEN


# paraphrased sentences
class PPDB(object):
    def __init__(self, usage='train', mode='str', voc_dict=None):
        if usage in ['train', 'test']:
            self.file = './data/msr_paraphrase_%s.txt' % usage
            self.usage = usage
        else:
            raise SystemError("usage can be only train/test")
        self.voc_dict = voc_dict
        if mode == 'index':
            assert voc_dict is not None, "must take vocabulary-index dictionary."
        self.mode = mode
        # index of return data contains sentence
        self.sent_indx = (0, 1)

    def __iter__(self):
        for line in open(self.file, 'r'):
            q1, q2 = line.strip().split('\t')
            q1_tokens, q2_tokens = [process_raw(s).split() for s in [q1, q2]]
            # insert sentence start/end symbols
            q1_tokens.insert(0, "[")
            q1_tokens.append("]")
            q2_tokens.insert(0, "[")
            q2_tokens.append("]")

            # paraphrased sentences similarity
            similarity_score = 0.0

            if self.mode == 'str':
                yield (q1_tokens, q2_tokens, similarity_score)
            elif self.mode == 'index':
                # index each word using hash dictionary
                q1_tokens_indx, q2_tokens_indx = [[word2index(w, self.voc_dict) for w in s] for s in [q1_tokens, q2_tokens]]
                yield (q1_tokens_indx, q2_tokens_indx, similarity_score)
            else:
                raise AttributeError("Mode can be only 'str' or 'index'")

    def __len__(self):
        if self.usage == 'train':
            return 4077
        elif self.usage == 'test':
            return 1726

    def __str__(self):
        return "PPDB"


# question answer pairs in WikiQA corpus
class WikiQA(object):
    def __init__(self, usage='train', mode='str', voc_dict=None):
        if usage in ['train', 'test']:
            self.file = './data/WikiQA-%s.txt' % usage
            self.usage = usage
        else:
            raise SystemError("usage can be only train/test")
        self.voc_dict = voc_dict
        if mode == 'index':
            assert voc_dict is not None, "must take vocabulary-index dictionary."
        self.mode = mode
        # index of return data contains sentence
        self.sent_indx = (0, 1)

    def __iter__(self):
        for line in open(self.file, 'r'):
            q, a, label = line.strip().split('\t')
            q_tokens, a_tokens = [process_raw(s).split() for s in [q, a]]
            # insert sentence start/end symbols
            q_tokens.insert(0, "[")
            q_tokens.append("]")
            a_tokens.insert(0, "[")
            a_tokens.append("]")
            # convert label as integer, right answer, wrong answer
            label = int(label)

            if self.mode == 'str':
                yield (q_tokens, a_tokens, label)
            elif self.mode == 'index':
                # index each word using hash dictionary
                q_tokens_indx, a_tokens_indx = [[word2index(w, self.voc_dict) for w in s] for s in [q_tokens, a_tokens]]
                yield (q_tokens_indx, a_tokens_indx, label)
            else:
                raise AttributeError("Mode can be only 'str' or 'index'")

    def __len__(self):
        if self.usage == 'train':
            return 1180
        elif self.usage == 'test':
            return 6165

    def __str__(self):
        return "WikiQA"


# question answer pairs generate from ReVerb corpus
class ReVerbRaw(object):
    def __init__(self):
        self.file = './data/reverb-tuples.db'
        conn = sqlite3.connect(self.file)
        c = conn.cursor()
        self.content = c.execute("select * from tuples")
        # define the pattern
        self.normal_pattern_list = [('who {r} {e2} ?', '{r} ( {e2}, {e1} )'),
                                    ('what {r} {e2} ?', '{r} ( {e2}, {e1} )'),
                                    ('who does {e1} {r} ?', '{r} ( {e1}, {e2} )'),
                                    ('what does {e1} {r} ?', '{r} ( {e1}, {e2} )'),
                                    ('what is the {r} of {e2} ?', '{r} ( {e2}, {e1} )'),
                                    ('who is the {r} of {e2} ?', '{r} ( {e2}, {e1} )'),
                                    ('what is {r} by {e1}', '{r} ( {e1}, {e2} )'),
                                    ('who is {e2}\'s {r} ?', '{r} ( {e2}, {e1} )'),
                                    ('what is {e2}\'s {r}', '{r} ( {e2}, {e1} )'),
                                    ('who is {r} by {e1} ?', '{r} ( {e1}, {e2} )')]
        # shared by *-in, *-on
        self.special_pattern_list = [('when did {e1} {r} ?', '{r} ( {e1}, {e2} )'),
                                     ('when was {e1} {r} ?', '{r} ( {e1}, {e2} )'),
                                     ('where was {e1} {r} ?', '{r} ( {e1}, {e2} )')]

    def __iter__(self):
        for r, e1, e2 in self.content:
            r = r.replace('.r', '')
            e1 = e1.replace('.e', '')
            e2 = e2.replace('.e', '')

            # find the suitable pattern
            if r.endswith('-in') or r.endswith('-on'):
                pattern_list = self.special_pattern_list
            else:
                pattern_list = self.normal_pattern_list

            # preprocess & replace '-' with space
            r, e1, e2 = [process_raw(w.replace('-', ' ')) for w in [r, e1, e2]]

            # generate the question
            for s, p in pattern_list:
                q = s.format(r=r, e1=e1, e2=e2)
                a = p.format(r=r, e1=e1, e2=e2)

                yield (q, a)

    def __str__(self):
        return "ReVerb tuples"


# q-a pairs generated by reverb
class ReVerbPairs(object):
    def __init__(self, usage='train', mode='str', voc_dict=None):
        if usage in ['train', 'test']:
            self.file = './data/reverb-%s.txt' % usage
            self.usage = usage
        else:
            raise SystemError("usage can be only train/test")
        self.voc_dict = voc_dict
        if mode == 'index':
            assert voc_dict is not None, "must take vocabulary-index dictionary."
        self.mode = mode
        # index of return data contains sentence
        self.sent_indx = (0, 1)

    def __iter__(self):
        for line in open(self.file, 'r'):
            if self.usage == 'train':
                q, a = line.strip().split('\t')
                q_tokens, a_tokens = [s.split() for s in [q, a]]
            else:
                l, q, a = line.strip().split('\t')
                l = int(l)
                q = re.sub(r'\?', ' ?', q)
                q_tokens = process_raw(q).split()
                a_tokens = a.split()
                # insert brackets into pattern
                a_tokens.insert(1, '(')
                a_tokens.append(')')
                a_tokens = [process_raw(re.sub(r'\.(r|e)', '', w.replace('-', ' '))) for w in a_tokens]

                def to_stem(w):
                    # only work for reverb test
                    if w == "called":
                        w = "call"
                    elif w == "females":
                        w = "female"
                    return w

                a_tokens = [to_stem(w) for p in a_tokens for w in p.split()]

            # produce the token per line
            if self.mode == 'str':
                if self.usage == 'train':
                    yield (q_tokens, a_tokens)
                else:
                    yield (q_tokens, a_tokens, l)
            elif self.mode == 'index':
                # index each word using hash dictionary
                q_tokens_indx = [word2index(w, self.voc_dict[0]) for w in q_tokens]
                a_tokens_indx = [word2index(w, self.voc_dict[1]) for w in a_tokens]
                if self.usage == 'train':
                    yield (q_tokens_indx, a_tokens_indx)
                else:
                    yield (q_tokens_indx, a_tokens_indx, l)
            else:
                raise AttributeError("Mode can be only 'str' or 'index'")

    def __len__(self):
        if self.usage == 'train':
            return 117202052
        elif self.usage == 'test':
            return 48910

    def __str__(self):
        return "ReVerb QA pairs"


# British National Corpus
class BNCembedding(object):
    def __init__(self):
        self.file = './data/embeddings.txt'

    def __iter__(self):
        for line in open(self.file, 'r'):
            w, *emb = line.strip().split()
            yield w, emb

    def __len__(self):
        return 100148

    def __str__(self):
        return "BNC word embeddings"
