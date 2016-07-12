from calendar import month_name, month_abbr
from collections import UserDict
from random import sample
from word2embedding import WORD_EMBEDDING_FILE
import sqlite3
import re


UNKNOWN_TOKEN = 'UNKNOWN'
UNKNOWN_TOKEN_INDX = 0


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
        return UNKNOWN_TOKEN_INDX


# question answer pairs (train) generate from ReVerb corpus
class ReVerbTrainRaw(object):
    def __init__(self):
        self.file = './data/reverb-tuples.db'
        conn = sqlite3.connect(self.file)
        c = conn.cursor()
        self.content = c.execute("select * from tuples")
        # define the pattern
        self.normal_pattern_list = [('who {r} {e2} ?', '{e2} {r} {e1}'),
                                    ('what {r} {e2} ?', '{e2} {r} {e1}'),
                                    ('who does {e1} {r} ?', '{e1} {r} {e2}'),
                                    ('what does {e1} {r} ?', '{e1} {r} {e2}'),
                                    ('what is the {r} of {e2} ?', '{e2} {r} {e1}'),
                                    ('who is the {r} of {e2} ?', '{e2} {r} {e1}'),
                                    ('what is {r} by {e1}', '{e1} {r} {e2}'),
                                    ('who is {e2} \'s {r} ?', '{e2} {r} {e1}'),
                                    ('what is {e2} \'s {r}', '{e2} {r} {e1}'),
                                    ('who is {r} by {e1} ?', '{e1} {r} {e2}')]
        # shared by *-in, *-on
        self.special_pattern_list = [('when did {e1} {r} ?', '{e1} {r} {e2}'),
                                     ('when was {e1} {r} ?', '{e1} {r} {e2}'),
                                     ('where was {e1} {r} ?', '{e1} {r} {e2}')]

    def __iter__(self):
        for r, e1, e2 in self.content:
            r = r.replace('.r', '')
            e1 = e1.replace('.e', '')
            e2 = e2.replace('.e', '')

            # find the suitable pattern
            # random choose some for training, reduce training size
            if r.endswith('-in') or r.endswith('-on'):
                pattern_list = sample(self.special_pattern_list, 1)
            else:
                pattern_list = sample(self.normal_pattern_list, 3)

            # preprocess & replace '-' with space
            r, e1, e2 = [process_raw(w.replace('-', ' ')) for w in [r, e1, e2]]

            # generate the question
            for s, p in pattern_list:
                q = s.format(r=r, e1=e1, e2=e2)
                a = p.format(r=r, e1=e1, e2=e2)

                yield (q, a)

    def __str__(self):
        return "ReVerb train tuples"


# question answer pairs (test) generate from ReVerb corpus
class ReVerbTestRaw(object):
    def __init__(self):
        self.file = './data/labels.txt'
        # load question cluster id
        self.q_id_map = UserDict()
        for line in open('./data/clusterid_questions.txt'):
            id, q = line.strip().split('\t')
            self.q_id_map[q] = id

    def __iter__(self):
        def to_stem(w):
            # only work for reverb test
            if w in ["called", "calls"]:
                w = "call"
            elif w == "females":
                w = "female"
            elif w in ["spoken", "speaks"]:
                w = "speak"
            elif w == "languages":
                w = "language"
            elif w == "found":
                w = "find"
            elif w in ["uses", "used"]:
                w = "use"
            elif w == "marked":
                w = "mark"
            elif w == "invented":
                w = "invent"
            elif w == "players":
                w = "player"
            elif w == "made":
                w = "make"
            elif w == "arguments":
                w = "argument"
            elif w == "causes":
                w = "cause"
            elif w in ["are", "is", "was", "were", "been"]:
                w = "be"
            return w

        for line in open(self.file, 'r'):
            l, q, a = line.strip().split('\t')
            q_id = self.q_id_map[q]
            # normalize question
            q = re.sub(r'\?', ' ?', q)
            q = re.sub(r'\'s', ' \'s', q)
            q = re.sub(r'\-', ' ', q)
            q = process_raw(q)
            q = ' '.join([to_stem(w) for w in q.split()])
            # normalize answer
            r, e1, e2 = [process_raw(re.sub(r'\.(r|e)', '', w.replace('-', ' '))) for w in a.split()]
            a = "{e1} {r} {e2}".format(r=r, e1=e1, e2=e2)

            yield q_id, q, a, l

    def __str__(self):
        return "ReVerb test raw"


# q-a pairs generated by reverb
class ReVerbPairs(object):
    def __init__(self, usage='train', part=None, mode='str', gram=1):
        if mode == 'str':
            suf = 'txt'
        elif mode == 'index':
            suf = 'indx'
        else:
            raise AttributeError("Mode can be only 'str' or 'index'")
        self.mode = mode

        if usage in ['train', 'test']:
            if part is not None:
                assert isinstance(part, int), "must provide a part number"
                self.file = './data/reverb-%s.part%d.%s' % (usage, part, suf)
            else:
                self.file = './data/reverb-%s.full.%s' % (usage, suf)
            self.part = part
            self.usage = usage
        else:
            raise SystemError("usage can be only train/test")
        # index of return data contains sentence
        self.sent_indx = (0, 1)
        # n-gram
        self.gram = gram

    def __iter__(self):
        for line in open(self.file, 'r'):
            if self.usage == 'train':
                # train
                q, a = line.strip().split('\t')
            else:
                # test
                q_id, q, a, l = line.strip().split('\t')
                l = int(l)

            q_tokens, a_tokens = [s.split() for s in [q, a]]

            if self.mode == 'str':
                if self.gram > 1:
                    q_tokens = [w1 + " " + w2 for w1, w2 in zip(*[q_tokens[j:] for j in range(self.gram)])]
                    a_tokens = [w1 + " " + w2 for w1, w2 in zip(*[a_tokens[j:] for j in range(self.gram)])]
            else:
                # mode == 'index
                q_tokens = map(int, q_tokens)
                a_tokens = map(int, a_tokens)

            # produce the token per line
            if self.usage == 'train':
                # train
                yield (q_tokens, a_tokens)
            else:
                # test
                yield (q_tokens, a_tokens, int(q_id), int(l))

    def get_voc_num(self, i):
        if self.gram == 1:
            voc_num = {0:251982, 1:303295}
        elif self.gram == 2:
            voc_num = {0: 0, 1: 0}
        elif self.gram == 3:
            voc_num = {0: 0, 1: 0}
        return voc_num[i]

    def __len__(self):
        if self.usage == 'train':
            # part patterns
            if self.part is None:
                return 35540263
            if self.part == 14:
                return 2369349
            else:
                return 2369351

            # full patterns
            # if self.part is None:
            #     return 117202052
            # 15 parts
            # if self.part == 29:
            #     return 3906708
            # else:
            #     return 3906736
            # 30 parts
            # if self.part == 14:
            #     return 7813458
            # else:
            #     return 7813471

        elif self.usage == 'test':
            return 48910

    def __str__(self):
        return "ReVerb QA pairs"


# Word Embedding
class WordEmbeddingRaw(object):
    def __init__(self):
        self.file = WORD_EMBEDDING_FILE

    def __iter__(self):
        line_num = 1
        for line in open(self.file, 'r'):
            # skip the first line
            # embedding files generated by gensim always have a size indicator in the first line
            if line_num == 1:
                continue
            w, *emb = line.strip().split()
            yield w, emb
            line_num += 1

    def __len__(self):
        return 3000000

    def __str__(self):
        return "ReVerb tuples word embeddings"


# paraphrased sentences
# class PPDB(object):
#     def __init__(self, usage='train', mode='str', voc_dict=None):
#         if usage in ['train', 'test']:
#             self.file = './data/msr_paraphrase_%s.txt' % usage
#             self.usage = usage
#         else:
#             raise SystemError("usage can be only train/test")
#         self.voc_dict = voc_dict
#         if mode == 'index':
#             assert voc_dict is not None, "must take vocabulary-index dictionary."
#         self.mode = mode
#         # index of return data contains sentence
#         self.sent_indx = (0, 1)
#
#     def __iter__(self):
#         for line in open(self.file, 'r'):
#             q1, q2 = line.strip().split('\t')
#             q1_tokens, q2_tokens = [process_raw(s).split() for s in [q1, q2]]
#             # insert sentence start/end symbols
#             q1_tokens.insert(0, "[")
#             q1_tokens.append("]")
#             q2_tokens.insert(0, "[")
#             q2_tokens.append("]")
#
#             # paraphrased sentences similarity
#             similarity_score = 0.0
#
#             if self.mode == 'str':
#                 yield (q1_tokens, q2_tokens, similarity_score)
#             elif self.mode == 'index':
#                 # index each word using hash dictionary
#                 q1_tokens_indx, q2_tokens_indx = [[word2index(w, self.voc_dict) for w in s] for s in
#                                                   [q1_tokens, q2_tokens]]
#                 yield (q1_tokens_indx, q2_tokens_indx, similarity_score)
#             else:
#                 raise AttributeError("Mode can be only 'str' or 'index'")
#
#     def __len__(self):
#         if self.usage == 'train':
#             return 4077
#         elif self.usage == 'test':
#             return 1726
#
#     def __str__(self):
#         return "PPDB"


# question answer pairs in WikiQA corpus
# class WikiQA(object):
#     def __init__(self, usage='train', mode='str', voc_dict=None):
#         if usage in ['train', 'test']:
#             self.file = './data/WikiQA-%s.txt' % usage
#             self.usage = usage
#         else:
#             raise SystemError("usage can be only train/test")
#         self.voc_dict = voc_dict
#         if mode == 'index':
#             assert voc_dict is not None, "must take vocabulary-index dictionary."
#         self.mode = mode
#         # index of return data contains sentence
#         self.sent_indx = (0, 1)
#
#     def __iter__(self):
#         for line in open(self.file, 'r'):
#             q, a, label = line.strip().split('\t')
#             q_tokens, a_tokens = [process_raw(s).split() for s in [q, a]]
#             # insert sentence start/end symbols
#             q_tokens.insert(0, "[")
#             q_tokens.append("]")
#             a_tokens.insert(0, "[")
#             a_tokens.append("]")
#             # convert label as integer, right answer, wrong answer
#             label = int(label)
#
#             if self.mode == 'str':
#                 yield (q_tokens, a_tokens, label)
#             elif self.mode == 'index':
#                 # index each word using hash dictionary
#                 q_tokens_indx, a_tokens_indx = [[word2index(w, self.voc_dict) for w in s] for s in [q_tokens, a_tokens]]
#                 yield (q_tokens_indx, a_tokens_indx, label)
#             else:
#                 raise AttributeError("Mode can be only 'str' or 'index'")
#
#     def __len__(self):
#         if self.usage == 'train':
#             return 1180
#         elif self.usage == 'test':
#             return 6165
#
#     def __str__(self):
#         return "WikiQA"
