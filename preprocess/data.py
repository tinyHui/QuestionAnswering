from calendar import month_name, month_abbr
from collections import UserDict
from random import sample
from word2vec import WORD_EMBEDDING_FILE
from nltk.tree import Tree
import sqlite3
import re
import numpy as np


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
    GRAMMAR_SYM = r'(\')'
    DATE = r'([0-9]{1,2})?[\.\/\- ][0-9]{1,2}(st|nd|rd|th)?[\.\/\- ][0-9]{4}|' \
           r'[0-9]{4}[\.\/\- ][0-9]{1,2}(st|nd|rd|th)?[\.\/\- ]([0-9]{1,2})?|' \
           '[0-9]{1,2}(st|nd|rd|th)?[\/\- ][0-9]{1,2}|' \
           '[0-9]{1,2}[\/\- ][0-9]{1,2}(st|nd|rd|th)?'
    YEAR = r'(19|20)\d{2}'
    TIME = r'[0-2]?[0-9]:[0-5][0-9][ \-]?(am|pm)?'
    MONEY = r'\$[ \-]?\d+(\,\d+)?\.?\d+'
    PRESENT = r'[-+]?\d+(\,\d+)?(\.\d+)?[ \-]?\%'
    NUMBER = r'[-+]?\d+(\,\d+)?(\.\d+)?(st|nd|rd|th)?'
    EMAIL = r'[_a-z0-9-]+(\.[_a-z0-9-]+)*@[a-z0-9-]+' \
            r'(\.[a-z0-9-]+)*\.(([0-9]{1,3})|([a-z]{2,3})|(aero|coop|info|museum|name))'
    SYM = r'(\.|\?|\$|\#|\&|\,|\!|\;|\`|\~|\"|\\|\:|\+|\-|\*|\/)'
    SPACES = r' +'
    SYM_AT = r'\@'
    # replace all matched phrase to TOKEN name
    RE_SET = [(GRAMMAR_SYM, ' \\1'), (DATE, ' DATE '), (YEAR, ' DATE '), (TIME, ' TIME '), (MONEY, ' MONEY '),
              (PRESENT, ' PRESENT '), (NUMBER, ' NUM '), (EMAIL, ' EMAIL '), (SPACES, ' '),
              (SYM, ' '), (SYM_AT, ' at '), (SPACES, ' ')]
    for p, t in RE_SET:
        s = re.sub(p, t, s)
    s = s.strip()
    return s


# parse grammar tree generated by stanford language tool
def parse_parser_tree(tree, vecs, result):
    for subtree in tree:
        leaves = subtree.leaves()
        leave_num = len(leaves)
        if leave_num > 1:
            crt_result = []
            parse_parser_tree(subtree, vecs, crt_result)
            result.append(crt_result)
        else:
            vec = vecs.pop(0)
            result.append(vec)


def get_struct(is_q_indx, vecs, struct_str):
    # parse the struct string in origin file
    struct = []
    if is_q_indx:
        # question
        tree = Tree.fromstring(struct_str)
        parse_parser_tree(tree, vecs, struct)
    else:
        # split tokens in struct string
        for part in struct_str.split("|"):
            crt_result = []
            for _ in part.split():
                vec = vecs.pop(0)
                crt_result.append(vec)
            struct.append(crt_result)
        e1, r, e2 = struct
        struct = r + [[e1, e2]]

    return struct


# question answer pairs (train) generate from ReVerb corpus
class ReVerbTrainRaw(object):
    def __init__(self):
        self.__file = './data/reverb-tuples.db'
        self.__conn = sqlite3.connect(self.__file)
        c = self.__conn.cursor()
        # 14377737 triples in total
        # 2697790 triples, relation ends with -in
        # 1098684 triples, relation ends with -on
        self.__content = c.execute("""
            SELECT  *
                FROM    (   SELECT  *
                            FROM    tuples
                            WHERE   rel not like '%-in.r' and rel not like '%-on.r'
                            LIMIT 30000
                        )
                UNION
                SELECT  *
                FROM    (   SELECT  *
                            FROM    tuples
                            WHERE   rel like '%-in.r' or rel like '%-on.r'
                            LIMIT 60000
                        );
        """)
        # define the pattern
        self.__normal_q_pattern_list = ['who {r} {e2} ?',
                                        'what {r} {e2} ?',
                                        'who does {e1} {r} ?',
                                        'what does {e1} {r} ?',
                                        'what is the {r} of {e2} ?',
                                        'who is the {r} of {e2} ?',
                                        'what is {r} by {e1} ?',
                                        'who is {e2} \'s {r} ?',
                                        'what is {e2} \'s {r} ?',
                                        'who is {r} by {e1} ?']
        # shared by *-in, *-on
        self.__special_in_q_pattern_list = ['when did {e1} {r} ?',
                                            'when was {e1} {r} ?',
                                            'where was {e1} {r} ?',
                                            'where did {e1} {r} ?']
        self.__special_on_q_pattern_list = ['when did {e1} {r} ?',
                                            'when was {e1} {r} ?']

        # answer pattern
        self.__normal_a_pattern = '{e1}|{r}|{e2}'

    def __iter__(self):
        for r, e1, e2 in self.__content:
            # remove ".e", ".r" in token
            r = r.replace('.r', '')
            e1 = e1.replace('.e', '')
            e2 = e2.replace('.e', '')
            r, e1, e2 = [process_raw(w) for w in [r, e1, e2]]

            # find the suitable pattern
            # random choose some for training, reduce training size
            if r.endswith('-in'):
                q_pattern_list = sample(self.__special_in_q_pattern_list, 1)
            elif r.endswith('-on'):
                q_pattern_list = sample(self.__special_on_q_pattern_list, 1)
            else:
                q_pattern_list = sample(self.__normal_q_pattern_list, 3)
            a_pattern = self.__normal_a_pattern
            # generate the question
            for q_pattern in q_pattern_list:
                q = q_pattern.format(r=r, e1=e1, e2=e2)
                a = a_pattern.format(r=r, e1=e1, e2=e2)
                yield (q, a)
        self.__conn.close()

    def __str__(self):
        return "ReVerb train tuples"


# question answer pairs (test) generate from ReVerb corpus
class ReVerbTestRaw(object):
    # example:
    # q: name of female octopus ?
    # a: (human)(be call)(social animal)
    def __init__(self):
        self.__file = './data/labels.txt'
        # load question cluster id
        self.__q_id_map = UserDict()
        for line in open('./data/clusterid_questions.txt'):
            id, q = line.strip().split('\t')
            self.__q_id_map[q] = id

    def __iter__(self):
        for raw in open(self.__file, 'r'):
            # remove ".e", ".r" in token
            line = raw.replace('.r', '').replace('.e', '')
            l, q, a = line.strip().split('\t')
            q_id = self.__q_id_map[q]

            try:
                r, e1, e2 = a.split()
            except ValueError:
                r, e1 = a.split()
                e2 = "PLACEHOLDER"

            q = process_raw(q) + ' ?'
            r, e1, e2 = [process_raw(w) for w in [r, e1, e2]]
            a = '{e1}|{r}|{e2}'.format(r=r, e1=e1, e2=e2)

            yield q_id, q, a, l

    def __str__(self):
        return "ReVerb test raw"


# Word Embedding
class WordEmbeddingRaw(object):
    def __init__(self):
        self.__file = WORD_EMBEDDING_FILE
        self.f = open(self.__file, 'r')
        for line in self.f:
            self.length, self.emb_size = line.split(' ')
            break

    def __iter__(self):
        for line in self.f:
            w, *emb = line.strip().split()
            yield w.lower(), list(map(float, emb))

        self.f.close()

    def __len__(self):
        return int(self.length)

    def __str__(self):
        return "Word Embeddings"


# paraphrase
class ParaphraseWikiAnswer(object):
    def __init__(self, mode='str', grams=1):
        if mode == 'raw':
            suf = 'txt'
        elif mode == 'str':
            suf = 'txt'
        elif mode == 'embedding':
            # unknown token embedding is the average of low frequency words' embedding
            suf = 'emb'
        elif mode == 'structure':
            suf = 'struct'
        else:
            raise AttributeError("Mode can be only 'str', 'embedding', 'structure'")
        self.__mode = mode
        self.__grams = grams

        self.__file = './data/paraphrases.wikianswer.%s' % suf
        # index of return data contains sentence
        self.sent_indx = (0, 1)

    def __iter__(self):
        for line in open(self.__file, 'r'):
            q1, q2 = line.strip().split('\t')
            if self.__mode == 'raw':
                yield q1, q2
                continue

            # to token
            if self.__mode == 'structure':
                yield q1, q2
            else:
                q1_tokens, q2_tokens = [s.split() for s in [q1, q2]]
                if self.__mode == 'str':
                    if self.__grams > 1:
                        q1_tokens = [w1 + " " + w2 for w1, w2 in zip(*[q1_tokens[j:] for j in range(self.__grams)])]
                        q2_tokens = [w1 + " " + w2 for w1, w2 in zip(*[q2_tokens[j:] for j in range(self.__grams)])]
                elif self.__mode == 'embedding':
                    q1_tokens = [np.asarray(list(map(float, w.split('|'))), dtype='float32') for w in q1_tokens]
                    q2_tokens = [np.asarray(list(map(float, w.split('|'))), dtype='float32') for w in q2_tokens]
                yield q1_tokens, q2_tokens

    def is_q_indx(self, i):
        # both two columns are sentences
        return i in self.sent_indx

    def get_mode(self):
        return self.__mode

    def __str__(self):
        return "WikiAnswer Paraphrase Questions"

    def __len__(self):
        return 267956


class ParaphraseMicrosoftRaw(object):
    def __init__(self, mode='str', grams=1):
        if mode == 'raw':
            suf = 'txt'
        elif mode == 'str':
            suf = 'txt'
        elif mode == 'embedding':
            # unknown token embedding is the average of low frequency words' embedding
            suf = 'emb'
        elif mode == 'structure':
            suf = 'struct'
        else:
            raise AttributeError("Mode can be only 'str', 'embedding', 'structure'")
        self.__mode = mode
        self.__grams = grams

        self.__file = './data/paraphrases.ms.%s' % suf
        # index of return data contains sentence
        self.sent_indx = (3, 4)

    def __iter__(self):
        for line in open(self.__file, 'r'):
            quality, id1, id2, s1, s2 = line.strip().split('\t')
            if self.__mode == 'raw':
                yield quality, id1, id2, s1, s2
                continue

            # word generalise
            if self.__mode == 'str':
                s1 = process_raw(s1)
                s2 = process_raw(s2)

            # to token
            if self.__mode == 'structure':
                yield quality, id1, id2, s1, s2
            else:
                s1_tokens, s2_tokens = [s.split() for s in [s1, s2]]
                if self.__mode == 'str':
                    if self.__grams > 1:
                        s1_tokens = [w1 + " " + w2 for w1, w2 in zip(*[s1_tokens[j:] for j in range(self.__grams)])]
                        s2_tokens = [w1 + " " + w2 for w1, w2 in zip(*[s2_tokens[j:] for j in range(self.__grams)])]
                elif self.__mode == 'embedding':
                    s1_tokens = [np.asarray(list(map(float, w.split('|'))), dtype='float32') for w in s1_tokens]
                    s2_tokens = [np.asarray(list(map(float, w.split('|'))), dtype='float32') for w in s2_tokens]
                yield quality, id1, id2, s1_tokens, s2_tokens

    def is_q_indx(self, i):
        return i in self.sent_indx

    def get_mode(self):
        return self.__mode

    def get_q_indx(self):
        return 3

    def get_a_indx(self):
        return 4

    def __str__(self):
        return "Microsoft Research Paraphrase raw"

    def __len__(self):
        return 3900
        # return 5801


# q-a pairs generated by reverb
class ReVerbPairs(object):
    def __init__(self, usage='train', mode='str', grams=1):
        if mode == 'raw':
            suf = 'txt'
        elif mode == 'str':
            suf = 'txt'
        elif mode == 'index':
            suf = 'indx'
        elif mode == 'embedding':
            # unknown token embedding is 0
            suf = 'emb'
        elif mode == 'structure':
            suf = 'struct'
        else:
            raise AttributeError("Mode can be only 'str', 'index', 'embedding' or 'structure'")
        self.__mode = mode

        if usage in ['train', 'test']:
            self.__file = './data/reverb-%s.%s' % (usage, suf)
            self.__usage = usage
        else:
            raise SystemError("usage can be only train/test")
        # index of return data contains sentence
        if self.__usage == 'train':
            self.sent_indx = (0, 1)
        elif self.__usage == 'test':
            self.sent_indx = (1, 2)
        # n-gram
        self.__grams = grams

    def __iter__(self):
        for line in open(self.__file, 'r'):
            if self.__usage == 'train':
                # train
                q, a = line.strip().split('\t')
            else:
                # test
                q_id, q, a, l = line.strip().split('\t')
                l = int(l)

            if self.__mode in ['raw', 'structure']:
                if self.__usage == 'train':
                    yield (q, a)
                else:
                    # test
                    yield (q_id, q, a, l)
            else:
                q_tokens = q.split()
                a_tokens = a.split()

                if self.__mode == 'str':
                    e1_tokens, r_tokens, e2_tokens = [t.strip().split() for t in a.split('|')]
                    a_tokens = e1_tokens + r_tokens + e2_tokens
                if self.__grams > 1:
                    q_tokens = [w1 + " " + w2 for w1, w2 in zip(*[q_tokens[j:] for j in range(self.__grams)])]
                elif self.__mode == 'index':
                    q_tokens = list(map(int, q_tokens))
                    a_tokens = list(map(int, a_tokens))
                elif self.__mode == 'embedding':
                    q_tokens = [np.asarray(list(map(float, w.split('|'))), dtype='float32') for w in q_tokens]
                    a_tokens = [np.asarray(list(map(float, w.split('|'))), dtype='float32') for w in a_tokens]
                # elif self.__mode == 'structure': keep same

                # produce the token per line
                if self.__usage == 'train':
                    yield (q_tokens, a_tokens)
                else:
                    # test
                    yield (int(q_id), q_tokens, a_tokens, int(l))

    def is_q_indx(self, i):
        if self.__usage == 'train':
            return i == 0
        elif self.__usage == 'test':
            return i == 1

    def get_voc_num(self, i):
        if self.__usage == 'train':
            q_indx = 0
            a_indx = 1
        elif self.__usage == 'test':
            q_indx = 1
            a_indx = 2

        if self.__grams == 1:
            voc_num = {q_indx: 14082, a_indx: 18402}
        elif self.__grams == 2:
            voc_num = {q_indx: 0, a_indx: 0}
        elif self.__grams == 3:
            voc_num = {q_indx: 0, a_indx: 0}
        return voc_num[i]

    def get_usage(self):
        return self.__usage

    def get_mode(self):
        return self.__mode

    def get_q_indx(self):
        if self.__usage == 'train':
            return 0
        elif self.__usage == 'test':
            return 1

    def get_a_indx(self):
        if self.__usage == 'train':
            return 1
        elif self.__usage == 'test':
            return 2

    def __len__(self):
        if self.__usage == 'train':
            return 269979

            # 3 patterns, use all triples
            # return 35540263

            # full patterns
            # return 117202052

        elif self.__usage == 'test':
            return 48910

    def __str__(self):
        return "ReVerb QA pairs"

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
