from calendar import month_name, month_abbr
from collections import UserDict
from random import sample
from word2vec import WORD_EMBEDDING_FILE
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
    SYM = r'(\.|\?|\$|\*|\#|\&)'
    SPACES = r' +'
    # replace all matched phrase to TOKEN name
    RE_SET = [(GRAMMAR_SYM, ' \\1'), (DATE, ' DATE '), (YEAR, ' DATE '), (TIME, ' TIME '), (MONEY, ' MONEY '),
              (PRESENT, ' PRESENT '), (NUMBER, ' NUM '), (EMAIL, ' EMAIL '), (SYM, ' \\1 '), (SPACES, ' ')]
    for p, t in RE_SET:
        s = re.sub(p, t, s)
    s = re.sub(r'\-', ' ', s).strip()
    return s


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
                q = process_raw(q)
                a = process_raw(a)
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
            # normalize question
            q = process_raw(q)
            # normalize answer
            try:
                r, e1, e2 = a.split()
            except ValueError:
                r, e1 = a.split()
                e2 = "PLACEHOLDER"
            r, e1, e2 = [process_raw(w) for w in [r, e1, e2]]
            a = '{e1}|{r}|{e2}'.format(r=r, e1=e1, e2=e2)

            yield q_id, q, a, l

    def __str__(self):
        return "ReVerb test raw"


# paraphrase
class ParaphraseQuestionRaw(object):
    def __init__(self, mode='str', grams=1):
        if mode == 'str':
            suf = 'txt'
        elif mode == 'index':
            suf = 'indx'
        elif mode == 'embedding':
            suf = 'emb'
        else:
            raise AttributeError("Mode can be only 'str', 'index' or 'embedding")
        self.__mode = mode
        self.__grams = grams

        self.__file = './data/paraphrases.%s' % suf
        # index of return data contains sentence
        self.sent_indx = (0, 1)

    def __iter__(self):
        for line in open(self.__file, 'r'):
            q1, q2, align = line.strip().split('\t')
            # word generalise
            if self.__mode == 'str':
                q1 = process_raw(q1)
                q2 = process_raw(q2)
            # to token
            q1_tokens, q2_tokens = [s.split() for s in [q1, q2]]
            if self.__mode == 'str':
                if self.__grams > 1:
                    q1_tokens = [w1 + " " + w2 for w1, w2 in zip(*[q1_tokens[j:] for j in range(self.__grams)])]
                    q2_tokens = [w1 + " " + w2 for w1, w2 in zip(*[q2_tokens[j:] for j in range(self.__grams)])]
            elif self.__mode == 'index':
                q1_tokens = list(map(int, q1_tokens))
                q2_tokens = list(map(int, q2_tokens))
            elif self.__mode == 'embedding':
                q1_tokens = [np.asarray(list(map(float, w.split('|'))), dtype='float32') for w in q1_tokens]
                q2_tokens = [np.asarray(list(map(float, w.split('|'))), dtype='float32') for w in q2_tokens]
            yield q1_tokens, q2_tokens, align

    def get_voc_num(self, i):
        voc_num = {0: 6711, 1: 6671}
        return voc_num[i]

    def is_q_indx(self, i):
        return True

    def __str__(self):
        return "ReVerb Paraphrase Questions raw"

    def __len__(self):
        # head -50000
        return 50000

        # full
        # return 35291309


# q-a pairs generated by reverb
class ReVerbPairs(object):
    def __init__(self, usage='train', mode='str', grams=1):
        if mode == 'str':
            suf = 'txt'
            self.__origin_answer = ""
        elif mode == 'index':
            suf = 'indx'
        elif mode == 'embedding':
            suf = 'emb'
            self.__q_struct_str = ""
            self.__a_struct_str = ""
        else:
            raise AttributeError("Mode can be only 'str', 'index' or 'embedding")
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

            q_tokens = q.split()
            a_tokens = a.split()

            if self.__mode == 'str':
                self.__origin_answer = a
                e1_tokens, r_tokens, e2_tokens = [t.split() for t in a.split('|')]
                a_tokens = e1_tokens + r_tokens + e2_tokens
            if self.__grams > 1:
                    q_tokens = [w1 + " " + w2 for w1, w2 in zip(*[q_tokens[j:] for j in range(self.__grams)])]
            elif self.__mode == 'index':
                q_tokens = list(map(int, q_tokens))
                a_tokens = list(map(int, a_tokens))
            elif self.__mode == 'embedding':
                # these splitter are from raw_converter.py
                q_tokens, self.__q_struct_str = q.split('@')
                a_tokens, self.__a_struct_str = a.split('@')
                q_tokens = [np.asarray(list(map(float, w.split('|'))), dtype='float32') for w in q_tokens]
                a_tokens = [np.asarray(list(map(float, w.split('|'))), dtype='float32') for w in a_tokens]

            # produce the token per line
            if self.__usage == 'train':
                yield (q_tokens, a_tokens)
            else:
                # test
                yield (int(q_id), q_tokens, a_tokens, int(l))

    def get_qa_struct(self):
        assert self.__mode == 'embedding'
        # TODO: parse struct string
        return self.__q_struct_str, self.__a_struct_str

    def get_origin_answer(self):
        assert self.__mode == 'str'
        return self.__origin_answer

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
            voc_num = {q_indx:10519, a_indx:15316}
        elif self.__grams == 2:
            voc_num = {q_indx:0, a_indx:0}
        elif self.__grams == 3:
            voc_num = {q_indx:0, a_indx:0}
        return voc_num[i]

    def get_usage(self):
        return self.__usage

    def get_mode(self):
        return self.__mode

    def __len__(self):
        if self.__usage == 'train':
            return 149993

            # 3 patterns, use all triples
            # return 35540263

            # full patterns
            # return 117202052

        elif self.__usage == 'test':
            return 48910

    def __str__(self):
        return "ReVerb QA pairs"


# Word Embedding
class WordEmbeddingRaw(object):
    def __init__(self):
        self.__file = WORD_EMBEDDING_FILE

    def __iter__(self):
        line_num = 0
        for line in open(self.__file, 'r'):
            # skip the first line
            # embedding files generated by gensim always have a size indicator in the first line
            line_num += 1
            if line_num == 1:
                continue
            w, *emb = line.strip().split()
            yield w.lower(), list(map(float, emb))

    def __len__(self):
        return 3000000

    def __str__(self):
        return "Word embeddings"

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
