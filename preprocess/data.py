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


def tokenize(raw):
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
    SPACES = r' +'
    # replace all matched phrase to TOKEN name
    RE_SET = [(GRAMMAR_SYM, ' \\1'), (DATE, ' DATE '), (YEAR, ' DATE '), (TIME, ' TIME '), (MONEY, ' MONEY '),
              (PRESENT, ' PRESENT '), (NUMBER, ' NUM '), (EMAIL, ' EMAIL '), (SPACES, ' '), (SPACES, ' ')]
    for p, t in RE_SET:
        s = re.sub(p, t, s)
    s = s.strip()
    return s


def no_symbol(s):
    SYM = r'(\.|\?|\$|\#|\&|\,|\!|\;|\`|\~|\"|\\|\:|\+|\-|\*|\/)'
    SYM_AT = r'\@'
    SPACES = r' +'
    RE_SET = [(SYM, ' '), (SYM_AT, ' at '), (SPACES, ' ')]
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
        self.__normal_q_pattern_list = ['Who {r} {e2}',
                                        'What {r} {e2}',
                                        'Who does {e1} {r}',
                                        'What does {e1} {r}',
                                        'What is the {r} of {e2}',
                                        'Who is the {r} of {e2}',
                                        'What is {r} by {e1}',
                                        'Who is {e2}\'s {r}',
                                        'What is {e2}\'s {r}',
                                        'Who is {r} by {e1}']
        # shared by *-in, *-on
        self.__special_in_q_pattern_list = ['When did {e1} {r}',
                                            'When was {e1} {r}',
                                            'Where was {e1} {r}',
                                            'Where did {e1} {r}']
        self.__special_on_q_pattern_list = ['When did {e1} {r}',
                                            'When was {e1} {r}']

        # answer pattern
        self.__normal_a_pattern = '{e1}|{r}|{e2}'

    def __iter__(self):
        for r, e1, e2 in self.__content:
            # remove ".e", ".r" in token
            r = r.replace('.r', '')
            e1 = e1.replace('.e', '')
            e2 = e2.replace('.e', '')
            # remove \'s for relation
            r = re.sub(r'\'\w', '', r)
            r, e1, e2 = [no_symbol(w) for w in [r, e1, e2]]

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

            q = no_symbol(q)
            r, e1, e2 = [no_symbol(w) for w in [r, e1, e2]]
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
            yield w, list(map(float, emb))

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
        elif mode == 'raw_token':
            suf = 'txt'
        elif mode == 'embedding':
            # unknown token embedding is the average of low frequency words' embedding
            suf = 'emb'
        elif mode == 'structure':
            suf = 'struct'
        else:
            raise AttributeError("Mode can be only 'raw', 'embedding', 'structure'")
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
            if self.__mode == 'raw_token':
                q1_tokens, q2_tokens = [s.split() for s in [q1, q2]]
                yield q1_tokens, q2_tokens
            elif self.__mode == 'structure':
                yield q1, q2
            elif self.__mode == 'embedding':
                q1_tokens, q2_tokens = [np.asarray(list(map(float, w.split('|'))), dtype='float32')
                                        for s in [q1, q2] for w in s.split()]
                yield q1_tokens, q2_tokens

    def is_q_indx(self, i):
        # both two columns are sentences
        return i in self.sent_indx

    def get_mode(self):
        return self.__mode

    def get_q_indx(self):
        return 0

    def get_a_indx(self):
        return 1

    def __str__(self):
        return "WikiAnswer Paraphrase Questions"

    def __len__(self):
        return 300000


class ParaphraseMicrosoftRaw(object):
    def __init__(self, mode='str', grams=1):
        if mode == 'raw':
            suf = 'txt'
        elif mode == 'raw_token':
            suf = 'txt'
        elif mode == 'embedding':
            # unknown token embedding is the average of low frequency words' embedding
            suf = 'emb'
        elif mode == 'structure':
            suf = 'struct'
        else:
            raise AttributeError("Mode can be only 'raw', 'embedding', 'structure'")
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

            # to token
            if self.__mode == 'raw_token':
                s1_tokens, s2_tokens = [s.split() for s in [s1, s2]]
                yield s1_tokens, s2_tokens
            elif self.__mode == 'structure':
                yield quality, id1, id2, s1, s2
            elif self.__mode == 'embedding':
                s1_tokens, s2_tokens = [np.asarray(list(map(float, w.split('|'))), dtype='float32')
                                        for s in [s1, s2] for w in s.split()]
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
            # give raw string sentence
            suf = 'txt'
        elif mode == 'raw_token':
            # give raw tokens
            suf = 'txt'
        elif mode == 'proc_token':
            # give tokenized tokens
            suf = 'txt'
        elif mode == 'index':
            # give index value
            suf = 'indx'
        elif mode == 'embedding':
            # give embedding value
            suf = 'emb'
        elif mode == 'structure':
            # give structure sentence
            suf = 'struct'
        else:
            raise AttributeError("Mode can be only 'raw', 'raw_token', "
                                 "'proc_token', 'index', 'embedding' or 'structure'")
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
                if self.__mode == 'proc_token':
                    q = tokenize(q)
                    a = tokenize(a)

                q_tokens = q.split()
                a_tokens = a.split()

                if self.__mode in ['raw_token', 'proc_token']:
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
            voc_num = {q_indx: 14106, a_indx: 18400}
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
