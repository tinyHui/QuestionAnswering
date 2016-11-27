from calendar import month_name, month_abbr
from collections import UserDict
from random import sample
from word2vec import WORD_EMBEDDING_FILE
from nltk.tree import Tree
import sqlite3
import re
import os
import logging


UNKNOWN_TOKEN = 'UNKNOWN'
UNKNOWN_TOKEN_INDX = 0


def tokenize(raw):
    if raw.upper() == raw:
        return raw

    # to lower case
    s = raw.lower()

    # replace month name to number
    MONTH_NAME = zip([name.lower() for name in month_name[1:]], [name.lower() for name in month_abbr[1:]])
    for i, (name, abbr) in enumerate(MONTH_NAME):
        s = re.sub(r'\b{}\b|\b{}\b'.format(name, abbr), '%02d' % (i + 1), s)

    # define replace pattern
    DATE = r'([0-9]{1,2})?[\.\/\- ][0-9]{1,2}(st|nd|rd|th)?[\.\/\- ][0-9]{4}|' \
           r'[0-9]{4}[\.\/\- ][0-9]{1,2}(st|nd|rd|th)?[\.\/\- ]([0-9]{1,2})?|' \
           '[0-9]{1,2}(st|nd|rd|th)?[\/\- ][0-9]{1,2}|' \
           '[0-9]{1,2}[\/\- ][0-9]{1,2}(st|nd|rd|th)?'
    YEAR = r'(19|20)\d{2}'
    TIME = r'[0-2]?[0-9]:[0-5][0-9][ \-]?(am|pm)?'
    MONEY = r'\$[ \-]?\d+(\,\d+)?\.?\d+'
    PRESENT = r'[-+]?\d+(\,\d+)?(\.\d+)?[ \-]?\%'
    METRIC = r'\d+ ?(ml|cc|l)'
    NUMBER = r'[-+]?\d+(\,\d+)?(\.\d+)?( \d+)?(stz|nd|rd|th)?'
    EMAIL = r'[_a-z0-9-]+(\.[_a-z0-9-]+)*@[a-z0-9-]+' \
            r'(\.[a-z0-9-]+)*\.(([0-9]{1,3})|([a-z]{2,3})|(aero|coop|info|museum|name))'
    SPACES = r' +'
    # replace all matched phrase to TOKEN name
    RE_SET = [(DATE, ' DATE '), (YEAR, ' DATE '), (TIME, ' TIME '), (MONEY, ' MONEY '),
              (PRESENT, ' PRESENT '), (METRIC, ' MTC '), (NUMBER, ' NUM '), (EMAIL, ' EMAIL '), (SPACES, ' '), (SPACES, ' ')]
    for p, t in RE_SET:
        s = re.sub(p, t, s)
    s = s.strip()
    return s


def no_symbol(s):
    GRAMMAR_SYM = r'(\')'
    SYM = r'(\.|\?|\$|\#|\&|\,|\!|\;|\`|\~|\"|\\|\:|\+|\-|\*|\/)'
    SYM_AT = r'\@'
    SPACES = r' +'
    START_QUOTE = r'^[\' ]+'
    END_QUOTE = r'[\' ]+$'
    RE_SET = [(GRAMMAR_SYM, ' \\1'), (SYM, ' '), (SYM_AT, ' at '), (SPACES, ' '), (START_QUOTE, ''), (END_QUOTE, '')]
    for p, t in RE_SET:
        s = re.sub(p, t, s)
    s = s.strip()
    return s


def word2index(w, hash_map):
    try:
        return hash_map[w]
    except KeyError:
        # unseen token
        return UNKNOWN_TOKEN_INDX


def word2hash(w, hash_map):
    try:
        return hash_map[w]
    except KeyError:
        # for unseen words, the embedding is the average \in R^Embedding_size
        return hash_map[UNKNOWN_TOKEN]


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
        # self.__content = c.execute("""
        #     SELECT  *
        #         FROM    (   SELECT  *
        #                     FROM    tuples
        #                     WHERE   rel not like '%-in.r' and rel not like '%-on.r'
        #                     LIMIT 30000
        #                 )
        #         UNION
        #         SELECT  *
        #         FROM    (   SELECT  *
        #                     FROM    tuples
        #                     WHERE   rel like '%-in.r' or rel like '%-on.r'
        #                     LIMIT 60000
        #                 );
        # """)
        self.__content = c.execute("""
            SELECT  *
            FROM    tuples
        """)
        # define the pattern
        self.__normal_q_pattern_list = ['Who {r} {e2}',
                                        'What {r} {e2}',
                                        'Who does {e1} {r}',
                                        'What does {e1} {r}',
                                        'What is the {r} of {e2}',
                                        'Who is the {r} of {e2}',
                                        'What is {r} by {e1}',
                                        'Who is {e2} \'s {r}',
                                        'What is {e2} \'s {r}',
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
            # if r.endswith('-in'):
            #     q_pattern_list = sample(self.__special_in_q_pattern_list, 1)
            # elif r.endswith('-on'):
            #     q_pattern_list = sample(self.__special_on_q_pattern_list, 1)
            # else:
            #     q_pattern_list = sample(self.__normal_q_pattern_list, 3)
            # choose full pattern
            if r.endswith('-in'):
                q_pattern_list = self.__special_in_q_pattern_list
            elif r.endswith('-on'):
                q_pattern_list = self.__special_on_q_pattern_list
            else:
                q_pattern_list = self.__normal_q_pattern_list
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


# Gigaword
class GigawordRaw(object):
    def __init__(self):
        self.__folder = '/disk/ocean/public/corpora/english_gigaword_segmented/5.0/data/afp_eng/'

    def __iter__(self):
        for fname in os.listdir(self.__folder):
            fname_abs = os.path.join(self.__folder, fname)
            with open(fname_abs, 'r') as f:
                for line in f:
                    content = line.strip()
                    content = no_symbol(content)
                    tokens = content.split()
                    yield tokens

    def __len__(self):
        return 27487013


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
    def __init__(self, mode='raw_token'):
        if mode not in ['raw', 'raw_token', 'proc_token', 'embedding', 'structure']:
            raise AttributeError("Mode can be only 'raw', 'raw_token', 'proc_token', 'embedding', 'structure'")

        if mode == 'embedding':
            import pickle as pkl
            logging.info("loading embedding hash")
            from word2vec import WORD_EMBEDDING_BIN_FILE
            with open(WORD_EMBEDDING_BIN_FILE, 'rb') as f:
                self.voc_dict = pkl.load(f)
        elif mode == 'structure':
            raise NotImplementedError()

        self.__mode = mode
        self.__file = './data/paraphrases.wikianswer.txt'
        # index of return data contains sentence
        self.sent_indx = (0, 1)

    def __iter__(self):
        for line in open(self.__file, 'r'):
            q1, q2 = line.strip().split('\t')
            q1 = no_symbol(q1)
            q2 = no_symbol(q2)

            if self.__mode == 'raw':
                yield q1, q2
                continue

            if self.__mode == 'proc_token':
                q1 = tokenize(q1)
                q2 = tokenize(q2)

            q1_tokens, q2_tokens = [s.split() for s in [q1, q2]]

            # to token
            if self.__mode in ['raw_token', 'proc_token']:
                yield q1_tokens, q2_tokens
            elif self.__mode == 'embedding':
                q1_tokens = [word2hash(token, self.voc_dict) for token in q1_tokens]
                q2_tokens = [word2hash(token, self.voc_dict) for token in q2_tokens]
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
        # full WikiAnswer Paraphrase Questions
        return 13710104

        # filter
        # return 41

        # main train
        # return 300000


# q-a pairs generated by reverb
class ReVerbPairs(object):
    def __init__(self, usage='train', mode='raw', grams=1):
        if mode not in ['raw', 'raw_token', 'proc_token', 'index', 'embedding', 'structure']:
            raise AttributeError("Mode can be only 'raw', 'raw_token', "
                                 "'proc_token', 'index', 'embedding' or 'structure'")

        import pickle as pkl
        if mode == 'index':
            logging.info("loading index hash")
            # give index value
            if grams == 1:
                from hash_index import UNIGRAM_DICT_FILE
                with open(UNIGRAM_DICT_FILE % "qa", 'rb') as f:
                    self.voc_dict = pkl.load(f)
            elif grams == 2:
                from hash_index import BIGRAM_DICT_FILE
                with open(BIGRAM_DICT_FILE % "qa", 'rb') as f:
                    self.voc_dict = pkl.load(f)
            elif grams == 3:
                from hash_index import TRIGRAM_DICT_FILE
                with open(TRIGRAM_DICT_FILE % "qa", 'rb') as f:
                    self.voc_dict = pkl.load(f)
            raise SystemError("ReVerbPairs only accept uni/bi/tri-grams")
        elif mode == 'embedding':
            logging.info("loading embedding hash")
            from word2vec import WORD_EMBEDDING_BIN_FILE
            with open(WORD_EMBEDDING_BIN_FILE, 'rb') as f:
                self.voc_dict = pkl.load(f)
        elif mode == 'structure':
            raise NotImplementedError()

        self.__mode = mode

        if usage in ['train', 'test']:
            self.__file = './data/reverb-%s.txt' % usage
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
                try:
                    q, a = line.strip().split('\t')
                except ValueError:
                    pass
            else:
                # test
                q_id, q, a, l = line.strip().split('\t')
                q_id = int(q_id)
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

                e1_tokens, r_tokens, e2_tokens = [t.strip().split() for t in a.split('|')]
                a_tokens = e1_tokens + r_tokens + e2_tokens

                if self.__mode in ['raw_token', 'proc_token']:
                    if self.__grams > 1:
                        q_tokens = [" ".join(ws) for ws in zip(*[q_tokens[j:] for j in range(self.__grams)])]
                elif self.__mode == 'index':
                    q_i = self.sent_indx[0]
                    a_i = self.sent_indx[1]
                    q_tokens = [int(word2index(token, self.voc_dict[q_i])) for token in q_tokens]
                    a_tokens = [int(word2index(token, self.voc_dict[a_i])) for token in a_tokens]
                elif self.__mode == 'embedding':
                    q_tokens = [word2hash(token, self.voc_dict) for token in q_tokens]
                    a_tokens = [word2hash(token, self.voc_dict) for token in a_tokens]
                # produce the token per line
                if self.__usage == 'train':
                    yield (q_tokens, a_tokens)
                else:
                    # test
                    yield (q_id, q_tokens, a_tokens, l)

    def is_q_indx(self, i):
        if self.__usage == 'train':
            return i == 0
        elif self.__usage == 'test':
            return i == 1

    def get_voc_num(self, i):
        if self.__usage == 'train':
            q_indx = 0
            a_indx = 1
        else:
            # self.__usage == 'test':
            q_indx = 1
            a_indx = 2

        if self.__grams == 1:
            # main train
            # voc_num = {q_indx: 14052, a_indx: 18400}
            # prove performance of uni-/bi-/tri-grams
            voc_num = {q_indx: 30027, a_indx: 44267}
        elif self.__grams == 2:
            # main train
            # voc_num = {q_indx: 40820, a_indx: 18400}
            # prove performance of uni-/bi-/tri-grams
            voc_num = {q_indx: 126754, a_indx: 44267}
        else:
            # self.__grams == 3:
            # main train
            # voc_num = {q_indx: 45020, a_indx: 18400}
            # prove performance of uni-/bi-/tri-grams
            voc_num = {q_indx: 162646, a_indx: 44267}
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
            # main train
            # return 269979

            # 3 patterns, use all triples
            # return 43133211

            # filtered
            # return 41592028

            # full patterns
            return 143777370

            # prove performance of uni-/bi-/tri-grams
            # return 1000000

        elif self.__usage == 'test':
            return 48910

    def __str__(self):
        return "ReVerb QA pairs"

