# convert all sentences to their representations but keep data in other columns
from calendar import month_name, month_abbr
from preprocess.data import ReVerbPairs
from word2vec import WORD_EMBEDDING_BIN_FILE, EMBEDDING_SIZE
from preprocess import utils
import numpy as np
import pickle as pkl
import re

FEATURE_OPTS = ['unigram', 'bigram', 'thrigram', 'avg', 'holographic']


def process_raw(raw):
    # to lower case
    s = raw.lower()
    # replace month name to number
    MONTH_NAME = zip([name.lower() for name in month_name[1:]], [name.lower() for name in month_abbr[1:]])
    for i, (name, abbr) in enumerate(MONTH_NAME):
        s = re.sub(r'\b{}\b|\b{}\b'.format(name, abbr), '%02d' % (i + 1), s)

    # define replace pattern
    GRAMMAR_SYM = r'(\')'
    DATE = r'(([0]?[1-9]|[1][0-2])[\.\/\- ]([0]?[1-9]|[1|2][0-9]|[3][0|1])[\.\/\- ]([0-9]{4}|[0-9]{2}))|' \
           r'(([0]?[1-9]|[1|2][0-9]|[3][0|1])[\.\/\- ]([0]?[1-9]|[1][0-2])[\.\/\- ]([0-9]{4}|[0-9]{2}))'
    TIME = r'[0-2]?[1-9]:[0-5][0-9][ \-]?(am|pm)?'
    MONEY = r'\$[ \-]?\d+(\,\d+)?\.?\d+'
    PRESENT = r'[-+]?\d+(\,\d+)?(\.\d+)?[ \-]?\%'
    NUMBER = r'[-+]?\d+(\,\d+)?(\.\d+)?'
    EMAIL = r'[_a-z0-9-]+(\.[_a-z0-9-]+)*@[a-z0-9-]+' \
            r'(\.[a-z0-9-]+)*\.(([0-9]{1,3})|([a-z]{2,3})|(aero|coop|info|museum|name))'
    SYM = r'(\.|\?|\$|\*|\#|\&)'
    SPACES = r' +'
    # replace all matched phrase to TOKEN name
    RE_SET = [(GRAMMAR_SYM, ' \\1'), (DATE, 'DATE'), (TIME, 'TIME'), (MONEY, 'MONEY'), (PRESENT, 'PRESENT'),
              (NUMBER, 'NUM'), (EMAIL, 'EMAIL'), (SYM, ' \\1 '), (SPACES, ' ')]
    for p, t in RE_SET:
        s = re.sub(p, t, s)
    return s


def feats_loader(feat_select, usage, part=None):
    '''
    :param data:
    :param feat_select: select when execute, in argument
    :return:
    '''

    if feat_select == FEATURE_OPTS[0]:
        # bag-of-word, unigram
        data = ReVerbPairs(usage=usage, part=part, mode='index', gram=1)
        feats = Ngram(data)

    elif feat_select == FEATURE_OPTS[1]:
        # bag-of-word, bigram
        data = ReVerbPairs(usage=usage, part=part, mode='index', gram=2)
        feats = Ngram(data)

    elif feat_select == FEATURE_OPTS[2]:
        # bag-of-word, thrigram
        data = ReVerbPairs(usage=usage, part=part, mode='index', gram=3)
        feats = Ngram(data)

    elif feat_select == FEATURE_OPTS[3]:
        # word embedding
        data = ReVerbPairs(usage=usage, part=part, mode='embedding')
        feats = WordEmbedding(data, WORD_EMBEDDING_BIN_FILE)

    elif feat_select == FEATURE_OPTS[4]:
        # holographic correlation
        data = ReVerbPairs(usage=usage, part=part, mode='embedding')
        feats = Holographic(data, WORD_EMBEDDING_BIN_FILE)

    # elif feat_select == FEATURE_OPTS[]:
    #     # word embedding
    #     feats = AutoEncoder(data)

    # elif feat_select == FEATURE_OPTS[]:
    #     # sentence embedding by paraphrased sentences
    #     if not os.path.exists(LSTM_FILE):
    #         train_lstm(
    #             max_epochs=100,
    #             test_size=2,
    #             saveto=LSTM_FILE,
    #             reload_model=True
    #         )
    #     feats = LSTM(data, lstm_file=LSTM_FILE, voc_dict=voc_dict)

    else:
        raise SystemError("%s is not an available feature" % feat_select)

    return feats


class Ngram(object):
    def __init__(self, data):
        '''
        Represent sentence data using OneHot BoW feature
        :param data: data source, such as PPDB, QAs
        :param voc_dict: word-index dictionary
        :param gram: default is unigram
        '''
        assert data.mode == 'index', "must use word index in input data"
        self.data = data

    def __iter__(self):
        for d in self.data:
            param_num = len(d)
            feat = [None] * param_num
            for i in range(param_num):
                if i in self.data.sent_indx:
                    # convert sentence to One-Hot representation
                    feat[i] = np.zeros(self.data.get_voc_num(i), dtype='float32')
                    s = ' '.join(d[i])
                    s = process_raw(s)
                    for w in s.split(' '):
                        # just accumulate on the position of word index
                        feat[i][w] += 1
                else:
                    # use original data
                    feat[i] = d[i]
            yield d, feat

    def __len__(self):
        return len(self.data)


# class AutoEncoder(object):
#     def __init__(self, data):
#         assert data.mode == 'index', "must use word index in input data"
#         logging.info("loading bigram dictionary")
#         with open(THRIGRAM_DICT_FILE, 'rb') as f:
#             voc_dict = pkl.load(f)
#         self.thrigram = Ngram(data, voc_dict, gram=3)
#         with open(AE_MODEL_FILE, 'rb') as f:
#             self.model = pkl.load(f)
#
#     def __iter__(self):
#         for _, feat in self.thrigram:
#             yield self.model.predict(feat)


class WordEmbedding(object):
    def __init__(self, data, embedding_dict_file):
        '''
        Represent sentence data using word embedding trained by British National Corpus
        :param data: data source, such as PPDB, QAs
        :param embedding_dict_file: word-embedding dictionary file name
        '''
        assert data.mode == 'index', "must use word index in input data"
        self.data = data
        self.embedding_dict_file = embedding_dict_file

    def __iter__(self):
        with open(self.embedding_dict_file, 'rb') as f:
            embedding_dict = pkl.load(f)

        for d in self.data:
            param_num = len(d)
            feat = [None] * param_num
            for i in range(param_num):
                if i in self.data.sent_indx:
                    feat[i] = np.zeros(EMBEDDING_SIZE, dtype='float32')
                    feat[i] = utils.avg_emb(list(d[i]), embedding_dict[i])
                else:
                    feat[i] = d[i]

            yield d, feat

    def __len__(self):
        return len(self.data)


class Holographic(object):
    def __init__(self, data, embedding_dict_file):
        '''
        Represent sentence data using word embedding trained by British National Corpus
        :param data: data source, such as PPDB, QAs
        :param embedding_dict_file: word-embedding dictionary file name
        '''
        assert data.mode == 'index', "must use word index in input data"
        self.data = data
        self.embedding_dict_file = embedding_dict_file

    def __iter__(self):
        for d in self.data:
            param_num = len(d)
            feat = [None] * param_num
            for i in range(param_num):
                if i in self.data.sent_indx:
                    feat[i] = utils.cc(list(d[i]), embedding_dict[i], EMBEDDING_SIZE)
                else:
                    feat[i] = d[i]

            yield d, feat

    def __len__(self):
        return len(self.data)


# class LSTM(object):
#     def __init__(self, data, lstm_file, voc_dict):
#         '''
#         Represent sentence data using LSTM sentence embedding
#         :param data: data source, such as PPDB, QAs
#         :param lstm_file: trained LSTM model file name
#         :param voc_dict: word-index dictionary
#         '''
#         assert data.mode == 'index', "must use word index in input data"
#         self.data = data
#         with open(lstm_file, 'rb') as f:
#             self.model = pkl.load(f)
#         self.voc_dict = voc_dict
#
#     def __iter__(self):
#         for d in self.data:
#             param_num = len(d)
#             feat = [None] * param_num
#             for i in range(param_num):
#                 if i in self.data.sent_indx:
#                     # TODO: represent sentence use LSTM, d[i] is a sentence
#                     feat[i] = d[i]
#                 else:
#                     # use original data
#                     feat[i] = d[i]
#             yield d, feat
#
#     def __len__(self):
#         return len(self.data)
