# import numpy as np
# import pyximport
# pyximport.install(setup_args={"include_dirs":np.get_include()})
# import utils

# convert all sentences to their representations but keep data in other columns
from word2vec import WORD_EMBEDDING_BIN_FILE, EMBEDDING_SIZE
import numpy as np
import pickle as pkl

FEATURE_OPTS = ['unigram', 'bigram', 'thrigram', 'avg', 'holographic']


def data2feats(data, feat_select):
    '''
    :param data:
    :param feat_select: select when execute, in argument
    :return:
    '''

    if feat_select == FEATURE_OPTS[0]:
        # bag-of-word, unigram
        data.gram = 1
        feats = Ngram(data)

    elif feat_select == FEATURE_OPTS[1]:
        # bag-of-word, bigram
        data.gram = 2
        feats = Ngram(data)

    elif feat_select == FEATURE_OPTS[2]:
        # bag-of-word, thrigram
        data.gram = 3
        feats = Ngram(data)

    elif feat_select == FEATURE_OPTS[3]:
        # word embedding
        feats = WordEmbedding(data, WORD_EMBEDDING_BIN_FILE)

    elif feat_select == FEATURE_OPTS[4]:
        # holographic correlation
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
                    for w in d[i]:
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
        with open(self.embedding_dict_file, 'rb') as f:
            embedding_dict = pkl.load(f)

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
