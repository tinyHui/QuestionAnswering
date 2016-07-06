# convert all sentences to their representations but keep data in other columns

from word2embedding import WORD_EMBEDDING_FILE, EMBEDDING_SIZE
from word2index import UNIGRAM_DICT_FILE, BIGRAM_DICT_FILE, THRIGRAM_DICT_FILE
import numpy as np
import pickle as pkl
import logging

FEATURE_OPTS = ['unigram', 'bigram', 'thrigram', 'we']


def data2feats(data, feat_select):
    '''
    :param data:
    :param feat_select: select when execute, in argument
    :return:
    '''

    if feat_select == FEATURE_OPTS[0]:
        # bag-of-word, unigram
        logging.info("loading unigram dictionary")
        with open(UNIGRAM_DICT_FILE, 'rb') as f:
            voc_dict = pkl.load(f)
        feats = Ngram(data, gram=1, voc_dict=voc_dict)

    elif feat_select == FEATURE_OPTS[1]:
        # bag-of-word, bigram
        logging.info("loading bigram dictionary")
        with open(BIGRAM_DICT_FILE, 'rb') as f:
            voc_dict = pkl.load(f)
        feats = Ngram(data, gram=2, voc_dict=voc_dict)

    elif feat_select == FEATURE_OPTS[2]:
        # bag-of-word, thrigram
        logging.info("loading thrigram dictionary")
        with open(THRIGRAM_DICT_FILE, 'rb') as f:
            voc_dict = pkl.load(f)
        feats = Ngram(data, gram=3, voc_dict=voc_dict)

    elif feat_select == FEATURE_OPTS[3]:
        # word embedding
        feats = WordEmbedding(data, WORD_EMBEDDING_FILE)

    # elif feat_select == FEATURE_OPTS[4]:
    #     # word embedding
    #     feats = AutoEncoder(data)

    # elif feat_select == FEATURE_OPTS[2]:
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
    def __init__(self, data, voc_dict, gram=1):
        '''
        Represent sentence data using OneHot BoW feature
        :param data: data source, such as PPDB, QAs
        :param voc_dict: word-index dictionary
        :param gram: default is unigram
        '''
        assert data.mode == 'index', "must use word index in input data"
        self.data = data
        self.voc_dict = voc_dict
        self.gram = gram

    def __iter__(self):
        voc_num = [len(self.voc_dict[i].keys()) for i in self.voc_dict.keys()]
        for d in self.data:
            param_num = len(d)
            feat = [None] * param_num
            for i in range(param_num):
                if i in self.data.sent_indx:
                    # convert sentence to One-Hot representation
                    feat[i] = np.zeros(voc_num[i], dtype='float64')
                    if self.gram == 1:
                        for w in d[i]:
                            # unigram, just accumulate on the position of word index
                            feat[i][w] += 1
                    else:
                        for w in zip(*[d[i][j:] for j in range(self.gram)]):
                            index = self.voc_dict[w]
                            feat[i][index] += 1
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
        with open(embedding_dict_file, 'rb') as f:
            self.embedding_dict = pkl.load(f)

    def __iter__(self):
        for d in self.data:
            param_num = len(d)
            feat = [None] * param_num
            for i in range(param_num):
                if i in self.data.sent_indx:
                    feat[i] = np.zeros(EMBEDDING_SIZE, dtype='float64')
                    for w in d[i]:
                        # for each token, find its embedding
                        # unseen token will automatically take 0 x R^300
                        feat[i] += self.embedding_dict[i][w]
                    # calculate the average of sum of embedding of all words
                    feat[i] /= len(d[i])

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
