# convert all sentences to their representations but keep data in other columns

from preprocess.data import UNKNOWN_TOKEN
from siamese_cosine import LSTM_FILE, train_lstm
from word2embedding import WORD_EMBEDDING_FILE
from word2index import VOC_DICT_FILE
import numpy as np
import pickle as pkl
import os

FEATURE_OPTS = ['bow', 'lstm', 'we']


def data2feats(data, feat_select):
    with open(VOC_DICT_FILE, 'rb') as f:
        voc_dict = pkl.load(f)

    if feat_select == FEATURE_OPTS[0]:
        # bag-of-word
        feats = BoW(data, voc_dict=voc_dict)

    elif feat_select == FEATURE_OPTS[1]:
        # sentence embedding by paraphrased sentences
        if not os.path.exists(LSTM_FILE):
            train_lstm(
                max_epochs=100,
                test_size=2,
                saveto=LSTM_FILE,
                reload_model=True
            )
        feats = LSTM(data, lstm_file=LSTM_FILE, voc_dict=voc_dict)

    elif feat_select == FEATURE_OPTS[2]:
        # word embedding
        feats = WordEmbedding(data, data.max_length, WORD_EMBEDDING_FILE)

    else:
        raise IndexError("%s is not an available feature" % feat_select)

    return feats


class BoW(object):
    def __init__(self, data, voc_dict):
        '''
        Represent sentence data using OneHot BoW feature
        :param data: data source, such as PPDB, QAs
        :param voc_dict: word-index dictionary
        '''
        assert data.mode == 'index', "must use word index in input data"
        self.data = data
        self.voc_dict = voc_dict

    def __iter__(self):
        voc_num = [len(self.voc_dict[k].keys()) for k in self.voc_dict.keys()]
        for d in self.data:
            param_num = len(d)
            feat = [None] * param_num
            for i in range(param_num):
                if i in self.data.sent_indx:
                    # convert sentence to One-Hot representation
                    feat[i] = [0] * voc_num[i]
                    for w in d[i]:
                        if w == UNKNOWN_TOKEN:
                            # deal with unseen token, pass
                            continue
                            # one hot
                        feat[i][w] += 1
                else:
                    # use original data
                    feat[i] = d[i]
            yield d, feat

    def __len__(self):
        return len(self.data)


class LSTM(object):
    def __init__(self, data, lstm_file, voc_dict):
        '''
        Represent sentence data using LSTM sentence embedding
        :param data: data source, such as PPDB, QAs
        :param lstm_file: trained LSTM model file name
        :param voc_dict: word-index dictionary
        '''
        assert data.mode == 'index', "must use word index in input data"
        self.data = data
        with open(lstm_file, 'rb') as f:
            self.model = pkl.load(f)
        self.voc_dict = voc_dict

    def __iter__(self):
        for d in self.data:
            param_num = len(d)
            feat = [None] * param_num
            for i in range(param_num):
                if i in self.data.sent_indx:
                    # TODO: represent sentence use LSTM, d[i] is a sentence
                    feat[i] = d[i]
                else:
                    # use original data
                    feat[i] = d[i]
            yield d, feat

    def __len__(self):
        return len(self.data)


class WordEmbedding(object):
    def __init__(self, data, max_length, embedding_dict_file):
        '''
        Represent sentence data using word embedding trained by British National Corpus
        :param data: data source, such as PPDB, QAs
        :param max_length: max legnth of the sentence for correlated column
        :param embedding_dict_file: word-embedding dictionary file name
        '''
        assert data.mode == 'index', "must use word index in input data"
        self.data = data
        with open(embedding_dict_file, 'rb') as f:
            self.embedding_dict = pkl.load(f)
        self.max_length = max_length

    def __iter__(self):
        for d in self.data:
            param_num = len(d)
            feat = [None] * param_num
            for i in range(param_num):
                if i in self.data.sent_indx:
                    feat[i] = np.zeros(300, dtype='float64')
                    for w in d[i]:
                        # for each token, find its embedding
                        try:
                            feat[i] += np.asarray(self.embedding_dict[w])
                        except TypeError:
                            continue
                    # calculate the average of sum of embedding of all words
                    feat[i] /= len(d[i])

                else:
                    feat[i] = d[i]

            yield d, feat

    def __len__(self):
        return len(self.data)


# self.freq_dict = defaultdict(int)
# try:
#     with open(FREQ_DICT, "rb") as f:
#         self.freq_dict = pkl.load(f)
# except FileNotFoundError:
#     for l in data:
#         for i in index:
#             for w in l[i]:
#                 self.freq_dict[w] += 1
#     with open(FREQ_DICT, "wb") as f:
#         pkl.dump(self.freq_dict, f)
