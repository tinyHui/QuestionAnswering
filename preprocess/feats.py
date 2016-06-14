# convert all sentences to their representations but keep data in other columns

from siamese_cosine import LSTM_FILE, train_lstm
from text2embedding import WORD_EMBEDDING_FILE
from text2index import VOC_DICT_FILE
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
        data.mode = 'str'
        # word embedding
        feats = WordEmbedding(data, data.max_length, WORD_EMBEDDING_FILE)

    else:
        raise IndexError("%s is not an available feature" % feature)

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
        voc_num = len(self.voc_dict.keys())
        for d in self.data:
            new_d = [None] * self.data.param_num
            for i in range(self.data.param_num):
                if i in self.data.sent_indx:
                    # convert sentence to One-Hot representation
                    new_d[i] = [0] * voc_num
                    for w in d[i]:
                        # one hot
                        new_d[i][w] += 1
                else:
                    # use original data
                    new_d[i] = d[i]
            yield new_d

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
            new_d = [None] * self.data.param_num
            for i in range(self.data.param_num):
                if i in self.data.sent_indx:
                    # TODO: represent sentence use LSTM, d[i] is a sentence
                    new_d[i] = d[i]

            yield new_d

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
        assert data.mode == 'str', "must use word string in input data"
        self.data = data
        with open(embedding_dict_file, 'rb') as f:
            self.embedding_dict = pkl.load(f)
        self.max_length = max_length

    def __iter__(self):
        for d in self.data:
            new_d = [None] * self.data.param_num
            for i in range(self.data.param_num):
                if i in self.data.sent_indx:
                    new_d[i] = [0] * 300 * self.max_length[i]
                    for j, w in enumerate(d[i]):
                        # for each token, find its embedding
                        try:
                            new_d[i][j*300:(j+1)*300] = self.embedding_dict[w]
                        except TypeError:
                            continue

                else:
                    new_d[i] = d[i]

            yield new_d

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

