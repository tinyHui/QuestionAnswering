# convert all sentences to their representations but keep data in other columns

import pickle as pkl


class BoW(object):
    def __init__(self, data, voc_dict):
        '''
        Represent sentence data using OneHot BoW feature
        :param data: data source, such as PPDB, QAs
        :param index: index that contains the sentence data in source data
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
        :param lstm_file: trained LSTM model
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
    def __init__(self, data, embedding_dict_file):
        '''
        Represent sentence data using word embedding trained by British National Corpus
        :param data:
        :param embedding_dict:
        '''
        assert data.mode == 'str', "must use word string in input data"
        self.data = data
        with open(embedding_dict_file, 'rb') as f:
            self.embedding_dict = pkl.load(f)

    def __iter__(self):
        for d in self.data:
            new_d = [None] * self.data.param_num
            for i in range(self.data.param_num):
                if i in self.data.set_indx:
                    # for each token, find its embedding
                    new_d[i] = [self.embedding_dict[w] for w in d[i]]
                else:
                    new_d[i] = d[i]

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

