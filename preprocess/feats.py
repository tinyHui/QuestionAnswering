# convert all sentences to their representations but keep data in other columns
from preprocess.data import ReVerbPairs, ParaphraseQuestionRaw
from word2vec import EMBEDDING_SIZE
import numpy as np
from preprocess.utils import Utils

FEATURE_OPTS = ['unigram', 'bigram', 'thrigram', 'avg', 'holographic']


def feats_loader(feat_select, usage, train_two_stage_cca=False):
    '''
    :param data:
    :param feat_select: select when execute, in argument
    :return:
    '''

    if feat_select == FEATURE_OPTS[0]:
        # bag-of-word, unigram
        if not train_two_stage_cca:
            data = ReVerbPairs(usage=usage, mode='index', gram=1)
        else:
            data = ParaphraseQuestionRaw(mode='index')
        feats = Ngram(data)

    elif feat_select == FEATURE_OPTS[1]:
        # bag-of-word, bigram
        if not train_two_stage_cca:
            data = ReVerbPairs(usage=usage, mode='index', gram=2)
        else:
            data = ParaphraseQuestionRaw(mode='index')
        feats = Ngram(data)

    elif feat_select == FEATURE_OPTS[2]:
        # bag-of-word, thrigram
        if not train_two_stage_cca:
            data = ReVerbPairs(usage=usage, mode='index', gram=3)
        else:
            data = ParaphraseQuestionRaw(mode='index')
        feats = Ngram(data)

    elif feat_select == FEATURE_OPTS[3]:
        # word embedding
        data = ReVerbPairs(usage=usage, mode='embedding')
        feats = WordEmbedding(data)

    elif feat_select == FEATURE_OPTS[4]:
        # holographic correlation
        data = ReVerbPairs(usage=usage, mode='embedding')
        feats = Holographic(data)

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
            yield d, self.__generate__

    def __generate__(self, d):
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
            return d, feat

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
    def __init__(self, data):
        '''
        Represent sentence data using word embedding trained by British National Corpus
        :param data: data source, such as PPDB, QAs
        :param embedding_dict_file: word-embedding dictionary file name
        '''
        assert data.mode == 'embedding', "must use word embedding in input data"
        self.data = data
        self.utils = Utils()

    def __iter__(self):
        for d in self.data:
            yield d, self.__generate__

    def __generate__(self, d):
            param_num = len(d)
            feat = [None] * param_num
            for i in range(param_num):
                if i in self.data.sent_indx:
                    feat[i] = self.utils.avg_emb(d[i], EMBEDDING_SIZE)
                else:
                    feat[i] = d[i]

            return d, feat

    def __len__(self):
        return len(self.data)


class Holographic(object):
    def __init__(self, data):
        '''
        Represent sentence data using word embedding trained by British National Corpus
        :param data: data source, such as PPDB, QAs
        :param embedding_dict_file: word-embedding dictionary file name
        '''
        assert data.mode == 'embedding', "must use word index in input data"
        self.data = data
        self.utils = Utils()

    def __iter__(self):
        for d in self.data:
            yield d, self.__generate__

    def __generate__(self, d):
            param_num = len(d)
            feat = [None] * param_num
            for i in range(param_num):
                if i in self.data.sent_indx:
                    feat[i] = self.utils.cc(d[i], EMBEDDING_SIZE)
                else:
                    feat[i] = d[i]

            return d, feat

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
