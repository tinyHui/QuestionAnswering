# convert all sentences to their representations but keep data in other columns
from preprocess.data import get_struct, ReVerbPairs, ParaphraseWikiAnswer
from word2vec import EMBEDDING_SIZE
from preprocess.utils import Utils
import numpy as np
import requests
import json

FEATURE_OPTS = ['unigram', 'bigram', 'thrigram', 'avg', 'holographic']


def feats_loader(feat_select, usage, train_two_stage_cca=False):
    '''
    :param data:
    :param feat_select: select when execute, in argument
    :return:
    '''

    if feat_select == FEATURE_OPTS[0]:
        # bag-of-word, unigram
        data = ReVerbPairs(usage=usage, mode='index', grams=1)
        q_indx = data.get_q_indx()
        a_indx = data.get_a_indx()
        feats = Ngram(data)

    elif feat_select == FEATURE_OPTS[1]:
        # bag-of-word, bigram
        data = ReVerbPairs(usage=usage, mode='index', grams=2)
        q_indx = data.get_q_indx()
        a_indx = data.get_a_indx()
        feats = Ngram(data)

    elif feat_select == FEATURE_OPTS[2]:
        # bag-of-word, thrigram
        data = ReVerbPairs(usage=usage, mode='index', grams=3)
        q_indx = data.get_q_indx()
        a_indx = data.get_a_indx()
        feats = Ngram(data)

    elif feat_select == FEATURE_OPTS[3]:
        # word embedding
        if not train_two_stage_cca:
            data = ReVerbPairs(usage=usage, mode='embedding')
        else:
            data = ParaphraseWikiAnswer(mode='embedding')
        q_indx = data.get_q_indx()
        a_indx = data.get_a_indx()
        feats = AvgEmbedding(data)

    elif feat_select == FEATURE_OPTS[4]:
        # holographic correlation
        if not train_two_stage_cca:
            data_emb = ReVerbPairs(usage=usage, mode='embedding')
            data_struct = ReVerbPairs(usage=usage, mode='structure')
        else:
            data_emb = ParaphraseWikiAnswer(mode='embedding')
            data_struct = ParaphraseWikiAnswer(mode='structure')
        q_indx = data_emb.get_q_indx()
        a_indx = data_emb.get_a_indx()
        feats = Holographic(data_emb=data_emb, data_struct=data_struct)

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

    return q_indx, a_indx, feats


def get_parse_tree(sentence, job_id):
    URL = "http://localhost:8080/jsonrpc"
    HEADERS = {'content-type': 'application/json'}

    payload = {
        "method": "parse",
        "params": [sentence],
        "jsonrpc": "2.0",
        "id": job_id,
    }
    response = requests.post(URL, data=json.dumps(payload), headers=HEADERS).json()
    return response['result']['parsetree']


def get_lemmas(sentence, job_id):
    URL = "http://localhost:8080/jsonrpc"
    HEADERS = {'content-type': 'application/json'}

    payload = {
        "method": "lemma",
        "params": [sentence],
        "jsonrpc": "2.0",
        "id": job_id,
    }
    response = requests.post(URL, data=json.dumps(payload), headers=HEADERS).json()
    return response['result']['lemma']


class Ngram(object):
    def __init__(self, data):
        '''
        Represent sentence data using OneHot BoW feature
        :param data: data source
        :param voc_dict: word-index dictionary
        :param gram: default is unigram
        '''
        # assert data.get_mode() == 'index', "must use word index in input data"
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
            return feat

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


class AvgEmbedding(object):
    def __init__(self, data):
        '''
        Represent sentence data using word embedding trained by British National Corpus
        :param data: data source
        :param embedding_dict_file: word-embedding dictionary file name
        '''
        # assert data.get_mode() == 'embedding', "must use word embedding in input data"
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

            return feat

    def __len__(self):
        return len(self.data)


class Holographic(object):
    def __init__(self, data_emb, data_struct):
        '''
        Represent sentence data using word embedding trained by British National Corpus
        :param data: data source, include raw string and embedding vector
        :param embedding_dict_file: word-embedding dictionary file name
        '''
        # assert data[0].get_mode() == 'str', "must use word embedding in input data"
        # assert data[1].get_mode() == 'embedding', "must use word embedding in input data"
        self.data_emb = data_emb
        self.data_struct = data_struct
        self.utils = Utils()

    def __iter__(self):
        for d_emb, d_struct in zip(self.data_emb, self.data_struct):
            yield (d_emb, d_struct), self.__generate__

    def __generate__(self, d):
            d_emb, d_struct = d
            # assert len(d_emb) == len(d_struct)
            param_num = len(d_emb)
            feat = [None] * param_num
            for i in range(param_num):
                if i in self.data_emb.sent_indx:
                    struct = get_struct(self.data_emb.is_q_indx(i), d_emb[i], d_struct[i])
                    feat[i] = self.utils.cc(struct, EMBEDDING_SIZE)
                else:
                    feat[i] = d_emb[i]

            return feat

    def __len__(self):
        return len(self.data_emb)


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
