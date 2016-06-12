import re
import logging


# paraphrased sentences
class PPDB(object):
    def __init__(self, usage='train', mode='str', voc_dict=None):
        if usage in ['train', 'test']:
            self.file = './data/msr_paraphrase_%s.txt' % usage
            self.usage = usage
        else:
            raise SystemError("usage can be only train/test")
        self.voc_dict = voc_dict
        if mode == 'index':
            assert voc_dict is not None, "must take vocabulary-index dictionary."
        self.mode = mode
        # column number for each iteration
        self.param_num = 3
        # index of return data contains sentence
        self.sent_indx = (0, 1)

    def __iter__(self):
        for line in open(self.file, 'r'):
            q1, q2 = line.strip().split('\t')
            # replace all numbers to "1"
            # to lower case
            # q1_tokens, q2_tokens = [s.lower().split(' ') for s in [q1, q2]]
            q1_tokens, q2_tokens = [re.sub('\d+(\.\d+)?', '1', s.lower()).split(' ') for s in [q1, q2]]
            # insert sentence start/end symbols
            q1_tokens.insert(0, "[")
            q1_tokens.append("]")
            q2_tokens.insert(0, "[")
            q2_tokens.append("]")

            # paraphrased sentences similarity
            similarity_score = 0.0

            if self.mode == 'str':
                yield (q1_tokens, q2_tokens, similarity_score)
            elif self.mode == 'index':
                # index each word using hash dictionary
                q1_tokens_indx, q2_tokens_indx = [[self.voc_dict[w] for w in s] for s in [q1_tokens, q2_tokens]]
                yield (q1_tokens_indx, q2_tokens_indx, similarity_score)
            else:
                raise AttributeError("Mode can be only 'str' or 'index'")

    def __len__(self):
        if self.usage == 'train':
            return 4077
        elif self.usage == 'test':
            return 1726

    def __str__(self):
        return "PPDB"


# question answer pairs
class QAs(object):
    def __init__(self, usage='train', mode='str', voc_dict=None):
        if usage in ['train', 'test', 'dev']:
            self.file = './data/WikiQA-%s.txt' % usage
            self.usage = usage
        else:
            raise SystemError("usage can be only train/test/dev")
        self.voc_dict = voc_dict
        if mode == 'index':
            assert voc_dict is not None, "must take vocabulary-index dictionary."
        self.mode = mode
        # column number for each iteration
        self.param_num = 2
        # index of return data contains sentence
        self.sent_indx = (0, 1)

    def __iter__(self):
        for line in open(self.file, 'r'):
            q, a, label = line.strip().split('\t')
            q_tokens, a_tokens = [re.sub('\d+(\.\d+)?', '1', s.lower()).split(' ') for s in [q, a]]
            # insert sentence start/end symbols
            q_tokens.insert(0, "[")
            q_tokens.append("]")
            a_tokens.insert(0, "[")
            a_tokens.append("]")

            if self.mode == 'str':
                yield (q_tokens, a_tokens)
            elif self.mode == 'index':
                # index each word using hash dictionary
                q_tokens_indx, a_tokens_indx = [[self.voc_dict[w] for w in s] for s in [q_tokens, a_tokens]]
                yield (q_tokens_indx, a_tokens_indx)
            else:
                raise AttributeError("Mode can be only 'str' or 'index'")

    def __len__(self):
        if self.usage == 'train':
            return 20360
        elif self.usage == 'test':
            return 6165
        elif self.usage == 'dev':
            return 2733

    def __str__(self):
        return "QApairs"


# British National Corpus
class BNCembedding(object):
    def __init__(self):
        self.file = './data/embeddings.txt'

    def __iter__(self):
        for line in open(self.file, 'r'):
            w, *emb = line.strip().split()
            yield w, emb