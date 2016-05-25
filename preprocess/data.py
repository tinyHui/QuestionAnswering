from gensim.models import word2vec
from collections import UserList, defaultdict
import re
import pickle as pkl
import logging
import sys
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

PPDB_TEXT_RAW = '../data/wikianswers-paraphrases-1.0/word_alignments.txt'
PPDB_TEXT = '../data/wikianswers-paraphrases-1.0/word_alignments_processed.txt'
GOOGLE_WORD2VEC = '../data/GoogleNews-vectors-negative300.bin'
WORD_INDX_HASH = '../data/word_indx_hash.pkl'
UPDATE_FREQ = 200


class ppdb(object):
    def __init__(self):
        self.file = PPDB_TEXT_RAW
        self.word2vec_model = word2vec.Word2Vec.load_word2vec_format(GOOGLE_WORD2VEC, binary=True)

    def __iter__(self):
        for line in open(self.file, 'r'):
            tmp = line.strip().split('\t')
            q1, q2, aligns = tmp
            # replace all numbers to "1"
            q1_tokens, q2_tokens = [re.sub('\d+(\.\d+)?', '1', s).split(' ') for s in [q1, q2]]
            tokens_align = aligns.split(' ')
            q1_manual = []
            q2_manual = []
            for align in tokens_align:
                l_indx, r_indx = [int(i) for i in align.split('-')]
                q1_manual.append(q1_tokens[l_indx])
                q2_manual.append(q2_tokens[r_indx])

            # calculate sentence similarity via average of each aligned word
            similarity_score = 0.0
            for w1, w2 in zip(q1_manual, q2_manual):
                try:
                    similarity_score += self.word2vec_model.similarity(w1, w2)
                except KeyError:
                    # most of these words are stop words
                    similarity_score += 1.0
            similarity_score /= len(tokens_align)

            unique_tokens = set(q1_tokens + q2_tokens)
            # insert sentence start/end symbols
            # do not re-add these two symbols into unique_tokens
            q1_tokens.insert(0, "[")
            q1_tokens.append("]")
            q2_tokens.insert(0, "[")
            q2_tokens.append("]")
            yield (q1_tokens, q2_tokens, similarity_score, unique_tokens)

    def __len__(self):
        return 35291309

class ppdb_indx(object):
    def __init__(self):
        self.file = PPDB_TEXT
        with open(WORD_INDX_HASH, 'rb') as f:
            self.dictionary = pkl.load(f)

    def __iter__(self):
        for line in open(self.file, 'r'):
            tmp = line.strip().split('\t')
            q1, q2, score = tmp
            q1_tokens, q2_tokens = [s.split(' ') for s in [q1, q2]]
            q1_tokens_indx, q2_tokens_indx = [[self.dictionary[w] for w in s] for s in [q1_tokens, q2_tokens]]
            yield q1_tokens_indx, q2_tokens_indx, float(score)

    def __len__(self):
        return 35291309


def text2indx(clean_up=False):
    if not clean_up and os.path.exists(WORD_INDX_HASH):
        logging.info("File trained, skip")
        return

    # remove previous processed files
    logging.info("Clean up")
    for f in [PPDB_TEXT, WORD_INDX_HASH]:
        if os.path.exists(f):
            os.remove(f)

    logging.info("Generating token dictionary")
    data = ppdb()
    # pre-add sentence start/end symbols
    token_list = UserList(["[", "]"])

    chunk = []
    for i, (q1_tokens, q2_tokens, similarity_score, tokens) in enumerate(data):
        token_list += tokens
        # write to mem-cache
        chunk.append("{0}\t{1}\t{2}\n".format(" ".join(q1_tokens), " ".join(q2_tokens), similarity_score))
        # give a process update each 500 items
        if i % UPDATE_FREQ == 0 or i == (len(data) - 1):
            sys.stdout.write("\r%d/%d lines have processed" % (i+1, len(data)))
            sys.stdout.flush()
            # write to disk
            with open(PPDB_TEXT, 'a') as f:
                for l in chunk:
                    f.write(l)
                del chunk[:]
    del chunk

    # generate dictionary
    unique_token_list = set(token_list)
    print("\nFound %d tokens" % len(unique_token_list))
    word_indx_hash = defaultdict(int)
    for i, token in enumerate(unique_token_list):
        word_indx_hash[token] = i

    logging.info("Saving word index hashing table")
    with open(WORD_INDX_HASH, 'wb') as f:
        pkl.dump(word_indx_hash, f)

    logging.info("Free up memory")
    del data
    del token_list
    del unique_token_list
    del word_indx_hash