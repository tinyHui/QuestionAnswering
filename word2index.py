from collections import defaultdict
from preprocess.data import ReVerbPairs, UNKNOWN_TOKEN, UNKNOWN_TOKEN_INDX
import logging
import pickle as pkl
import os
import sys

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
VOC_DICT_FILE = './bin/word_indx_hash.pkl'
LOWEST_FREQ = 3


if __name__ == "__main__":
    if os.path.exists(VOC_DICT_FILE):
        logging.info("Word index file exists, skip")
        sys.exit(0)

    logging.info("Generating source data")
    # data is a group of sentences
    src_data = ReVerbPairs(usage='train', mode='str')

    logging.info("Extracting tokens")
    logging.warning("Ignore tokens appears less than %d" % LOWEST_FREQ)
    token_count_group = {}
    token_group = defaultdict(list)
    for line in src_data:
        for i in src_data.sent_indx:
            for w in line[i]:
                try:
                    if token_count_group[i][w] > LOWEST_FREQ:
                        token_group[i].append(w)
                    token_count_group[i][w] += 1
                except KeyError:
                    token_count_group[i] = defaultdict(int)
                    token_count_group[i][w] += 1

    logging.info("Generating token dictionary")
    word_indx_hash_group = {}
    for i in src_data.sent_indx:
        token_list = token_group[i]
        word_indx_hash_group[i] = zip(token_list, range(1, len(token_list) + 1))
        word_indx_hash_group[UNKNOWN_TOKEN] = UNKNOWN_TOKEN_INDX
        logging.info("Found %d tokens in %d column" % len(token_list))

    logging.info("Saving word index hashing table")
    with open(VOC_DICT_FILE, 'wb') as f:
        pkl.dump(word_indx_hash_group, f)

    logging.info("Free up memory")
    del word_indx_hash_group
