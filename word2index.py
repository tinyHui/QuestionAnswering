from collections import defaultdict
from preprocess.data import ReVerbPairs, UNKNOWN_TOKEN, UNKNOWN_TOKEN_INDX
import logging
import pickle as pkl
import os
import sys

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
VOC_DICT_FILE = './bin/word_indx_hash.pkl'
LOWEST_FREQ = 4


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
    line_num = 0
    for i in src_data.sent_indx:
        token_count_group[i] = defaultdict(int)
        for line in src_data:
            for token in line[i]:
                sys.stdout.write("\rLoad: %d/%d" % (line_num, len(src_data)))
                # check if the token appears count reach requirement
                if token_count_group[i][token] > LOWEST_FREQ:
                    token_group[i].append(token)
                else:
                    token_count_group[i][token] += 1

    logging.info("Generating token dictionary")
    word_indx_hash_group = {}
    for i in src_data.sent_indx:
        # unique
        token_list = list(set(token_group[i]))
        # add unknown token
        token_group[i].insert(UNKNOWN_TOKEN_INDX, UNKNOWN_TOKEN)
        # generate index
        word_indx_hash_group[i] = dict(zip(token_list, range(len(token_list))))
        logging.info("Found %d tokens" % len(token_list))

    logging.info("Saving word index hashing table")
    with open(VOC_DICT_FILE, 'wb') as f:
        pkl.dump(word_indx_hash_group, f)

    logging.info("Free up memory")
    del word_indx_hash_group
