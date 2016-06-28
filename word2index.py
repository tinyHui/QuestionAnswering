from collections import defaultdict, UserList, UserDict
from preprocess.data import ReVerbPairs, UNKNOWN_TOKEN, UNKNOWN_TOKEN_INDX
import logging
import pickle as pkl
import os
import sys

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
VOC_DICT_FILE = './bin/word_indx_hash.pkl'
LOWEST_FREQ = 4


def generate_dictionary(token_list):
    # generate dictionary
    token_list.insert(UNKNOWN_TOKEN_INDX, UNKNOWN_TOKEN)
    unique_token_list = set(token_list)
    word_indx_hash = UserDict(zip(unique_token_list, range(len(unique_token_list))))
    logging.info("Found %d tokens" % len(unique_token_list))
    del unique_token_list

    return word_indx_hash


if __name__ == "__main__":
    if os.path.exists(VOC_DICT_FILE):
        logging.info("Word index file exists, skip")
        sys.exit(0)

    logging.info("Generating source data")
    # data is a group of sentences
    src_data = ReVerbPairs(usage='train', mode='str')

    logging.info("Extracting tokens")
    logging.warning("Ignore tokens appears less than %d" % LOWEST_FREQ)
    word_indx_hash_group = {}
    for i in src_data.sent_indx:
        token_list = UserList()
        token_count = defaultdict(int)
        for line in src_data:
            for token in line[i]:
                if token_count[token] >= LOWEST_FREQ:
                    token_list.append(token)
                else:
                    token_count += 1
        word_indx_hash_group[i] = generate_dictionary(token_list)
        logging.info("Found %d tokens" % len(token_list))
        del token_list[:]

    logging.info("Saving word index hashing table")
    with open(VOC_DICT_FILE, 'wb') as f:
        pkl.dump(word_indx_hash_group, f)

    logging.info("Free up memory")
    del word_indx_hash_group
